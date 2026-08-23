# ============================================================
# FILE: api.py
# ============================================================
"""
FastAPI backend — fully durable version.

Wires the conversational agent, the compiled LangGraph (upsell ->
gatekeeper -> Razorpay -> recovery), the campaign orchestrator
(revenue growth), the agent-readable catalog feed, and the immutable,
DynamoDB-backed audit trail into HTTP endpoints.

Nothing in this file holds financially-meaningful state in a Python
process global. Cart state, conversation history, gateway
failure-injection flags, the campaign budget, and the audit trail are
all persisted in DynamoDB, so this survives restarts and is safe to
run as multiple concurrent Lambda instances.

Run with:
    uvicorn api:app --reload

Env vars required:
    ANTHROPIC_API_KEY
    RAZORPAY_TEST_KEY_ID
    RAZORPAY_TEST_KEY_SECRET
    AUDIT_TABLE, SESSION_TABLE, CAMPAIGN_TABLE   (default to *-dev-* names)
    AWS_REGION                                    (default us-east-1)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.messages import messages_from_dict, messages_to_dict

from schema import Catalog, CatalogItem, CartState, CartStatus, ProposedAction
from audit_trail_dynamo import get_audit_trail
from razorpay_client import RazorpayGateway
from agent_graph import build_graph
from agents.conversational_agent import ConversationalAgent
from agents.campaign_orchestrator import (
    DurableCampaignBudget,
    CampaignStatus,
    run_campaign_cycle_durable,
)
from campaign_store import get_campaign_store
from session_store import get_session_store, SessionConflictError
from catalog_feed import build_catalog_feed, build_well_known_manifest

logger = logging.getLogger("agentictrade.api")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------
# Demo catalog — swap for a DB-backed lookup in real production.
# ------------------------------------------------------------------
def make_catalog() -> Catalog:
    return Catalog(
        items={
            "SKU_COFFEE_001": CatalogItem(
                sku="SKU_COFFEE_001", name="Filter Coffee (250g)",
                base_price=Decimal("399.00"), max_discount_pct=Decimal("10.0"),
            ),
            "SKU_MUG_002": CatalogItem(
                sku="SKU_MUG_002", name="Ceramic Mug",
                base_price=Decimal("249.00"), max_discount_pct=Decimal("15.0"),
                is_upsell_eligible=True,
            ),
            "SKU_GRINDER_003": CatalogItem(
                sku="SKU_GRINDER_003", name="Manual Coffee Grinder",
                base_price=Decimal("1299.00"), max_discount_pct=Decimal("5.0"),
            ),
        }
    )


CATALOG = make_catalog()
GRAPH = build_graph()
AGENT = ConversationalAgent(catalog=CATALOG)

# Durable stores — each is a lazily-constructed singleton (see their
# respective modules), safe to call get_*() repeatedly.
AUDIT = get_audit_trail()
SESSION_STORE = get_session_store()
CAMPAIGN_STORE = get_campaign_store()

DEFAULT_CAMPAIGN_BUDGET = DurableCampaignBudget(period_label="default-period")

app = FastAPI(title="AgenticTrade — Growth & Agentic Commerce Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _load_or_create_cart(cart_id: Optional[str]) -> tuple[CartState, int]:
    """Returns (cart, version). version=0 means brand new / not yet
    persisted, which _persist_cart_with_retry treats as an
    unconditional first write."""
    if cart_id:
        existing = SESSION_STORE.get_cart(cart_id)
        if existing:
            return existing, SESSION_STORE.get_version(cart_id)
        return CartState(cart_id=cart_id), 0
    return CartState(), 0


def _load_history(cart_id: Optional[str]) -> list:
    if not cart_id:
        return []
    raw = SESSION_STORE.get_history(cart_id)
    return messages_from_dict(raw) if raw else []


def _save_history(cart_id: str, history: list) -> None:
    SESSION_STORE.save_history(cart_id, messages_to_dict(history))


def _persist_cart_with_retry(cart: CartState, expected_version: int, max_attempts: int = 3) -> None:
    """
    Optimistic-concurrency save with a small retry loop. Protects
    against two overlapping requests for the same cart_id (e.g. a
    double-submitted checkout click) silently clobbering one
    another's writes — the loser re-reads the current version and
    retries, rather than losing an update invisibly.
    """
    attempt = 0
    version = expected_version
    while True:
        try:
            SESSION_STORE.save_cart(cart, expected_version=version if version > 0 else None)
            return
        except SessionConflictError:
            attempt += 1
            if attempt >= max_attempts:
                raise HTTPException(
                    status_code=409,
                    detail=f"Cart {cart.cart_id} was modified concurrently; please retry.",
                )
            version = SESSION_STORE.get_version(cart.cart_id)
            logger.warning(
                "Session conflict on cart=%s, retrying with version=%s (attempt %s/%s)",
                cart.cart_id, version, attempt, max_attempts,
            )


def _resolve_gateway(cart_id: str, requested_failure: Optional[str]) -> RazorpayGateway:
    """Builds a fresh RazorpayGateway per request (it's a thin,
    stateless-except-for-test-hooks wrapper) and applies either an
    explicitly-requested one-shot failure or a previously-armed one
    persisted from a prior /api/chat call (e.g. via CLI's /fail)."""
    gateway = RazorpayGateway()
    if requested_failure:
        gateway.force_failure(requested_failure)
    else:
        armed = SESSION_STORE.get_failure_injection(cart_id)
        if armed:
            gateway.force_failure(armed)
    return gateway


# ==================================================================
# POST /api/chat — conversational in-app checkout
# ==================================================================
class ChatRequest(BaseModel):
    cart_id: Optional[str] = None
    message: str
    simulate_failure: Optional[str] = None  # "timeout" | "signature" | None — test hook


class ChatResponse(BaseModel):
    cart_id: str
    assistant_message: str
    cart_status: str
    line_items: list[dict]
    computed_total: str
    order_id: Optional[str] = None
    system_notes: list[str] = []


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    cart, version = _load_or_create_cart(req.cart_id)
    history = _load_history(req.cart_id)
    gateway = _resolve_gateway(cart.cart_id, req.simulate_failure)

    AUDIT.log("USER_MESSAGE", cart.cart_id, {"message": req.message})

    try:
        assistant_text, proposed_actions, updated_history = AGENT.run_turn(
            history=history, user_message=req.message
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM turn failed: {e}") from e

    system_notes: list[str] = []
    order_id: Optional[str] = None

    # Only invoke the graph (gatekeeper + Razorpay) if the model actually
    # proposed cart mutations this turn. Pure conversation (e.g. "hi",
    # "what do you have?") shouldn't trigger a checkout attempt.
    if proposed_actions:
        result = GRAPH.invoke(
            {
                "cart": cart,
                "catalog": CATALOG,
                "proposed_actions": proposed_actions,
                "gateway": gateway,
                "user_messages": [],
            }
        )
        cart = result["cart"]
        order_id = result.get("order_id")
        system_notes.extend(result.get("user_messages", []))

    # Clear any one-shot failure injection now that this turn has run,
    # whether it came from this request or a previously-armed CLI /fail.
    gateway.force_failure(None)
    SESSION_STORE.set_failure_injection(cart.cart_id, None)

    _persist_cart_with_retry(cart, version)
    _save_history(cart.cart_id, updated_history)

    return ChatResponse(
        cart_id=cart.cart_id,
        assistant_message=assistant_text,
        cart_status=cart.status.value,
        line_items=[li.model_dump(mode="json") for li in cart.line_items],
        computed_total=str(cart.computed_total),
        order_id=order_id,
        system_notes=system_notes,
    )


# ==================================================================
# POST /api/chat/arm-failure — persist a one-shot failure for the
# NEXT /api/chat call on this cart (durable equivalent of the old
# in-memory "armed" test hook used by cli_chat.py's /fail command)
# ==================================================================
class ArmFailureRequest(BaseModel):
    cart_id: str
    mode: str  # "timeout" | "signature"


@app.post("/api/chat/arm-failure")
def arm_failure(req: ArmFailureRequest):
    if req.mode not in ("timeout", "signature"):
        raise HTTPException(status_code=400, detail="mode must be 'timeout' or 'signature'")
    SESSION_STORE.set_failure_injection(req.cart_id, req.mode)
    return {"cart_id": req.cart_id, "armed_failure": req.mode}


# ==================================================================
# POST /api/agent/propose — direct machine-to-machine entrypoint
# ------------------------------------------------------------------
# Lets an EXTERNAL AI buyer agent (not our own chat UI) submit
# ProposedAction objects directly, still routed through the exact
# same LangGraph (UpsellAgent -> PaymentGatekeeper -> ...). This is
# the "transactable by an AI buyer end to end" surface: no human-
# facing chat turn is required, but no proposal skips the gatekeeper
# either — same guarantees, different front door.
# ==================================================================
class AgentProposeRequest(BaseModel):
    cart_id: Optional[str] = None
    actions: list[ProposedAction]
    agent_identity: Optional[str] = Field(
        default=None,
        description="Free-text identifier for the calling agent, logged to the "
                    "audit trail. Not an auth mechanism in this demo — see the "
                    "'auth' field in /.well-known/agentic-commerce.json.",
    )


class AgentProposeResponse(BaseModel):
    cart_id: str
    cart_status: str
    computed_total: str
    order_id: Optional[str] = None
    system_notes: list[str] = []


@app.post("/api/agent/propose", response_model=AgentProposeResponse)
def agent_propose(req: AgentProposeRequest):
    if not req.actions:
        raise HTTPException(status_code=400, detail="actions must be a non-empty list")

    cart, version = _load_or_create_cart(req.cart_id)
    gateway = _resolve_gateway(cart.cart_id, requested_failure=None)

    AUDIT.log(
        "EXTERNAL_AGENT_PROPOSAL_RECEIVED",
        cart.cart_id,
        {
            "agent_identity": req.agent_identity or "unidentified",
            "action_count": len(req.actions),
            "action_types": [a.action_type for a in req.actions],
        },
    )

    result = GRAPH.invoke(
        {
            "cart": cart,
            "catalog": CATALOG,
            "proposed_actions": req.actions,
            "gateway": gateway,
            "user_messages": [],
        }
    )
    cart = result["cart"]
    _persist_cart_with_retry(cart, version)

    return AgentProposeResponse(
        cart_id=cart.cart_id,
        cart_status=cart.status.value,
        computed_total=str(cart.computed_total),
        order_id=result.get("order_id"),
        system_notes=result.get("user_messages", []),
    )


# ==================================================================
# GET /.well-known/agentic-commerce.json  — agent discovery manifest
# GET /api/catalog/feed                    — agent-readable catalog
# ==================================================================
@app.get("/.well-known/agentic-commerce.json")
def well_known_manifest(request: Request):
    """
    Single discovery entrypoint. An AI buyer agent that only knows
    this store's domain should be able to fetch this ONE url and
    learn where the catalog, proposal, checkout, and audit endpoints
    live — the ACP/UAP-style discovery contract.
    """
    return build_well_known_manifest(_base_url(request))


@app.get("/api/catalog/feed")
def catalog_feed():
    """The machine-readable product feed itself: SKUs, prices, and
    the exact negotiable discount ceilings the gatekeeper enforces."""
    return build_catalog_feed(CATALOG)


# ==================================================================
# Campaign orchestrator endpoints — durable, race-safe budget
# ==================================================================
@app.post("/api/campaigns/run")
def run_campaigns(period_label: Optional[str] = None):
    """
    Triggers one campaign-proposal cycle: gathers growth signals,
    proposes time-boxed discount campaigns, and runs each through the
    deterministic, atomically-budgeted CampaignGatekeeper. In
    production, wire this to a scheduled EventBridge rule instead of
    calling it manually.
    """
    budget = DurableCampaignBudget(
        period_label=period_label or DEFAULT_CAMPAIGN_BUDGET.period_label,
        max_discount_spend=DEFAULT_CAMPAIGN_BUDGET.max_discount_spend,
        max_concurrent_campaigns=DEFAULT_CAMPAIGN_BUDGET.max_concurrent_campaigns,
        max_single_campaign_discount_pct=DEFAULT_CAMPAIGN_BUDGET.max_single_campaign_discount_pct,
    )
    CAMPAIGN_STORE.ensure_budget(budget)

    decided = run_campaign_cycle_durable(CATALOG, budget, CAMPAIGN_STORE)

    return {
        "period": budget.period_label,
        "remaining_budget": str(CAMPAIGN_STORE.remaining_budget(budget.period_label)),
        "campaigns": [c.model_dump(mode="json") for c in decided],
        "activated": sum(1 for c in decided if c.status == CampaignStatus.ACTIVE),
        "rejected": sum(1 for c in decided if c.status == CampaignStatus.REJECTED),
    }


@app.get("/api/campaigns/status")
def campaign_status(period_label: Optional[str] = None):
    label = period_label or DEFAULT_CAMPAIGN_BUDGET.period_label
    try:
        state = CAMPAIGN_STORE.get_state(label)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No budget found for period '{label}'")
    return {
        "period_label": state["period_label"],
        "max_discount_spend": str(state["max_discount_spend"]),
        "committed_spend": str(state["committed_spend"]),
        "remaining_budget": str(
            Decimal(state["max_discount_spend"]) - Decimal(state["committed_spend"])
        ),
        "active_count": int(state["active_count"]),
        "max_concurrent_campaigns": int(state["max_concurrent_campaigns"]),
    }


@app.get("/api/campaigns/audit")
def campaign_audit():
    entries = AUDIT.history_for_cart("campaign_system")
    if not entries:
        raise HTTPException(status_code=404, detail="No campaign audit history yet")
    return {
        "entry_count": len(entries),
        "chain_intact": AUDIT.verify_integrity(),
        "entries": entries,
    }


# ==================================================================
# POST /api/checkout/verify
# ==================================================================
class VerifyRequest(BaseModel):
    cart_id: str
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class VerifyResponse(BaseModel):
    cart_id: str
    verified: bool
    cart_status: str


@app.post("/api/checkout/verify", response_model=VerifyResponse)
def verify_checkout(req: VerifyRequest):
    cart = SESSION_STORE.get_cart(req.cart_id)
    if cart is None:
        raise HTTPException(status_code=404, detail="Unknown cart_id")
    version = SESSION_STORE.get_version(req.cart_id)

    if cart.razorpay_order_id != req.razorpay_order_id:
        AUDIT.log(
            "SIGNATURE_VERIFICATION_ORDER_MISMATCH",
            req.cart_id,
            {"expected": cart.razorpay_order_id, "got": req.razorpay_order_id},
        )
        raise HTTPException(status_code=400, detail="order_id does not match this cart's active order")

    gateway = RazorpayGateway()
    verified = gateway.verify_payment_signature(
        {
            "razorpay_order_id": req.razorpay_order_id,
            "razorpay_payment_id": req.razorpay_payment_id,
            "razorpay_signature": req.razorpay_signature,
        }
    )

    if verified:
        cart.status = CartStatus.COMPLETED
        AUDIT.log(
            "PAYMENT_VERIFIED",
            req.cart_id,
            {"payment_id": req.razorpay_payment_id, "order_id": req.razorpay_order_id},
        )
    else:
        cart.status = CartStatus.PAYMENT_FAILED
        AUDIT.log(
            "PAYMENT_VERIFICATION_FAILED",
            req.cart_id,
            {"payment_id": req.razorpay_payment_id, "order_id": req.razorpay_order_id},
        )

    _persist_cart_with_retry(cart, version)

    return VerifyResponse(
        cart_id=req.cart_id,
        verified=verified,
        cart_status=cart.status.value,
    )


# ==================================================================
# GET /api/audit/{cart_id}
# ==================================================================
@app.get("/api/audit/{cart_id}")
def get_audit_trail_endpoint(cart_id: str):
    entries = AUDIT.history_for_cart(cart_id)
    if not entries:
        raise HTTPException(status_code=404, detail="No audit history for this cart_id")
    return {
        "cart_id": cart_id,
        "entry_count": len(entries),
        "chain_intact": AUDIT.verify_integrity(),
        "entries": entries,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "storage": "dynamodb"}