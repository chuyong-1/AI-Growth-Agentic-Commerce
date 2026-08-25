# ============================================================
# FILE: api.py
# ============================================================
"""
AgenticTrade — FastAPI backend.

Zero-cloud, single-process design: every store below (session, audit,
campaign budget) is an in-memory, thread-safe singleton. This trades
cross-restart durability and multi-instance scaling for a genuinely
zero-config `uvicorn api:app --reload` — no AWS account, no table
provisioning, no environment variables required to run.

Run with:
    uvicorn api:app --reload
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.messages import messages_from_dict, messages_to_dict

from schema import Catalog, CatalogItem, CartState, CartStatus, ProposedAction
from audit_trail import AUDIT
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

SESSION_STORE = get_session_store()
CAMPAIGN_STORE = get_campaign_store()

DEFAULT_CAMPAIGN_BUDGET = DurableCampaignBudget(period_label="default-period")

# Soft cap purely to give /api/health a meaningful memory-footprint
# signal in a long-running local demo — this store has no TTL/eviction
# (deliberately: it's an in-memory demo store, not a production
# cache), so a multi-day-running instance will accumulate carts
# indefinitely. This constant documents that tradeoff rather than
# silently hiding it.
SESSION_STORE_SOFT_LIMIT_WARNING = 5000

app = FastAPI(title="AgenticTrade — Growth & Agentic Commerce Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# ==================================================================
# Global error boundaries — see docstring notes below each handler
# ==================================================================
@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """
    FastAPI/Pydantic returns 422 for malformed bodies by default, but
    the default error shape is a nested list of per-field errors —
    awkward for a frontend to render as a single toast. This
    normalizes it to a flat `detail` string while still logging the
    full structured error server-side for debugging.
    """
    first_error = exc.errors()[0] if exc.errors() else {}
    field_path = ".".join(str(p) for p in first_error.get("loc", []))
    logger.warning("Request validation failed on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "detail": f"Invalid request body"
                      + (f" (field: {field_path})" if field_path else "")
                      + f" — {first_error.get('msg', 'malformed payload')}",
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """
    Last-resort boundary: any bug anywhere in the LangGraph/LLM/store
    stack returns a clean 500 instead of an unhandled stack trace
    leaking to the client or destabilizing the worker process.
    Deliberately does NOT re-raise — catching here and returning a
    normal HTTP response keeps the worker alive for the next request.
    """
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. No charge was made and no state was corrupted."},
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _load_or_create_cart(cart_id: Optional[str]) -> tuple[CartState, int]:
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
    another's writes. After max_attempts, surfaces a 409 rather than
    retrying forever — an infinite retry loop under sustained
    contention would hang the request.
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
    gateway = RazorpayGateway()
    if requested_failure:
        gateway.force_failure(requested_failure)
    else:
        armed = SESSION_STORE.get_failure_injection(cart_id)
        if armed:
            gateway.force_failure(armed)
    return gateway


def _safe_graph_invoke(graph_input: dict, cart_id: str) -> dict:
    """
    Defensive boundary around GRAPH.invoke(). Two failure classes:

      1. Any exception raised inside a graph node that wasn't already
         caught by that node's own try/except (agent_graph.py's
         create_order_node handles Razorpay-specific errors, including
         rate limits, directly — this is the belt-and-suspenders
         backstop for anything unanticipated).
      2. A malformed/incomplete GraphState returned from invoke()
         (should be structurally impossible given the graph's own
         typing — checked anyway, so the boundary between "trusted
         internal graph" and "response serialization" isn't just
         assumed).

    On any failure, nothing here mutates the caller's cart object —
    upsell_agent_node and payment_gatekeeper_node only mutate a cart
    that's passed by reference and always leave it in a clean
    terminal status before any exception could reach this wrapper.
    """
    try:
        result = GRAPH.invoke(graph_input)
    except Exception as e:
        logger.exception("Graph invocation failed for cart=%s", cart_id)
        AUDIT.log(
            "GRAPH_INVOCATION_FAILED",
            cart_id,
            {"error_type": type(e).__name__, "error": str(e)},
        )
        raise HTTPException(
            status_code=502,
            detail="The checkout engine encountered an unexpected error. No charge was made.",
        ) from e

    if "cart" not in result:
        logger.error("Graph result missing 'cart' key for cart=%s: %s", cart_id, result)
        raise HTTPException(status_code=500, detail="Checkout engine returned an incomplete result.")

    return result


# ==================================================================
# POST /api/chat — conversational in-app checkout
# ==================================================================
class ChatRequest(BaseModel):
    cart_id: Optional[str] = None
    message: str = Field(min_length=1, max_length=4000)
    simulate_failure: Optional[str] = None


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
    if req.simulate_failure and req.simulate_failure not in ("timeout", "signature"):
        raise HTTPException(
            status_code=400,
            detail="simulate_failure must be 'timeout', 'signature', or omitted",
        )

    cart, version = _load_or_create_cart(req.cart_id)
    history = _load_history(req.cart_id)
    gateway = _resolve_gateway(cart.cart_id, req.simulate_failure)

    AUDIT.log("USER_MESSAGE", cart.cart_id, {"message": req.message})

    try:
        assistant_text, proposed_actions, updated_history = AGENT.run_turn(
            history=history, user_message=req.message
        )
    except Exception as e:
        logger.exception("LLM turn failed for cart=%s", cart.cart_id)
        AUDIT.log("LLM_TURN_FAILED", cart.cart_id, {"error": str(e)})
        raise HTTPException(status_code=502, detail=f"Assistant is temporarily unavailable: {e}") from e

    system_notes: list[str] = []
    order_id: Optional[str] = None

    if proposed_actions:
        result = _safe_graph_invoke(
            {
                "cart": cart,
                "catalog": CATALOG,
                "proposed_actions": proposed_actions,
                "gateway": gateway,
                "user_messages": [],
            },
            cart.cart_id,
        )
        cart = result["cart"]
        order_id = result.get("order_id")
        system_notes.extend(result.get("user_messages", []))

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


class ArmFailureRequest(BaseModel):
    cart_id: str = Field(min_length=1)
    mode: str


@app.post("/api/chat/arm-failure")
def arm_failure(req: ArmFailureRequest):
    if req.mode not in ("timeout", "signature"):
        raise HTTPException(status_code=400, detail="mode must be 'timeout' or 'signature'")
    SESSION_STORE.set_failure_injection(req.cart_id, req.mode)
    return {"cart_id": req.cart_id, "armed_failure": req.mode}


class AgentProposeRequest(BaseModel):
    cart_id: Optional[str] = None
    actions: list[ProposedAction] = Field(min_length=1, max_length=20)
    agent_identity: Optional[str] = Field(default=None, max_length=200)


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

    result = _safe_graph_invoke(
        {
            "cart": cart,
            "catalog": CATALOG,
            "proposed_actions": req.actions,
            "gateway": gateway,
            "user_messages": [],
        },
        cart.cart_id,
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


@app.get("/.well-known/agentic-commerce.json")
def well_known_manifest(request: Request):
    return build_well_known_manifest(_base_url(request))


@app.get("/api/catalog/feed")
def catalog_feed():
    return build_catalog_feed(CATALOG)


@app.post("/api/campaigns/run")
def run_campaigns(period_label: Optional[str] = None):
    if period_label is not None and not period_label.strip():
        raise HTTPException(status_code=400, detail="period_label, if provided, cannot be blank")

    budget = DurableCampaignBudget(
        period_label=period_label or DEFAULT_CAMPAIGN_BUDGET.period_label,
        max_discount_spend=DEFAULT_CAMPAIGN_BUDGET.max_discount_spend,
        max_concurrent_campaigns=DEFAULT_CAMPAIGN_BUDGET.max_concurrent_campaigns,
        max_single_campaign_discount_pct=DEFAULT_CAMPAIGN_BUDGET.max_single_campaign_discount_pct,
    )
    CAMPAIGN_STORE.ensure_budget(budget)

    try:
        decided = run_campaign_cycle_durable(CATALOG, budget, CAMPAIGN_STORE)
    except Exception as e:
        logger.exception("Campaign cycle failed")
        raise HTTPException(status_code=500, detail=f"Campaign cycle failed: {e}") from e

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
        "remaining_budget": str(state["max_discount_spend"] - state["committed_spend"]),
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


class VerifyRequest(BaseModel):
    cart_id: str = Field(min_length=1)
    razorpay_order_id: str = Field(min_length=1)
    razorpay_payment_id: str = Field(min_length=1)
    razorpay_signature: str = Field(min_length=1)


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
        AUDIT.log("SIGNATURE_VERIFICATION_ORDER_MISMATCH", req.cart_id, {
            "expected": cart.razorpay_order_id, "got": req.razorpay_order_id,
        })
        raise HTTPException(status_code=400, detail="order_id does not match this cart's active order")

    gateway = RazorpayGateway()
    try:
        verified = gateway.verify_payment_signature({
            "razorpay_order_id": req.razorpay_order_id,
            "razorpay_payment_id": req.razorpay_payment_id,
            "razorpay_signature": req.razorpay_signature,
        })
    except Exception as e:
        # Fail closed, not open: any exception in the verification
        # step itself is treated as "not verified", never as verified.
        logger.exception("Signature verification raised for cart=%s", req.cart_id)
        AUDIT.log("PAYMENT_VERIFICATION_ERROR", req.cart_id, {"error": str(e)})
        verified = False

    if verified:
        cart.status = CartStatus.COMPLETED
        AUDIT.log("PAYMENT_VERIFIED", req.cart_id, {
            "payment_id": req.razorpay_payment_id, "order_id": req.razorpay_order_id,
        })
    else:
        cart.status = CartStatus.PAYMENT_FAILED
        AUDIT.log("PAYMENT_VERIFICATION_FAILED", req.cart_id, {
            "payment_id": req.razorpay_payment_id, "order_id": req.razorpay_order_id,
        })

    _persist_cart_with_retry(cart, version)

    return VerifyResponse(cart_id=req.cart_id, verified=verified, cart_status=cart.status.value)


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
    return {
        "status": "ok",
        "storage": "in-memory (thread-safe, single-process)",
        "cart_count": SESSION_STORE.cart_count(),
        "audit_entry_count": AUDIT.entry_count(),
        "memory_note": (
            "In-memory store has no eviction; for long-running demo "
            "instances beyond a few thousand carts, restart the process."
        ) if SESSION_STORE.cart_count() > SESSION_STORE_SOFT_LIMIT_WARNING else None,
    }