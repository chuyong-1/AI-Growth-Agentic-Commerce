# ============================================================
# FILE: api.py
# ============================================================
"""
FastAPI backend wiring the conversational agent, the compiled
LangGraph (upsell -> gatekeeper -> Razorpay -> recovery), and the
immutable audit trail into HTTP endpoints.

Run with:
    uvicorn api:app --reload

Env vars required:
    ANTHROPIC_API_KEY
    RAZORPAY_TEST_KEY_ID
    RAZORPAY_TEST_KEY_SECRET
"""

from __future__ import annotations

import os
import uuid
from decimal import Decimal
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schema import Catalog, CatalogItem, CartState, CartStatus
from audit_trail import AUDIT
from razorpay_client import RazorpayGateway
from agent_graph import build_graph
from agents.conversational_agent import ConversationalAgent

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

app = FastAPI(title="Agentic Commerce — Razorpay Upsell & Checkout Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# In-memory session store: cart_id -> session data.
# In production this would be Redis/DB-backed; kept simple here.
# ------------------------------------------------------------------
class Session(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    cart: CartState
    history: list = []  # list of LangChain message objects


SESSIONS: dict[str, Session] = {}
GATEWAYS: dict[str, RazorpayGateway] = {}  # one gateway per session (holds failure-injection state)


def _get_or_create_session(cart_id: Optional[str]) -> Session:
    if cart_id and cart_id in SESSIONS:
        return SESSIONS[cart_id]
    cart = CartState() if not cart_id else CartState(cart_id=cart_id)
    session = Session(cart=cart, history=[])
    SESSIONS[session.cart.cart_id] = session
    GATEWAYS[session.cart.cart_id] = RazorpayGateway()
    return session


# ==================================================================
# POST /api/chat
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
    session = _get_or_create_session(req.cart_id)
    gateway = GATEWAYS[session.cart.cart_id]

    if req.simulate_failure:
        gateway.force_failure(req.simulate_failure)

    AUDIT.log("USER_MESSAGE", session.cart.cart_id, {"message": req.message})

    try:
        assistant_text, proposed_actions, updated_history = AGENT.run_turn(
            history=session.history, user_message=req.message
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM turn failed: {e}") from e

    session.history = updated_history

    system_notes: list[str] = []
    order_id: Optional[str] = None

    # Only invoke the graph (gatekeeper + Razorpay) if the model actually
    # proposed cart mutations this turn. Pure conversation (e.g. "hi",
    # "what do you have?") shouldn't trigger a checkout attempt.
    has_finalize = any(a.action_type == "FINALIZE" for a in proposed_actions)
    has_mutations = any(a.action_type in ("ADD_ITEM", "APPLY_DISCOUNT") for a in proposed_actions)

    if proposed_actions:
        result = GRAPH.invoke(
            {
                "cart": session.cart,
                "catalog": CATALOG,
                "proposed_actions": proposed_actions,
                "gateway": gateway,
                "user_messages": [],
            }
        )
        session.cart = result["cart"]
        order_id = result.get("order_id")
        system_notes.extend(result.get("user_messages", []))

        # If the gatekeeper rejected, give the LLM the chance to explain
        # itself to the user in its own voice on the *next* turn — for
        # now we surface the deterministic rejection directly, since it's
        # a financial-safety message that must not be paraphrased away.
        if session.cart.status == CartStatus.AUDIT_FAILED:
            # Reset cart back to DRAFT-equivalent line items minus the
            # rejected proposal isn't automatic here — a real system would
            # roll back the specific bad line item. For this demo we
            # surface the rejection and leave the cart for correction.
            pass

    gateway.force_failure(None)  # clear one-shot failure injection

    SESSIONS[session.cart.cart_id] = session

    return ChatResponse(
        cart_id=session.cart.cart_id,
        assistant_message=assistant_text,
        cart_status=session.cart.status.value,
        line_items=[li.model_dump(mode="json") for li in session.cart.line_items],
        computed_total=str(session.cart.computed_total),
        order_id=order_id,
        system_notes=system_notes,
    )


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
    if req.cart_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown cart_id")

    session = SESSIONS[req.cart_id]
    gateway = GATEWAYS[req.cart_id]

    if session.cart.razorpay_order_id != req.razorpay_order_id:
        AUDIT.log(
            "SIGNATURE_VERIFICATION_ORDER_MISMATCH",
            req.cart_id,
            {"expected": session.cart.razorpay_order_id, "got": req.razorpay_order_id},
        )
        raise HTTPException(status_code=400, detail="order_id does not match this cart's active order")

    verified = gateway.verify_payment_signature(
        {
            "razorpay_order_id": req.razorpay_order_id,
            "razorpay_payment_id": req.razorpay_payment_id,
            "razorpay_signature": req.razorpay_signature,
        }
    )

    if verified:
        session.cart.status = CartStatus.COMPLETED
        AUDIT.log(
            "PAYMENT_VERIFIED",
            req.cart_id,
            {"payment_id": req.razorpay_payment_id, "order_id": req.razorpay_order_id},
        )
    else:
        session.cart.status = CartStatus.PAYMENT_FAILED
        AUDIT.log(
            "PAYMENT_VERIFICATION_FAILED",
            req.cart_id,
            {"payment_id": req.razorpay_payment_id, "order_id": req.razorpay_order_id},
        )

    SESSIONS[req.cart_id] = session

    return VerifyResponse(
        cart_id=req.cart_id,
        verified=verified,
        cart_status=session.cart.status.value,
    )


# ==================================================================
# GET /api/audit/{cart_id}
# ==================================================================
@app.get("/api/audit/{cart_id}")
def get_audit_trail(cart_id: str):
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
    return {"status": "ok"}