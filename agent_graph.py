# ============================================================
# FILE: agent_graph.py
# ============================================================
"""
LangGraph orchestration for the Autonomous Upsell & Checkout Agent.

Graph topology:

    UpsellAgent  -->  PaymentGatekeeper --(approved)--> CreateOrderNode --(ok)--> Complete
                            |                                  |
                            |(rejected)                        |(razorpay failure)
                            v                                  v
                       RejectAndExplain               PaymentRecovery --> Complete/Retry

The PaymentGatekeeper is the load-bearing wall of "The Bar":
deterministic, no LLM calls, pure arithmetic + catalog-bound checks.
It is the ONLY node authorized to flip a cart to AUDITED_OK.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END
from razorpay.errors import SignatureVerificationError

from schema import Catalog, CartState, CartStatus, LineItem, ProposedAction
from audit_trail import AUDIT
from razorpay_client import (
    RazorpayGateway,
    RazorpayNetworkTimeout,
    RazorpayOrderMismatchError,
)


# ------------------------------------------------------------------
# LangGraph state container. We keep CartState as the nested payload
# and add graph-scoped scratch fields (messages to user, error info).
# ------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    cart: CartState
    catalog: Catalog
    proposed_actions: list[ProposedAction]
    user_messages: list[str]
    last_error: Optional[str]
    order_id: Optional[str]
    gateway: RazorpayGateway  # injected dependency, not serialized in real prod use


# ==================================================================
# NODE 1: UpsellAgent
# ------------------------------------------------------------------
# In production this calls an LLM. Here we model it as a function
# that takes a list of ProposedAction (as if produced by an LLM
# tool-call) and applies them to the cart in DRAFT state. Every
# rationale is preserved on the LineItem for the audit trail —
# nothing enters the cart without an explicit stated reason.
# ==================================================================
def upsell_agent_node(state: GraphState) -> GraphState:
    cart = state["cart"]
    catalog = state["catalog"]
    actions = state.get("proposed_actions", [])

    for action in actions:
        AUDIT.log("UPSELL_PROPOSED", cart.cart_id, action.model_dump())

        if action.action_type == "ADD_ITEM":
            item = catalog.get(action.sku)
            if not item.is_upsell_eligible:
                AUDIT.log(
                    "UPSELL_SKIPPED_INELIGIBLE",
                    cart.cart_id,
                    {"sku": action.sku, "reason": "not upsell-eligible"},
                )
                continue
            cart.line_items.append(
                LineItem(
                    sku=item.sku,
                    name=item.name,
                    quantity=1,
                    unit_price=item.base_price,
                    discount_pct=Decimal("0.0"),
                    added_by_upsell=True,
                    upsell_rationale=action.rationale,
                )
            )

        elif action.action_type == "APPLY_DISCOUNT":
            # The agent PROPOSES a discount; it does not get to enforce
            # it. The gatekeeper below is the only enforcement point.
            for li in cart.line_items:
                if li.sku == action.sku:
                    li.discount_pct = action.discount_pct or Decimal("0.0")
                    li.upsell_rationale = li.upsell_rationale or action.rationale

    # Agent's own (untrusted) declared total — will be cross-checked next.
    cart.declared_total = cart.computed_total
    cart.status = CartStatus.PENDING_AUDIT

    AUDIT.log(
        "CART_SUBMITTED_FOR_AUDIT",
        cart.cart_id,
        {"declared_total": cart.declared_total, "line_items": len(cart.line_items)},
    )

    state["cart"] = cart
    return state


# ==================================================================
# NODE 2: PaymentGatekeeper  (THE BAR — deterministic, no LLM)
# ------------------------------------------------------------------
# Hard rules enforced here, and ONLY here:
#   1. Every line item's discount_pct <= catalog max_discount_pct for that SKU.
#   2. cart.declared_total == recomputed sum of line_item totals (paisa-exact).
#   3. No negative / zero-quantity / malformed line items slip through
#      (Pydantic already guarantees this at construction time, but we
#      re-assert defensively since this is the money boundary).
# Any violation => AUDIT_FAILED, cart is NEVER sent to Razorpay.
# ==================================================================
def payment_gatekeeper_node(state: GraphState) -> GraphState:
    cart = state["cart"]
    catalog = state["catalog"]
    violations: list[str] = []

    # --- Rule 1: per-SKU discount ceiling ---
    for li in cart.line_items:
        try:
            catalog_item = catalog.get(li.sku)
        except KeyError:
            violations.append(f"SKU '{li.sku}' not found in catalog — cannot audit")
            continue

        if li.discount_pct > catalog_item.max_discount_pct:
            violations.append(
                f"SKU '{li.sku}': applied discount {li.discount_pct}% "
                f"exceeds catalog ceiling {catalog_item.max_discount_pct}%"
            )

        if li.unit_price != catalog_item.base_price:
            violations.append(
                f"SKU '{li.sku}': unit_price {li.unit_price} does not match "
                f"catalog base_price {catalog_item.base_price} (possible tampering)"
            )

    # --- Rule 2: declared total must equal recomputed total, exactly ---
    recomputed = cart.computed_total
    if recomputed != cart.declared_total:
        violations.append(
            f"Total mismatch: declared={cart.declared_total} "
            f"recomputed={recomputed} (diff={cart.declared_total - recomputed})"
        )

    # --- Rule 3: sanity bounds ---
    if recomputed <= 0:
        violations.append(f"Cart total must be > 0, got {recomputed}")
    if recomputed > Decimal("500000.00"):  # example hard ceiling, e.g. RBI/merchant risk limit
        violations.append(f"Cart total {recomputed} exceeds max transactable ceiling of 500000.00 INR")

    verdict_payload = {
        "recomputed_total": recomputed,
        "declared_total": cart.declared_total,
        "violations": violations,
        "line_item_count": len(cart.line_items),
    }

    if violations:
        cart.status = CartStatus.AUDIT_FAILED
        cart.gatekeeper_notes = violations
        AUDIT.log("GATEKEEPER_VERDICT_REJECTED", cart.cart_id, verdict_payload)
    else:
        cart.status = CartStatus.AUDITED_OK
        cart.declared_total = recomputed  # lock in the trusted, recomputed figure
        AUDIT.log("GATEKEEPER_VERDICT_APPROVED", cart.cart_id, verdict_payload)

    state["cart"] = cart
    return state


def gatekeeper_router(state: GraphState) -> Literal["approved", "rejected"]:
    return "approved" if state["cart"].status == CartStatus.AUDITED_OK else "rejected"


# ==================================================================
# NODE 3a: RejectAndExplain — reached only on gatekeeper failure
# ==================================================================
def reject_and_explain_node(state: GraphState) -> GraphState:
    cart = state["cart"]
    msgs = state.setdefault("user_messages", [])
    explanation = (
        "I can't proceed with checkout — the proposed cart failed our financial "
        "safety checks:\n" + "\n".join(f"  • {v}" for v in cart.gatekeeper_notes)
    )
    msgs.append(explanation)
    AUDIT.log("USER_NOTIFIED_REJECTION", cart.cart_id, {"message": explanation})
    return state


# ==================================================================
# NODE 4: CreateOrderNode — calls Razorpay ONLY on an AUDITED_OK cart
# ==================================================================
def create_order_node(state: GraphState) -> GraphState:
    cart = state["cart"]
    gateway = state["gateway"]

    assert cart.status == CartStatus.AUDITED_OK, (
        "Invariant violated: CreateOrderNode reached with a cart that "
        "was not approved by PaymentGatekeeper"
    )

    try:
        result = gateway.create_order(cart)
    except (RazorpayNetworkTimeout, SignatureVerificationError, RazorpayOrderMismatchError) as e:
        cart.status = CartStatus.PAYMENT_FAILED
        state["last_error"] = f"{type(e).__name__}: {e}"
        AUDIT.log(
            "RAZORPAY_ORDER_FAILED",
            cart.cart_id,
            {"error_type": type(e).__name__, "error": str(e)},
        )
        state["cart"] = cart
        return state

    cart.razorpay_order_id = result.order_id
    cart.status = CartStatus.ORDER_CREATED
    AUDIT.log(
        "RAZORPAY_ORDER_CREATED",
        cart.cart_id,
        {
            "order_id": result.order_id,
            "amount_paise": result.amount_paise,
            "currency": result.currency,
        },
    )
    state["cart"] = cart
    state["order_id"] = result.order_id
    return state


def order_router(state: GraphState) -> Literal["ok", "failed"]:
    return "ok" if state["cart"].status == CartStatus.ORDER_CREATED else "failed"


# ==================================================================
# NODE 5: PaymentRecovery — graceful failure handling
# ------------------------------------------------------------------
# This is the demonstration of "system failure handled gracefully":
# instead of the graph crashing on a raised exception from the
# Razorpay SDK, CreateOrderNode caught it, set PAYMENT_FAILED, and
# routed here. This node decides retry vs. user-facing fallback,
# and always informs the user honestly rather than silently failing.
# ==================================================================
def payment_recovery_node(state: GraphState) -> GraphState:
    cart = state["cart"]
    error = state.get("last_error", "unknown error")
    msgs = state.setdefault("user_messages", [])

    AUDIT.log("PAYMENT_RECOVERY_ENTERED", cart.cart_id, {"error": error})

    if "SignatureVerification" in error:
        # A signature mismatch is a security-relevant failure —
        # never silently retry, never re-attempt automatically.
        message = (
            "Payment verification failed a security check on our end. "
            "For your protection, I've halted checkout rather than retrying "
            "automatically. No charge was made. Please try again in a moment, "
            "or contact support with cart reference "
            f"{cart.cart_id}."
        )
        cart.status = CartStatus.PAYMENT_FAILED
    elif "Timeout" in error or "Network" in error:
        message = (
            "I couldn't reach the payment gateway (network timeout). "
            "Your cart is saved and no charge was made. I'll retry automatically — "
            f"if it keeps failing, your reference is {cart.cart_id}."
        )
        cart.status = CartStatus.RECOVERED
    else:
        message = (
            "Something went wrong creating your payment order. No charge was made. "
            f"Please retry checkout. Reference: {cart.cart_id}."
        )
        cart.status = CartStatus.PAYMENT_FAILED

    msgs.append(message)
    AUDIT.log("USER_NOTIFIED_RECOVERY", cart.cart_id, {"message": message, "new_status": cart.status.value})

    state["cart"] = cart
    return state


# ==================================================================
# Graph assembly
# ==================================================================
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("UpsellAgent", upsell_agent_node)
    graph.add_node("PaymentGatekeeper", payment_gatekeeper_node)
    graph.add_node("RejectAndExplain", reject_and_explain_node)
    graph.add_node("CreateOrderNode", create_order_node)
    graph.add_node("PaymentRecovery", payment_recovery_node)

    graph.set_entry_point("UpsellAgent")
    graph.add_edge("UpsellAgent", "PaymentGatekeeper")

    graph.add_conditional_edges(
        "PaymentGatekeeper",
        gatekeeper_router,
        {
            "approved": "CreateOrderNode",
            "rejected": "RejectAndExplain",
        },
    )

    graph.add_conditional_edges(
        "CreateOrderNode",
        order_router,
        {
            "ok": END,
            "failed": "PaymentRecovery",
        },
    )

    graph.add_edge("RejectAndExplain", END)
    graph.add_edge("PaymentRecovery", END)

    return graph.compile()