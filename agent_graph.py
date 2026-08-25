# ============================================================
# PATCH: agent_graph.py
# ============================================================
"""
Changes from the previous version:
  1. Dropped BadRequestError / ServerError / GatewayError / RazorpayError
     from the `razorpay.errors` import — this installed SDK version does
     not export them, and importing names that don't exist crashes the
     Uvicorn worker at startup before it ever binds a socket. We only
     import the error classes we've confirmed exist:
     SignatureVerificationError (specific) plus our own
     RazorpayNetworkTimeout / RazorpayOrderMismatchError.
  2. create_order_node no longer relies on a specific RateLimit exception
     type or a broad RazorpayError base class, neither of which we can
     assume are importable/stable across SDK versions. Instead:
       - The pre-existing specific exceptions (RazorpayNetworkTimeout,
         SignatureVerificationError, RazorpayOrderMismatchError) are
         still caught FIRST, by type, exactly as before.
       - A trailing `except Exception` acts as the catch-all backstop
         for anything else the SDK raises (including 429s, which this
         SDK version apparently surfaces as a generic exception rather
         than a dedicated RateLimit class). We sniff the stringified,
         lowercased message for "429" / "rate limit" / "too many
         requests" to route it into the same retryable RateLimit path
         as before; anything that doesn't match falls into a final
         UnclassifiedError branch, logged and handled gracefully rather
         than propagating and taking down the worker.
     NOTE: `except Exception` must come AFTER the specific `except`
     clauses, not before — Python matches except clauses top-to-bottom,
     so a broad Exception handler placed first would silently swallow
     the more specific exceptions below it and make that block dead code.
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


class GraphState(TypedDict, total=False):
    cart: CartState
    catalog: Catalog
    proposed_actions: list[ProposedAction]
    user_messages: list[str]
    last_error: Optional[str]
    order_id: Optional[str]
    gateway: RazorpayGateway


def upsell_agent_node(state: GraphState) -> GraphState:
    cart = state["cart"]
    catalog = state["catalog"]
    actions = state.get("proposed_actions", [])

    for action in actions:
        AUDIT.log("UPSELL_PROPOSED", cart.cart_id, action.model_dump())

        if action.action_type == "ADD_ITEM":
            try:
                item = catalog.get(action.sku)
            except KeyError:
                # Defense in depth: even though the gatekeeper below
                # ALSO independently re-validates every SKU against the
                # catalog, we don't let a hallucinated SKU silently
                # vanish here either — log it explicitly so the audit
                # trail shows exactly what was rejected and why, at
                # the earliest point it was detected.
                AUDIT.log(
                    "UPSELL_REJECTED_UNKNOWN_SKU",
                    cart.cart_id,
                    {"sku": action.sku, "reason": "SKU not found in catalog"},
                )
                continue

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
            for li in cart.line_items:
                if li.sku == action.sku:
                    li.discount_pct = action.discount_pct or Decimal("0.0")
                    li.upsell_rationale = li.upsell_rationale or action.rationale

    cart.declared_total = cart.computed_total
    cart.status = CartStatus.PENDING_AUDIT

    AUDIT.log(
        "CART_SUBMITTED_FOR_AUDIT",
        cart.cart_id,
        {"declared_total": cart.declared_total, "line_items": len(cart.line_items)},
    )

    state["cart"] = cart
    return state


def payment_gatekeeper_node(state: GraphState) -> GraphState:
    cart = state["cart"]
    catalog = state["catalog"]
    violations: list[str] = []

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

    recomputed = cart.computed_total
    if recomputed != cart.declared_total:
        violations.append(
            f"Total mismatch: declared={cart.declared_total} "
            f"recomputed={recomputed} (diff={cart.declared_total - recomputed})"
        )

    if recomputed <= 0:
        violations.append(f"Cart total must be > 0, got {recomputed}")
    if recomputed > Decimal("500000.00"):
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
        cart.declared_total = recomputed
        AUDIT.log("GATEKEEPER_VERDICT_APPROVED", cart.cart_id, verdict_payload)

    state["cart"] = cart
    return state


def gatekeeper_router(state: GraphState) -> Literal["approved", "rejected"]:
    return "approved" if state["cart"].status == CartStatus.AUDITED_OK else "rejected"


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


def create_order_node(state: GraphState) -> GraphState:
    """
    Calls Razorpay ONLY on an AUDITED_OK cart.

    Exception handling, in order of specificity:
      1. RazorpayNetworkTimeout / SignatureVerificationError /
         RazorpayOrderMismatchError — the pre-existing specific
         failure modes, matched by type, unchanged. These MUST be
         caught before the generic handler below or they'd never be
         reached (except clauses match top-to-bottom).
      2. A trailing `except Exception` — a defensive backstop catching
         ANY other exception `gateway.create_order()` raises, since we
         can no longer assume which exception classes this SDK version
         actually exports (BadRequestError / ServerError / RazorpayError
         don't exist here — that's what crashed the worker on import).
         We inspect the exception's stringified message for HTTP 429 /
         rate-limit signatures to route it into the same explicitly
         retryable RateLimit path as before; anything else is logged as
         UnclassifiedError and handled gracefully rather than propagating
         up through the graph and FastAPI into an unhandled 500 (or a
         dead worker).
    """
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

    except Exception as e:
        error_str = str(e).lower()

        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
            cart.status = CartStatus.PAYMENT_FAILED
            state["last_error"] = f"RateLimit: {e}"
            AUDIT.log(
                "RAZORPAY_RATE_LIMITED",
                cart.cart_id,
                {"error_type": type(e).__name__, "error": str(e)},
            )
        else:
            # Ultimate backstop: any exception we didn't anticipate by
            # name or message still lands here rather than propagating
            # uncaught and crashing the FastAPI worker.
            cart.status = CartStatus.PAYMENT_FAILED
            state["last_error"] = f"UnclassifiedError: {e}"
            AUDIT.log(
                "RAZORPAY_ORDER_FAILED_UNCLASSIFIED",
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


def payment_recovery_node(state: GraphState) -> GraphState:
    """
    Graceful failure handling. Three distinct user-facing narratives
    now, up from two — rate limiting gets its own honest message
    rather than being folded into the generic "something went wrong"
    branch, because telling a user "try again in a few seconds" is a
    materially better experience than a vague failure message when
    the underlying cause is just gateway throttling.
    """
    cart = state["cart"]
    error = state.get("last_error", "unknown error")
    msgs = state.setdefault("user_messages", [])

    AUDIT.log("PAYMENT_RECOVERY_ENTERED", cart.cart_id, {"error": error})

    if error.startswith("RateLimit"):
        message = (
            "Our payment gateway is momentarily busy (rate-limited). This is "
            "not an issue with your order — please try checking out again in "
            f"a few seconds. Reference: {cart.cart_id}."
        )
        cart.status = CartStatus.RECOVERED

    elif "SignatureVerification" in error:
        message = (
            "Payment verification failed a security check on our end. "
            "For your protection, I've halted checkout rather than retrying "
            "automatically. No charge was made. Please try again in a moment, "
            f"or contact support with cart reference {cart.cart_id}."
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
        {"approved": "CreateOrderNode", "rejected": "RejectAndExplain"},
    )

    graph.add_conditional_edges(
        "CreateOrderNode",
        order_router,
        {"ok": END, "failed": "PaymentRecovery"},
    )

    graph.add_edge("RejectAndExplain", END)
    graph.add_edge("PaymentRecovery", END)

    return graph.compile()