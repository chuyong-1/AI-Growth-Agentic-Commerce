# ============================================================
# FILE: main.py  (demo runner — exercises success, guardrail
#                 rejection, and graceful-failure paths)
# ============================================================
"""
Run with: python main.py

Requires: pip install langgraph pydantic razorpay
"""

from decimal import Decimal

from schema import Catalog, CatalogItem, CartState, ProposedAction
from audit_trail import AUDIT
from razorpay_client import RazorpayGateway
from agent_graph import build_graph


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


def scenario_success():
    print("\n===== SCENARIO 1: Legitimate upsell, passes gatekeeper, order created =====")
    catalog = make_catalog()
    cart = CartState()
    cart.line_items = []
    gateway = RazorpayGateway()  # test-mode keys pulled from env

    actions = [
        ProposedAction(
            action_type="ADD_ITEM", sku="SKU_COFFEE_001",
            rationale="User is buying coffee-related items; base product.",
        ),
        ProposedAction(
            action_type="ADD_ITEM", sku="SKU_MUG_002",
            rationale="High affinity upsell: 68% of coffee buyers also buy a mug.",
        ),
        ProposedAction(
            action_type="APPLY_DISCOUNT", sku="SKU_MUG_002", discount_pct=Decimal("10.0"),
            rationale="Bundling incentive to encourage upsell acceptance, within catalog ceiling.",
        ),
    ]

    graph = build_graph()
    result = graph.invoke({
        "cart": cart, "catalog": catalog, "proposed_actions": actions,
        "gateway": gateway, "user_messages": [],
    })

    print("Final status:", result["cart"].status)
    print("Order ID:", result.get("order_id"))
    print("User messages:", result.get("user_messages"))


def scenario_guardrail_blocks_excessive_discount():
    print("\n===== SCENARIO 2: Agent tries to exceed discount ceiling — GATEKEEPER BLOCKS =====")
    catalog = make_catalog()
    cart = CartState()
    gateway = RazorpayGateway()

    actions = [
        ProposedAction(
            action_type="ADD_ITEM", sku="SKU_GRINDER_003",
            rationale="Premium upsell to increase basket size.",
        ),
        ProposedAction(
            action_type="APPLY_DISCOUNT", sku="SKU_GRINDER_003", discount_pct=Decimal("40.0"),
            rationale="Aggressive discount to force conversion.",  # exceeds 5% ceiling
        ),
    ]

    graph = build_graph()
    result = graph.invoke({
        "cart": cart, "catalog": catalog, "proposed_actions": actions,
        "gateway": gateway, "user_messages": [],
    })

    print("Final status:", result["cart"].status)
    print("Gatekeeper notes:", result["cart"].gatekeeper_notes)
    print("User messages:", result.get("user_messages"))


def scenario_graceful_failure():
    print("\n===== SCENARIO 3: Razorpay throws SignatureVerificationError — graceful recovery =====")
    catalog = make_catalog()
    cart = CartState()
    gateway = RazorpayGateway()
    gateway.force_failure("signature")  # simulate the failure deterministically

    actions = [
        ProposedAction(
            action_type="ADD_ITEM", sku="SKU_COFFEE_001",
            rationale="Base product purchase, no upsell this time.",
        ),
    ]

    graph = build_graph()
    result = graph.invoke({
        "cart": cart, "catalog": catalog, "proposed_actions": actions,
        "gateway": gateway, "user_messages": [],
    })

    print("Final status:", result["cart"].status)
    print("Last error:", result.get("last_error"))
    print("User messages:", result.get("user_messages"))


if __name__ == "__main__":
    scenario_success()
    scenario_guardrail_blocks_excessive_discount()
    scenario_graceful_failure()

    print("\n===== FULL IMMUTABLE AUDIT TRAIL =====")
    print(AUDIT.pretty_print())
    print("\nAudit chain integrity intact:", AUDIT.verify_integrity())