# ============================================================
# FILE: main.py  (demo runner — checkout, agent-propose, and
#                 campaign lifecycle scenarios, zero-cloud)
# ============================================================
"""
Run with: python main.py

Fully offline: no cloud account, no network access, and no environment
variables required. Every store this script touches (audit trail,
campaign budget) is an in-memory singleton, and Razorpay runs in
simulation mode unless real test credentials are configured.
"""

import sys
from decimal import Decimal

from schema import Catalog, CatalogItem, CartState, ProposedAction
from audit_trail import AUDIT
from razorpay_client import RazorpayGateway
from agent_graph import build_graph
from campaign_store import CampaignBudgetStore, DurableCampaignBudget, BudgetExceededError

# Output contains ₹ and em dashes; Windows consoles default to cp1252
# and raise UnicodeEncodeError mid-run rather than substituting.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
    gateway = RazorpayGateway()

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
            rationale="Aggressive discount to force conversion.",
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
    gateway.force_failure("signature")

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


def scenario_external_agent_propose():
    print("\n===== SCENARIO 4: External AI buyer agent proposes via /api/agent/propose path =====")
    catalog = make_catalog()
    cart = CartState()
    gateway = RazorpayGateway()

    actions = [
        ProposedAction(
            action_type="ADD_ITEM", sku="SKU_COFFEE_001",
            rationale="External agent: user asked for their usual filter coffee order.",
        ),
        ProposedAction(
            action_type="APPLY_DISCOUNT", sku="SKU_COFFEE_001", discount_pct=Decimal("10.0"),
            rationale="External agent: applying published max ceiling for loyalty user, per catalog feed.",
        ),
    ]

    graph = build_graph()
    result = graph.invoke({
        "cart": cart, "catalog": catalog, "proposed_actions": actions,
        "gateway": gateway, "user_messages": [],
    })

    print("Final status:", result["cart"].status)
    print("Order ID:", result.get("order_id"))
    print("(This exact graph is what /api/agent/propose invokes server-side.)")


def make_campaign_budget() -> DurableCampaignBudget:
    return DurableCampaignBudget(
        period_label="demo-period-1",
        max_discount_spend=Decimal("5000.00"),
        max_concurrent_campaigns=3,
    )


def scenario_campaign_cycle_mixed_outcomes(store: CampaignBudgetStore):
    print("\n===== SCENARIO 5: Campaign cycle — one approved, one rejected =====")
    budget = make_campaign_budget()
    store.ensure_budget(budget)

    campaign_a_amount = Decimal("2000.00")
    try:
        state = store.try_reserve(
            "demo-period-1", campaign_a_amount,
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        print(f"Campaign A APPROVED — reserved ₹{campaign_a_amount}, "
              f"committed_spend now ₹{state['committed_spend']}")
    except BudgetExceededError as e:
        print(f"Campaign A unexpectedly rejected: {e}")

    campaign_b_amount = Decimal("4000.00")
    try:
        store.try_reserve(
            "demo-period-1", campaign_b_amount,
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        print("Campaign B unexpectedly APPROVED — this should not happen")
    except BudgetExceededError as e:
        print(f"Campaign B REJECTED (expected): {e}")

    final_state = store.get_state("demo-period-1")
    print(f"Final committed_spend for demo-period-1: ₹{final_state['committed_spend']} "
          f"(ceiling ₹{budget.max_discount_spend})")


def scenario_campaign_cycle_budget_depletion(store: CampaignBudgetStore):
    print("\n===== SCENARIO 6: Second cycle — prior spend blocks an otherwise-affordable campaign =====")
    # Deliberately the SAME store and period as Scenario 5: the point
    # of this scenario is that spend committed by an earlier cycle
    # constrains a later one. A fresh store here would reset the
    # ledger and the scenario would silently demonstrate nothing.
    budget = make_campaign_budget()
    store.ensure_budget(budget)

    remaining_before = store.remaining_budget("demo-period-1")
    print(f"Remaining budget entering this cycle: ₹{remaining_before}")

    campaign_c_amount = Decimal("2500.00")
    try:
        state = store.try_reserve(
            "demo-period-1", campaign_c_amount,
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        print(f"Campaign C APPROVED — reserved ₹{campaign_c_amount}, "
              f"committed_spend now ₹{state['committed_spend']}")
    except BudgetExceededError as e:
        print(f"Campaign C REJECTED: {e}")

    campaign_d_amount = Decimal("800.00")
    remaining_now = store.remaining_budget("demo-period-1")
    print(f"Remaining budget before Campaign D: ₹{remaining_now} "
          f"(Campaign D needs ₹{campaign_d_amount}, and would fit under the "
          f"raw ₹{budget.max_discount_spend} ceiling in isolation)")
    try:
        store.try_reserve(
            "demo-period-1", campaign_d_amount,
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        print("Campaign D unexpectedly APPROVED")
    except BudgetExceededError as e:
        print(f"Campaign D REJECTED (expected — budget depleted by prior campaigns): {e}")

    final_state = store.get_state("demo-period-1")
    print(f"Final committed_spend for demo-period-1: ₹{final_state['committed_spend']} "
          f"of ₹{budget.max_discount_spend} ceiling")


if __name__ == "__main__":
    scenario_success()
    scenario_guardrail_blocks_excessive_discount()
    scenario_graceful_failure()
    scenario_external_agent_propose()

    campaign_store = CampaignBudgetStore()
    scenario_campaign_cycle_mixed_outcomes(campaign_store)
    scenario_campaign_cycle_budget_depletion(campaign_store)

    print("\n===== FULL IMMUTABLE AUDIT TRAIL =====")
    print(AUDIT.pretty_print())
    print("\nAudit chain integrity intact:", AUDIT.verify_integrity())