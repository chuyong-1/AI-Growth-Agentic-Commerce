# ============================================================
# FILE: main.py  (demo runner — exercises checkout, agent
#                 propose, and campaign lifecycle scenarios)
# ============================================================
"""
Run with: python main.py

Requires: see requirements.txt
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
    """
    Scenario 4: an EXTERNAL AI shopping agent (not our conversational
    UI) proposes a valid order directly, using the exact same graph
    that backs POST /api/agent/propose in api.py. Demonstrates that
    the machine-to-machine entrypoint gets identical gatekeeper
    enforcement — no separate, weaker code path for agent traffic.
    """
    print("\n===== SCENARIO 4: External AI buyer agent proposes via /api/agent/propose path =====")
    catalog = make_catalog()
    cart = CartState()
    gateway = RazorpayGateway()

    # This mirrors what an external agent would submit after reading
    # the /.well-known/agentic-commerce.json manifest and the catalog
    # feed's published discount ceilings.
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


def scenario_campaign_cycle_mixed_outcomes():
    """
    Scenario 5: a full campaign growth cycle against a fresh budget
    period, showing at least one campaign approved (fits under the
    ceiling) and one rejected (exceeds the per-campaign discount cap
    or would blow the period ceiling).

    NOTE: this exercises campaign_store.CampaignBudgetStore directly.
    The full LLM-driven campaign_orchestrator (agents/campaign_orchestrator.py)
    wraps this same store with its own gatekeeper node, analogous to
    agent_graph.py's PaymentGatekeeper.
    """
    print("\n===== SCENARIO 5: Campaign cycle — one approved, one rejected =====")
    from campaign_store import CampaignBudgetStore, DurableCampaignBudget, BudgetExceededError

    store = CampaignBudgetStore()
    budget = DurableCampaignBudget(
        period_label="demo-period-1",
        max_discount_spend=Decimal("5000.00"),
        max_concurrent_campaigns=3,
    )
    store.ensure_budget(budget)

    campaign_a_amount = Decimal("2000.00")  # fits comfortably
    try:
        state = store.try_reserve(
            "demo-period-1", campaign_a_amount,
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        print(f"Campaign A APPROVED — reserved ₹{campaign_a_amount}, "
              f"committed_spend now ₹{state['committed_spend']}")
    except BudgetExceededError as e:
        print(f"Campaign A unexpectedly rejected: {e}")

    campaign_b_amount = Decimal("4000.00")  # would push total to 6000 > 5000 ceiling
    try:
        store.try_reserve(
            "demo-period-1", campaign_b_amount,
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        print(f"Campaign B unexpectedly APPROVED — this should not happen")
    except BudgetExceededError as e:
        print(f"Campaign B REJECTED (expected): {e}")

    final_state = store.get_state("demo-period-1")
    print(f"Final committed_spend for demo-period-1: ₹{final_state['committed_spend']} "
          f"(ceiling ₹{budget.max_discount_spend})")


def scenario_campaign_cycle_budget_depletion():
    """
    Scenario 6: runs a SECOND campaign cycle immediately after
    Scenario 5, against the SAME period, demonstrating that a
    campaign which would have been perfectly affordable in isolation
    (well under the raw ceiling) is now blocked purely because prior
    reservations already consumed the remaining headroom.
    """
    print("\n===== SCENARIO 6: Second cycle — prior spend blocks an otherwise-affordable campaign =====")
    from campaign_store import CampaignBudgetStore, DurableCampaignBudget, BudgetExceededError

    store = CampaignBudgetStore()
    budget = DurableCampaignBudget(
        period_label="demo-period-1",  # same period as scenario 5 — budget already partially spent
        max_discount_spend=Decimal("5000.00"),
        max_concurrent_campaigns=3,
    )
    store.ensure_budget(budget)  # no-op, already initialized by scenario 5

    remaining_before = store.remaining_budget("demo-period-1")
    print(f"Remaining budget entering this cycle: ₹{remaining_before}")

    # This amount (₹2500) would be well under the FULL ₹5000 ceiling in
    # isolation, but Scenario 5 already committed ₹2000, leaving only
    # ₹3000 — still enough here, so first show it succeeding...
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

    # ...then show a campaign that WOULD be affordable against the full
    # ₹5000 ceiling (₹800 <<< ₹5000) but is now blocked because only
    # scraps of headroom remain after A and C.
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
    scenario_campaign_cycle_mixed_outcomes()
    scenario_campaign_cycle_budget_depletion()

    print("\n===== FULL IMMUTABLE AUDIT TRAIL (in-memory demo trail) =====")
    print(AUDIT.pretty_print())
    print("\nAudit chain integrity intact:", AUDIT.verify_integrity())