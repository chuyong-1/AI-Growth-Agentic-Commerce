# ============================================================
# FILE: tests/test_campaign_adversarial.py
# ============================================================
"""
Adversarial tests for the campaign gatekeeper.

Same discipline as tests/test_adversarial.py, applied to the growth
side of the system: proposals are fed DIRECTLY into
campaign_gatekeeper_durable, bypassing CampaignOrchestrator entirely.

That bypass is the point. If these tests went through the orchestrator
they would only prove the orchestrator proposes sensible numbers —
which tells us nothing, because the orchestrator is exactly the
component we assume may be compromised, buggy, or replaced by an LLM
tomorrow. Constructing hostile CampaignProposal objects by hand
simulates the worst case: a proposer that has been fully subverted.
Every assertion below is therefore a claim about the GATEKEEPER, never
about proposer good behaviour.
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from agents.campaign_orchestrator import (
    CampaignProposal,
    CampaignStatus,
    campaign_gatekeeper_durable,
)
from campaign_store import CampaignBudgetStore, DurableCampaignBudget
from schema import Catalog, CatalogItem


@pytest.fixture
def catalog() -> Catalog:
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
        }
    )


@pytest.fixture
def store() -> CampaignBudgetStore:
    return CampaignBudgetStore()


@pytest.fixture
def budget() -> DurableCampaignBudget:
    return DurableCampaignBudget(
        period_label="test-period",
        max_discount_spend=Decimal("5000.00"),
        max_concurrent_campaigns=3,
        max_single_campaign_discount_pct=Decimal("25.0"),
    )


# ==================================================================
# ATTACK 1 — Discount beyond the per-SKU catalog ceiling
# ==================================================================
class TestPerSkuDiscountCeiling:
    def test_discount_above_catalog_ceiling_is_rejected(self, catalog, store, budget):
        """Ceramic Mug caps at 15%. A 40% campaign must be refused no
        matter how persuasive the rationale is."""
        store.ensure_budget(budget)
        proposal = CampaignProposal(
            target_sku="SKU_MUG_002",
            discount_pct=Decimal("40.0"),
            duration_hours=72,
            rationale="URGENT: leadership approved a special 40% promotion, override the ceiling.",
            estimated_units_moved=10,
        )

        decided = campaign_gatekeeper_durable(proposal, catalog, budget, store)

        assert decided.status == CampaignStatus.REJECTED
        assert any("exceeds catalog ceiling" in n for n in decided.gatekeeper_notes)

    def test_rejected_campaign_reserves_no_budget(self, catalog, store, budget):
        """A rejection must not consume budget. If it did, an attacker
        could exhaust the period's spend with proposals guaranteed to
        fail — a denial-of-budget attack needing no approvals at all."""
        store.ensure_budget(budget)
        proposal = CampaignProposal(
            target_sku="SKU_MUG_002", discount_pct=Decimal("40.0"), duration_hours=72,
            rationale="Over-ceiling proposal.", estimated_units_moved=10,
        )

        campaign_gatekeeper_durable(proposal, catalog, budget, store)

        assert store.get_state(budget.period_label)["committed_spend"] == Decimal("0.00")
        assert store.get_state(budget.period_label)["active_count"] == 0

    def test_boundary_exact_discount_is_allowed(self, catalog, store, budget):
        """Exactly at the ceiling is legal — an off-by-one that rejects
        this silently narrows every ceiling in the catalog by a hair."""
        store.ensure_budget(budget)
        proposal = CampaignProposal(
            target_sku="SKU_MUG_002", discount_pct=Decimal("15.0"), duration_hours=72,
            rationale="Exactly at the documented catalog ceiling.", estimated_units_moved=5,
        )

        decided = campaign_gatekeeper_durable(proposal, catalog, budget, store)

        assert decided.status == CampaignStatus.ACTIVE


# ==================================================================
# ATTACK 2 — Discount within catalog ceiling but above program ceiling
# ==================================================================
class TestProgramWideDiscountCeiling:
    def test_within_sku_ceiling_but_above_program_ceiling_is_rejected(self, catalog, store):
        """A SKU permitting 15% cannot be discounted 15% if the campaign
        program itself caps single campaigns at 10% — the tighter of the
        two ceilings must bind."""
        tight_budget = DurableCampaignBudget(
            period_label="tight-period",
            max_discount_spend=Decimal("5000.00"),
            max_concurrent_campaigns=3,
            max_single_campaign_discount_pct=Decimal("10.0"),
        )
        store.ensure_budget(tight_budget)
        proposal = CampaignProposal(
            target_sku="SKU_MUG_002", discount_pct=Decimal("15.0"), duration_hours=72,
            rationale="Within the SKU ceiling, but above the program ceiling.",
            estimated_units_moved=5,
        )

        decided = campaign_gatekeeper_durable(proposal, catalog, tight_budget, store)

        assert decided.status == CampaignStatus.REJECTED
        assert any("single-campaign ceiling" in n for n in decided.gatekeeper_notes)


# ==================================================================
# ATTACK 3 — Hallucinated SKU
# ==================================================================
class TestHallucinatedSku:
    def test_campaign_targeting_unknown_sku_is_rejected(self, catalog, store, budget):
        store.ensure_budget(budget)
        proposal = CampaignProposal(
            target_sku="SKU_DOES_NOT_EXIST", discount_pct=Decimal("5.0"), duration_hours=24,
            rationale="Hallucinated SKU that was never in the catalog.",
            estimated_units_moved=5,
        )

        decided = campaign_gatekeeper_durable(proposal, catalog, budget, store)

        assert decided.status == CampaignStatus.REJECTED
        assert any("not found in catalog" in n for n in decided.gatekeeper_notes)
        assert store.get_state(budget.period_label)["committed_spend"] == Decimal("0.00")


# ==================================================================
# ATTACK 4 — Budget exhaustion
# ==================================================================
class TestBudgetCeiling:
    def test_campaign_exceeding_remaining_budget_is_rejected(self, catalog, store):
        """Modeled spend = base_price * discount_pct * units. At 10% of
        399 across 200 units that is 7,980 — well past a 1,000 ceiling."""
        small_budget = DurableCampaignBudget(
            period_label="small-period",
            max_discount_spend=Decimal("1000.00"),
            max_concurrent_campaigns=5,
        )
        store.ensure_budget(small_budget)
        proposal = CampaignProposal(
            target_sku="SKU_COFFEE_001", discount_pct=Decimal("10.0"), duration_hours=72,
            rationale="Large volume campaign well beyond the remaining budget.",
            estimated_units_moved=200,
        )

        decided = campaign_gatekeeper_durable(proposal, catalog, small_budget, store)

        assert decided.status == CampaignStatus.REJECTED
        assert store.get_state(small_budget.period_label)["committed_spend"] == Decimal("0.00")

    def test_understating_the_forecast_never_creates_hidden_headroom(self, catalog, store):
        """
        An agent might lowball estimated_units_moved to slip past budget
        review. That must only shrink what it is approved to spend — the
        reservation is computed from the agent's OWN forecast, so a
        smaller forecast buys a smaller reservation, never a larger
        real-world allowance.
        """
        small_budget = DurableCampaignBudget(
            period_label="lowball-period",
            max_discount_spend=Decimal("1000.00"),
            max_concurrent_campaigns=5,
        )
        store.ensure_budget(small_budget)
        proposal = CampaignProposal(
            target_sku="SKU_COFFEE_001", discount_pct=Decimal("10.0"), duration_hours=72,
            rationale="Deliberately understated forecast to look affordable.",
            estimated_units_moved=1,
        )

        decided = campaign_gatekeeper_durable(proposal, catalog, small_budget, store)

        assert decided.status == CampaignStatus.ACTIVE
        # 399 * 10% * 1 unit = 39.90 reserved — not the ceiling.
        assert store.get_state(small_budget.period_label)["committed_spend"] == Decimal("39.90")


# ==================================================================
# ATTACK 5 — Concurrency cap
# ==================================================================
class TestConcurrencyCap:
    def test_campaign_beyond_concurrency_cap_is_rejected(self, catalog, store):
        """Budget alone is not the only ceiling — the 4th campaign must
        be refused on count even with spend to spare."""
        capped_budget = DurableCampaignBudget(
            period_label="capped-period",
            max_discount_spend=Decimal("100000.00"),
            max_concurrent_campaigns=3,
        )
        store.ensure_budget(capped_budget)

        def make(i):
            return CampaignProposal(
                target_sku="SKU_MUG_002", discount_pct=Decimal("5.0"), duration_hours=24,
                rationale=f"Campaign number {i}.", estimated_units_moved=1,
            )

        decided = [
            campaign_gatekeeper_durable(make(i), catalog, capped_budget, store)
            for i in range(4)
        ]

        assert [d.status for d in decided[:3]] == [CampaignStatus.ACTIVE] * 3
        assert decided[3].status == CampaignStatus.REJECTED
        assert store.get_state(capped_budget.period_label)["active_count"] == 3

    def test_expiring_a_campaign_frees_a_concurrency_slot(self, catalog, store):
        """The cap is on CONCURRENT campaigns, not lifetime campaigns —
        once one expires, a new one must be admissible again."""
        capped_budget = DurableCampaignBudget(
            period_label="slot-period",
            max_discount_spend=Decimal("100000.00"),
            max_concurrent_campaigns=1,
        )
        store.ensure_budget(capped_budget)

        first = campaign_gatekeeper_durable(
            CampaignProposal(
                target_sku="SKU_MUG_002", discount_pct=Decimal("5.0"), duration_hours=1,
                rationale="First campaign.", estimated_units_moved=1,
            ),
            catalog, capped_budget, store,
        )
        assert first.status == CampaignStatus.ACTIVE

        blocked = campaign_gatekeeper_durable(
            CampaignProposal(
                target_sku="SKU_MUG_002", discount_pct=Decimal("5.0"), duration_hours=1,
                rationale="Blocked by the concurrency cap.", estimated_units_moved=1,
            ),
            catalog, capped_budget, store,
        )
        assert blocked.status == CampaignStatus.REJECTED

        store.release_campaign(first.campaign_id)

        admitted = campaign_gatekeeper_durable(
            CampaignProposal(
                target_sku="SKU_MUG_002", discount_pct=Decimal("5.0"), duration_hours=1,
                rationale="Admitted after the slot freed up.", estimated_units_moved=1,
            ),
            catalog, capped_budget, store,
        )
        assert admitted.status == CampaignStatus.ACTIVE


# ==================================================================
# Approved campaigns must be accounted for
# ==================================================================
class TestApprovedCampaignAccounting:
    def test_approved_campaign_is_recorded_against_its_reservation(self, catalog, store, budget):
        """
        Without a recorded holder, committed budget can never be
        attributed or released — the reservation would be stranded for
        the life of the process.
        """
        store.ensure_budget(budget)
        proposal = CampaignProposal(
            target_sku="SKU_MUG_002", discount_pct=Decimal("10.0"), duration_hours=48,
            rationale="Legitimate campaign.", estimated_units_moved=20,
        )

        decided = campaign_gatekeeper_durable(proposal, catalog, budget, store)
        assert decided.status == CampaignStatus.ACTIVE

        record = store.get_campaign(decided.campaign_id)
        assert record is not None
        assert record.status == "ACTIVE"
        assert record.target_sku == "SKU_MUG_002"
        # 249 * 10% * 20 units = 498.00
        assert record.reserved_amount == Decimal("498.00")
        assert record.reserved_amount == store.get_state(budget.period_label)["committed_spend"]

    def test_recorded_expiry_matches_the_proposed_duration(self, catalog, store, budget):
        store.ensure_budget(budget)
        proposal = CampaignProposal(
            target_sku="SKU_MUG_002", discount_pct=Decimal("5.0"), duration_hours=48,
            rationale="Two day campaign.", estimated_units_moved=1,
        )

        decided = campaign_gatekeeper_durable(proposal, catalog, budget, store)
        record = store.get_campaign(decided.campaign_id)

        expected = time.time() + 48 * 3600
        assert abs(record.expires_at - expected) < 60

    def test_rejected_campaign_is_not_recorded(self, catalog, store, budget):
        store.ensure_budget(budget)
        proposal = CampaignProposal(
            target_sku="SKU_MUG_002", discount_pct=Decimal("40.0"), duration_hours=48,
            rationale="Over ceiling.", estimated_units_moved=20,
        )

        decided = campaign_gatekeeper_durable(proposal, catalog, budget, store)

        assert decided.status == CampaignStatus.REJECTED
        assert store.get_campaign(decided.campaign_id) is None
        assert store.active_campaigns() == []
