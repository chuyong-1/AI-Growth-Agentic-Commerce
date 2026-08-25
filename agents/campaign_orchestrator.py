# ============================================================
# FILE: agents/campaign_orchestrator.py
# ============================================================
"""
Campaign Orchestrator — the "grow revenue" half of the brief, durable
and race-safe end to end.

Where the upsell agent reacts to ONE user's cart in real time, the
Campaign Orchestrator runs periodically (cron / scheduled Lambda /
manual trigger) over aggregate signals and proposes STORE-WIDE,
time-boxed promotional campaigns: "discount SKU X for the next 72h to
drive attach rate on SKU Y."

Same non-negotiable split as the rest of the system:
  - CampaignOrchestrator (rules-based here; swap for an LLM call the
    same way conversational_agent.py does, if desired) PROPOSES
    campaigns.
  - campaign_gatekeeper_durable (deterministic, no LLM) enforces a
    hard campaign spend ceiling and per-SKU discount ceiling, and is
    the ONLY thing that can flip a campaign to ACTIVE.
  - Every proposal and verdict is written to the same hash-chained
    AUDIT trail used by checkout — one audit trail, one source of
    truth, whether money moved via a single sale or a campaign.

The budget check-and-commit is delegated to CampaignBudgetStore
(campaign_store.py), which performs it as a single atomic DynamoDB
conditional update — so two campaign cycles racing each other can
never both "fit" against the same remaining budget and jointly
overspend it. This module contains zero read-then-write budget logic
itself; that TOCTOU-prone pattern was deliberately removed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from schema import Catalog
from audit_trail import AUDIT
from campaign_store import CampaignBudgetStore, BudgetExceededError


# ------------------------------------------------------------------
# Campaign schema
# ------------------------------------------------------------------
class CampaignStatus(str, Enum):
    PROPOSED = "PROPOSED"
    REJECTED = "REJECTED"
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class CampaignProposal(BaseModel):
    campaign_id: str = Field(default_factory=lambda: f"camp_{uuid.uuid4().hex[:10]}")
    target_sku: str
    discount_pct: Decimal
    duration_hours: int = Field(gt=0, le=24 * 30)  # hard cap: no campaign longer than 30 days
    rationale: str = Field(description="Why this campaign is expected to grow revenue — "
                                        "must cite a specific signal, not a vague claim.")
    estimated_units_moved: int = Field(gt=0, description="Agent's forecast — advisory only for "
                                                           "planning, but IS what budget is "
                                                           "reserved against (see gatekeeper).")
    status: CampaignStatus = CampaignStatus.PROPOSED
    gatekeeper_notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None

    @field_validator("discount_pct", mode="before")
    @classmethod
    def coerce_decimal(cls, v):
        return Decimal(str(v))


@dataclass
class DurableCampaignBudget:
    """
    Config-carrying handle only. Mutable state (committed spend,
    active count) lives entirely in DynamoDB via CampaignBudgetStore —
    this object is intentionally NOT a cache of that state, to avoid
    the staleness/race bug an in-memory mirror would reintroduce.
    """
    period_label: str
    max_discount_spend: Decimal = Decimal("15000.00")  # INR, modeled foregone-margin ceiling
    max_concurrent_campaigns: int = 3
    max_single_campaign_discount_pct: Decimal = Decimal("25.0")


# ------------------------------------------------------------------
# Signal gathering (stubbed here; wire to real analytics in prod)
# ------------------------------------------------------------------
def gather_growth_signals(catalog: Catalog) -> list[dict]:
    """
    In production: pull from order history / DynamoDB aggregates
    (attach rate, cart-abandon rate per SKU, inventory age). Stubbed
    here with static demo signals so the orchestrator is runnable
    standalone without a live order history.
    """
    return [
        {
            "sku": "SKU_MUG_002",
            "signal": "low_attach_rate",
            "detail": "Ceramic Mug attaches to only 12% of Filter Coffee purchases "
                      "despite being upsell-eligible; comparable SKUs attach at 35%+.",
        },
        {
            "sku": "SKU_GRINDER_003",
            "signal": "aging_inventory",
            "detail": "Manual Coffee Grinder has had zero sales velocity in the "
                      "simulated last 14 days.",
        },
    ]


# ------------------------------------------------------------------
# NODE 1: CampaignOrchestrator (proposal only — rules-based by
# default; swap _propose_for_signal's body for an LLM call if you
# want the same Anthropic tool-calling pattern as conversational_agent.py)
# ------------------------------------------------------------------
class CampaignOrchestrator:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def _propose_for_signal(self, signal: dict) -> Optional[CampaignProposal]:
        sku = signal["sku"]
        try:
            item = self.catalog.get(sku)
        except KeyError:
            AUDIT.log("CAMPAIGN_SKIPPED_UNKNOWN_SKU", "campaign_system", {"sku": sku})
            return None

        if signal["signal"] == "low_attach_rate":
            # Propose a moderate discount, never exceeding catalog ceiling —
            # gatekeeper re-checks this anyway, but proposing sane numbers
            # up front means fewer rejections in the common case.
            proposed_pct = min(item.max_discount_pct, Decimal("10.0"))
            return CampaignProposal(
                target_sku=sku,
                discount_pct=proposed_pct,
                duration_hours=72,
                rationale=f"{signal['detail']} Proposing a {proposed_pct}% "
                          f"time-boxed discount to test price elasticity on attach rate.",
                estimated_units_moved=25,
            )

        if signal["signal"] == "aging_inventory":
            proposed_pct = min(item.max_discount_pct, Decimal("5.0"))
            return CampaignProposal(
                target_sku=sku,
                discount_pct=proposed_pct,
                duration_hours=168,
                rationale=f"{signal['detail']} Proposing a {proposed_pct}% clearance "
                          f"nudge for one week to restore sales velocity.",
                estimated_units_moved=10,
            )

        return None

    def propose_campaigns(self) -> list[CampaignProposal]:
        signals = gather_growth_signals(self.catalog)
        proposals = []
        for signal in signals:
            proposal = self._propose_for_signal(signal)
            if proposal:
                AUDIT.log(
                    "CAMPAIGN_PROPOSED",
                    "campaign_system",
                    proposal.model_dump(mode="json"),
                )
                proposals.append(proposal)
        return proposals


# ------------------------------------------------------------------
# NODE 2: campaign_gatekeeper_durable (THE BAR — deterministic, no LLM)
# ------------------------------------------------------------------
# Hard rules enforced, and ONLY enforced, here:
#   1. discount_pct <= catalog max_discount_pct for target SKU.
#   2. discount_pct <= budget.max_single_campaign_discount_pct.
#   3. Campaign's modeled worst-case spend (base_price * discount_pct *
#      estimated_units_moved) must fit inside remaining budget — this
#      check-and-commit is a SINGLE ATOMIC DynamoDB conditional update
#      (CampaignBudgetStore.try_reserve), not a Python read-then-write,
#      so it cannot be raced by a concurrent campaign cycle.
#   4. Concurrent active campaign count <= max_concurrent_campaigns —
#      enforced by the SAME atomic update as (3).
# Any violation => REJECTED. Nothing goes ACTIVE without passing all four.
# ------------------------------------------------------------------
def campaign_gatekeeper_durable(
    proposal: CampaignProposal,
    catalog: Catalog,
    budget: DurableCampaignBudget,
    store: CampaignBudgetStore,
) -> CampaignProposal:
    violations: list[str] = []

    try:
        item = catalog.get(proposal.target_sku)
    except KeyError:
        proposal.status = CampaignStatus.REJECTED
        proposal.gatekeeper_notes = [f"SKU '{proposal.target_sku}' not found in catalog"]
        AUDIT.log(
            "CAMPAIGN_VERDICT_REJECTED",
            "campaign_system",
            {"campaign_id": proposal.campaign_id, "violations": proposal.gatekeeper_notes},
        )
        return proposal

    # --- Rule 1: per-SKU discount ceiling ---
    if proposal.discount_pct > item.max_discount_pct:
        violations.append(
            f"Proposed discount {proposal.discount_pct}% exceeds catalog ceiling "
            f"{item.max_discount_pct}% for {proposal.target_sku}"
        )

    # --- Rule 2: campaign-program-wide single-campaign ceiling ---
    if proposal.discount_pct > budget.max_single_campaign_discount_pct:
        violations.append(
            f"Proposed discount {proposal.discount_pct}% exceeds campaign-wide "
            f"single-campaign ceiling {budget.max_single_campaign_discount_pct}%"
        )

    if violations:
        proposal.status = CampaignStatus.REJECTED
        proposal.gatekeeper_notes = violations
        AUDIT.log(
            "CAMPAIGN_VERDICT_REJECTED",
            "campaign_system",
            {"campaign_id": proposal.campaign_id, "violations": violations},
        )
        return proposal

    # --- Rules 3 & 4: atomic budget + concurrency reservation ---
    # Modeled spend is computed against the agent's OWN forecast
    # (estimated_units_moved). An agent that lowballs this forecast to
    # dodge budget review only shrinks what it's approved to spend —
    # it never creates hidden headroom.
    modeled_spend = (
        item.base_price
        * (proposal.discount_pct / Decimal("100"))
        * proposal.estimated_units_moved
    ).quantize(Decimal("0.01"))

    store.ensure_budget(budget)  # no-op if already initialized for this period

    try:
        new_state = store.try_reserve(
            period_label=budget.period_label,
            amount=modeled_spend,
            max_discount_spend=budget.max_discount_spend,
            max_concurrent_campaigns=budget.max_concurrent_campaigns,
        )
    except BudgetExceededError as e:
        proposal.status = CampaignStatus.REJECTED
        proposal.gatekeeper_notes = [str(e)]
        AUDIT.log(
            "CAMPAIGN_VERDICT_REJECTED",
            "campaign_system",
            {
                "campaign_id": proposal.campaign_id,
                "modeled_spend": str(modeled_spend),
                "violations": [str(e)],
            },
        )
        return proposal

    proposal.status = CampaignStatus.ACTIVE
    proposal.expires_at = (
        datetime.now(timezone.utc) + timedelta(hours=proposal.duration_hours)
    ).isoformat()

    AUDIT.log(
        "CAMPAIGN_VERDICT_APPROVED",
        "campaign_system",
        {
            "campaign_id": proposal.campaign_id,
            "target_sku": proposal.target_sku,
            "modeled_spend": str(modeled_spend),
            "committed_spend_after": str(new_state["committed_spend"]),
            "active_count_after": int(new_state["active_count"]),
            "expires_at": proposal.expires_at,
        },
    )
    return proposal


# ------------------------------------------------------------------
# Orchestration entrypoint — run this on a schedule
# ------------------------------------------------------------------
def run_campaign_cycle_durable(
    catalog: Catalog,
    budget: DurableCampaignBudget,
    store: CampaignBudgetStore,
) -> list[CampaignProposal]:
    """
    Call this from a scheduled trigger (EventBridge rule, cron, or the
    /api/campaigns/run endpoint). Returns every proposal this cycle
    produced, each carrying its final gatekeeper-decided status.
    """
    store.ensure_budget(budget)

    orchestrator = CampaignOrchestrator(catalog)
    proposals = orchestrator.propose_campaigns()

    decided = [campaign_gatekeeper_durable(p, catalog, budget, store) for p in proposals]

    AUDIT.log(
        "CAMPAIGN_CYCLE_COMPLETE",
        "campaign_system",
        {
            "period": budget.period_label,
            "proposed": len(decided),
            "activated": sum(1 for p in decided if p.status == CampaignStatus.ACTIVE),
            "rejected": sum(1 for p in decided if p.status == CampaignStatus.REJECTED),
            "remaining_budget": str(store.remaining_budget(budget.period_label)),
        },
    )
    return decided