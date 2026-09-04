# ============================================================
# FILE: campaign_store.py
# ============================================================
"""
In-memory campaign budget store — thread-safe, zero-dependency.

Holds two things: the per-period budget ledger (how much has been
committed, how many campaigns are active) and the campaign records
that account for who is holding each reservation.

CONCURRENCY MODEL
-------------------
This is the module where correctness actually matters most, because
it's a shared numeric ceiling (a budget) rather than a per-entity
record like a cart. The naive bug this guards against:

    Thread A reads committed_spend = 800
    Thread B reads committed_spend = 800
    Thread A computes 800 + 500 = 1300 <= 1500 ceiling -> "fits", writes 1300
    Thread B computes 800 + 500 = 1300 <= 1500 ceiling -> "fits", writes 1300
    Real total should have been 1800, which EXCEEDS the 1500 ceiling.
    Both writes "succeeded" and the ceiling was silently violated.

The fix: the ENTIRE read -> compute -> ceiling-check -> write
sequence happens as one atomic critical section. Thread B is BLOCKED
from even reading committed_spend until Thread A's full
reserve-or-reject sequence has completed and released the lock — so
Thread B's read is always the post-A value, and the race above is
structurally impossible, not just improbable.

THE TRADEOFF, STATED PLAINLY: a single-process lock gives true mutual
exclusion, but only within one Python process. This store is correct
under `uvicorn api:app` (one worker, many threads) and is NOT correct
across multiple workers or hosts — two processes each have their own
lock and their own dict, so they would not see each other's spend at
all. Scaling out means replacing this class with a backend whose
check-and-commit is atomic on the server side (a DynamoDB conditional
UpdateItem, a Postgres `UPDATE ... WHERE committed + :amt <= ceiling`,
or a Redis Lua script). Every caller depends only on this class's
public method signatures, so that swap touches no other module.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

logger = logging.getLogger("agentictrade.campaign_store")


class BudgetExceededError(Exception):
    """Raised when a reservation would push committed spend or
    active-campaign count past the configured ceiling."""


@dataclass
class DurableCampaignBudget:
    """Config-carrying handle. Mutable state (committed spend, active
    count) lives entirely inside CampaignBudgetStore's internal dict,
    not on this object — keeping this a pure config value prevents
    any caller from accidentally reading a stale cached total."""
    period_label: str
    max_discount_spend: Decimal = Decimal("15000.00")
    max_concurrent_campaigns: int = 3
    max_single_campaign_discount_pct: Decimal = Decimal("25.0")


@dataclass
class _BudgetRecord:
    max_discount_spend: Decimal
    max_concurrent_campaigns: int
    max_single_campaign_discount_pct: Decimal
    committed_spend: Decimal = Decimal("0.00")
    active_count: int = 0


@dataclass
class CampaignRecord:
    """What an approved campaign is holding against a budget period.

    `reserved_amount` is what release() must give back — storing it on
    the record rather than recomputing it at expiry time means a later
    catalog price change can never cause a release to return a
    different amount than was originally reserved, which would drift
    the ledger."""
    campaign_id: str
    period_label: str
    target_sku: str
    reserved_amount: Decimal
    expires_at: float
    status: str = "ACTIVE"


class CampaignBudgetStore:
    """
    Thread-safe, in-process budget ledger. `try_reserve` holds the
    lock for its ENTIRE read-check-write sequence — this is the
    critical design point, not an incidental detail. See module
    docstring for why a narrower lock scope would reintroduce the
    exact race this class exists to prevent.

    Note that no public method calls another public method while
    holding the lock; `threading.Lock` is not reentrant, so doing so
    would deadlock. `expire_due_campaigns` is the one place that could
    be tempted to, and it deliberately releases the lock between
    selecting due campaigns and releasing them.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._budgets: dict[str, _BudgetRecord] = {}
        self._campaigns: dict[str, CampaignRecord] = {}

    def ensure_budget(self, budget: DurableCampaignBudget) -> None:
        with self._lock:
            if budget.period_label in self._budgets:
                return  # already initialized — matches the DynamoDB version's idempotent put
            self._budgets[budget.period_label] = _BudgetRecord(
                max_discount_spend=budget.max_discount_spend,
                max_concurrent_campaigns=budget.max_concurrent_campaigns,
                max_single_campaign_discount_pct=budget.max_single_campaign_discount_pct,
            )
            logger.info("Initialized campaign budget for period=%s", budget.period_label)

    def get_state(self, period_label: str) -> dict:
        with self._lock:
            record = self._budgets.get(period_label)
            if record is None:
                raise KeyError(f"No budget initialized for period '{period_label}'")
            # Return a snapshot dict (not the live object) so callers can't
            # mutate internal state by reference — mirrors the DynamoDB
            # version returning a fresh dict per get_item call.
            return {
                "period_label": period_label,
                "max_discount_spend": record.max_discount_spend,
                "max_concurrent_campaigns": record.max_concurrent_campaigns,
                "max_single_campaign_discount_pct": record.max_single_campaign_discount_pct,
                "committed_spend": record.committed_spend,
                "active_count": record.active_count,
            }

    def remaining_budget(self, period_label: str) -> Decimal:
        state = self.get_state(period_label)
        return state["max_discount_spend"] - state["committed_spend"]

    def try_reserve(
        self,
        period_label: str,
        amount: Decimal,
        max_discount_spend: Decimal,
        max_concurrent_campaigns: int,
    ) -> dict:
        """
        Atomically reserves `amount` against the budget and increments
        the active-campaign count, or raises BudgetExceededError with
        no side effects at all if either ceiling would be breached.

        The whole method body runs under one lock acquisition — this
        is what makes it atomic. No other thread can observe or
        mutate committed_spend / active_count while this method is
        mid-execution.
        """
        with self._lock:
            record = self._budgets.get(period_label)
            if record is None:
                raise KeyError(f"No budget initialized for period '{period_label}'")

            new_committed = record.committed_spend + amount
            new_active = record.active_count + 1

            if new_committed > max_discount_spend:
                raise BudgetExceededError(
                    f"Reserving {amount} for period '{period_label}' would push "
                    f"committed spend to {new_committed}, exceeding ceiling "
                    f"{max_discount_spend}"
                )
            if new_active > max_concurrent_campaigns:
                raise BudgetExceededError(
                    f"Reserving this campaign would raise active campaigns to "
                    f"{new_active}, exceeding concurrency cap {max_concurrent_campaigns}"
                )

            # Both checks passed — commit the reservation.
            record.committed_spend = new_committed
            record.active_count = new_active

            return {
                "period_label": period_label,
                "committed_spend": record.committed_spend,
                "active_count": record.active_count,
            }

    # ---------------- campaign records ----------------
    def record_campaign(self, record: CampaignRecord) -> None:
        """Registers an approved campaign as the holder of a reservation.

        Called only after try_reserve() has already succeeded, so this
        never performs a ceiling check itself — it records WHO holds
        budget that has already been committed, which is what makes a
        later release attributable to a specific campaign instead of
        being an unaccounted-for adjustment to the ledger."""
        with self._lock:
            self._campaigns[record.campaign_id] = record

    def get_campaign(self, campaign_id: str) -> Optional[CampaignRecord]:
        with self._lock:
            return self._campaigns.get(campaign_id)

    def active_campaigns(self, period_label: Optional[str] = None) -> list[CampaignRecord]:
        with self._lock:
            return [
                c for c in self._campaigns.values()
                if c.status == "ACTIVE"
                and (period_label is None or c.period_label == period_label)
            ]

    def release_campaign(self, campaign_id: str) -> bool:
        """
        Releases one campaign's reservation back to its budget period
        and marks it EXPIRED. Returns True if this call performed the
        release, False if the campaign was unknown or already released.

        IDEMPOTENCE IS THE POINT: the status check and the ledger
        decrement happen under one lock acquisition, so two concurrent
        expiry sweeps (or a sweep racing a manual cancel) cannot both
        see the campaign as ACTIVE and each give its budget back —
        which would silently free twice what was ever reserved.
        """
        with self._lock:
            campaign = self._campaigns.get(campaign_id)
            if campaign is None:
                logger.warning("release_campaign() for unknown campaign '%s' — ignoring", campaign_id)
                return False
            if campaign.status != "ACTIVE":
                return False

            budget = self._budgets.get(campaign.period_label)
            if budget is None:
                logger.warning(
                    "Campaign '%s' references unknown period '%s' — marking expired without release",
                    campaign_id, campaign.period_label,
                )
                campaign.status = "EXPIRED"
                return False

            # Floored defensively: a release must never be able to drive
            # the ledger negative, even under a bookkeeping bug upstream.
            budget.committed_spend = max(
                Decimal("0.00"), budget.committed_spend - campaign.reserved_amount
            )
            budget.active_count = max(0, budget.active_count - 1)
            campaign.status = "EXPIRED"
            return True

    def expire_due_campaigns(self, now_epoch: float, period_label: Optional[str] = None) -> list[CampaignRecord]:
        """
        Finds every ACTIVE campaign past its expires_at and releases it.

        Selection and release both go through the lock via
        release_campaign(), so a campaign that another sweep already
        claimed is skipped rather than double-released — the scan is
        advisory, the release is authoritative.
        """
        with self._lock:
            due = [
                c for c in self._campaigns.values()
                if c.status == "ACTIVE"
                and c.expires_at <= now_epoch
                and (period_label is None or c.period_label == period_label)
            ]

        return [c for c in due if self.release_campaign(c.campaign_id)]

    def clear_all(self) -> None:
        """Test/demo utility only."""
        with self._lock:
            self._budgets = {}
            self._campaigns = {}


_CAMPAIGN_STORE_SINGLETON: Optional[CampaignBudgetStore] = None


def get_campaign_store() -> CampaignBudgetStore:
    global _CAMPAIGN_STORE_SINGLETON
    if _CAMPAIGN_STORE_SINGLETON is None:
        _CAMPAIGN_STORE_SINGLETON = CampaignBudgetStore()
    return _CAMPAIGN_STORE_SINGLETON