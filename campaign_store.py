# ============================================================
# FILE: campaign_store.py
# ============================================================
"""
In-memory campaign budget store — thread-safe, zero-dependency
replacement for the DynamoDB-backed version.

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

The fix is the same pattern used in the DynamoDB version, just
enforced with a Python Lock instead of a database-side
ConditionExpression: the ENTIRE read -> compute -> ceiling-check ->
write sequence happens as one atomic critical section. Thread B is
BLOCKED from even reading committed_spend until Thread A's full
reserve-or-reject sequence has completed and released the lock — so
Thread B's read is always the post-A value, and the race above is
structurally impossible, not just improbable.

This is actually a STRONGER guarantee than the DynamoDB optimistic-
locking version (which could still lose a retry race under enough
contention) — a single-process lock gives true mutual exclusion. The
tradeoff, called out honestly: this only holds within one Python
process. A real multi-instance deployment (multiple Lambda/uvicorn
workers) needs the DynamoDB version's cross-process coordination.
That tradeoff is exactly why the DynamoDB version exists as the
production target and this one is explicitly the local-dev stand-in.
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


class CampaignBudgetStore:
    """
    Thread-safe, in-process budget ledger. `try_reserve` holds the
    lock for its ENTIRE read-check-write sequence — this is the
    critical design point, not an incidental detail. See module
    docstring for why a narrower lock scope would reintroduce the
    exact race this class exists to prevent.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._budgets: dict[str, _BudgetRecord] = {}

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

    def release(self, period_label: str, amount: Decimal) -> None:
        """Frees budget and decrements the active count — called when
        a campaign expires or is cancelled early. Floors both values
        at zero defensively (a release should never be able to drive
        the ledger negative even under a bookkeeping bug upstream)."""
        with self._lock:
            record = self._budgets.get(period_label)
            if record is None:
                logger.warning("release() called for unknown period '%s' — ignoring", period_label)
                return
            record.committed_spend = max(Decimal("0.00"), record.committed_spend - amount)
            record.active_count = max(0, record.active_count - 1)

    def clear_all(self) -> None:
        """Test/demo utility only."""
        with self._lock:
            self._budgets = {}


_CAMPAIGN_STORE_SINGLETON: Optional[CampaignBudgetStore] = None


def get_campaign_store() -> CampaignBudgetStore:
    global _CAMPAIGN_STORE_SINGLETON
    if _CAMPAIGN_STORE_SINGLETON is None:
        _CAMPAIGN_STORE_SINGLETON = CampaignBudgetStore()
    return _CAMPAIGN_STORE_SINGLETON