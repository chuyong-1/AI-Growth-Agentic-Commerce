# ============================================================
# FILE: tests/test_store_concurrency.py
# ============================================================
"""
Concurrency tests for the three stateful stores: the audit chain, the
campaign budget ledger, and the session/cart store.

These are the tests that matter most in this project, because every
guarantee the system advertises — "the chain is tamper-evident", "the
budget cannot be overspent", "a double-submit cannot clobber a
committed cart" — is a claim about behaviour under concurrent access.
Sequential tests cannot falsify any of them.

WHAT MAKES THESE REAL
-----------------------
Every test below throws genuine, unserialized threads at a single
shared store instance and asserts on the state that survives. Threads
synchronize on a `threading.Barrier` so they arrive at the contended
call together, maximizing the overlap window rather than hoping for
an unlucky interleave. Nothing is mocked and nothing is serialized by
the test itself — if a store's locking were wrong, these would fail.

The counted assertions are deliberately exact ("exactly one winner",
"committed spend equals the sum of accepted reservations"), not
directional ("no more than"), so a store that silently drops or
double-applies an operation fails rather than passes quietly.
"""

from __future__ import annotations

import threading
import time
from decimal import Decimal

import pytest

from audit_trail import GENESIS_HASH, AuditTrail
from campaign_store import (
    BudgetExceededError,
    CampaignBudgetStore,
    CampaignRecord,
    DurableCampaignBudget,
)
from schema import CartState, CartStatus
from session_store import SessionConflictError, SessionStore


def run_concurrently(fn, n_threads: int):
    """Runs fn(i) on n_threads real threads released from a barrier.

    The barrier is what gives these tests their teeth: without it,
    threads trickle in and the contended window is usually empty, so a
    broken lock would still pass most runs.
    """
    barrier = threading.Barrier(n_threads)

    def worker(i):
        barrier.wait()
        fn(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ==================================================================
# Audit chain
# ==================================================================
class TestAuditChainConcurrency:
    def test_chain_never_forks_under_concurrent_writers(self):
        """
        The core risk in a hash-chained log: two writers both read the
        same tail hash, both build an entry claiming that tail as
        prev_hash, and both append. Each resulting fork verifies fine
        on its own while disagreeing with the other about history.
        """
        trail = AuditTrail()
        n_threads, per_thread = 12, 25

        def writer(thread_id):
            for i in range(per_thread):
                trail.log("CONCURRENT_TEST_EVENT", f"cart_{thread_id}", {"thread": thread_id, "i": i})

        run_concurrently(writer, n_threads)

        entries = trail.dump()
        expected = n_threads * per_thread
        assert len(entries) == expected

        # No two entries may claim the same predecessor — that is a fork.
        prev_hashes = [e["prev_hash"] for e in entries]
        assert len(prev_hashes) == len(set(prev_hashes)), "chain forked: duplicate prev_hash"

        # Sequence numbers must be a contiguous 0..n-1 with no gaps or repeats.
        assert sorted(e["seq"] for e in entries) == list(range(expected))

        # Every entry except the first must link to its predecessor.
        assert entries[0]["prev_hash"] == GENESIS_HASH
        for earlier, later in zip(entries, entries[1:]):
            assert later["prev_hash"] == earlier["entry_hash"]

        assert trail.verify_integrity() is True

    def test_verify_integrity_detects_payload_tampering(self):
        trail = AuditTrail()
        trail.log("EVENT_A", "cart_1", {"amount": 100})
        trail.log("EVENT_B", "cart_1", {"amount": 200})
        assert trail.verify_integrity() is True

        # Reach past the public API to simulate an attacker with write
        # access to the underlying storage — the whole point of a hash
        # chain is that this is still detectable.
        trail._entries[0].payload = {"amount": 999999}

        assert trail.verify_integrity() is False

    def test_verify_integrity_detects_deletion(self):
        """Excising a middle entry breaks the prev_hash linkage of
        everything after it, even though each surviving entry's own
        hash still matches its own contents."""
        trail = AuditTrail()
        for i in range(5):
            trail.log("EVENT", "cart_1", {"i": i})
        assert trail.verify_integrity() is True

        del trail._entries[2]

        assert trail.verify_integrity() is False


# ==================================================================
# Campaign budget ledger
# ==================================================================
@pytest.fixture
def budget_store():
    return CampaignBudgetStore()


@pytest.fixture
def budget():
    return DurableCampaignBudget(
        period_label="2026-08",
        max_discount_spend=Decimal("1000.00"),
        max_concurrent_campaigns=5,
    )


class TestCampaignBudgetConcurrency:
    def test_racing_reservations_cannot_jointly_overspend(self, budget_store, budget):
        """
        Two reservations of 700 against a 1000 ceiling. Each is
        individually affordable against a stale read, so a
        read-then-write implementation lets both through and commits
        1400 against a 1000 ceiling. Exactly one must win.
        """
        budget_store.ensure_budget(budget)
        amount = Decimal("700.00")

        results, errors = {}, {}
        lock = threading.Lock()

        def reserve(i):
            try:
                res = budget_store.try_reserve(
                    budget.period_label, amount,
                    budget.max_discount_spend, budget.max_concurrent_campaigns,
                )
                with lock:
                    results[i] = res
            except BudgetExceededError as e:
                with lock:
                    errors[i] = e

        run_concurrently(reserve, 2)

        assert len(results) == 1, f"expected exactly one winner, got {len(results)}"
        assert len(errors) == 1, f"expected exactly one rejection, got {len(errors)}"

        state = budget_store.get_state(budget.period_label)
        assert state["committed_spend"] == amount
        assert state["active_count"] == 1
        assert budget_store.remaining_budget(budget.period_label) == Decimal("300.00")

    def test_many_racing_reservations_commit_exactly_what_was_accepted(self, budget_store, budget):
        """
        20 threads race for a budget that fits only 5 reservations of
        200. The ledger must agree exactly with the number of
        successes — no partially-applied reservation, no reservation
        that succeeded without being charged.
        """
        budget_store.ensure_budget(budget)
        amount = Decimal("200.00")

        accepted = []
        lock = threading.Lock()

        def reserve(i):
            try:
                budget_store.try_reserve(
                    budget.period_label, amount,
                    budget.max_discount_spend, budget.max_concurrent_campaigns,
                )
                with lock:
                    accepted.append(i)
            except BudgetExceededError:
                pass

        run_concurrently(reserve, 20)

        state = budget_store.get_state(budget.period_label)
        assert state["committed_spend"] == amount * len(accepted)
        assert state["active_count"] == len(accepted)
        assert state["committed_spend"] <= budget.max_discount_spend

    def test_concurrency_cap_enforced_independently_of_spend(self, budget_store):
        """A cheap campaign still consumes a concurrency slot — the
        count ceiling must bind even when there is budget to spare."""
        cheap_budget = DurableCampaignBudget(
            period_label="2026-09",
            max_discount_spend=Decimal("100000.00"),  # effectively unlimited
            max_concurrent_campaigns=3,
        )
        budget_store.ensure_budget(cheap_budget)

        accepted = []
        lock = threading.Lock()

        def reserve(i):
            try:
                budget_store.try_reserve(
                    cheap_budget.period_label, Decimal("1.00"),
                    cheap_budget.max_discount_spend, cheap_budget.max_concurrent_campaigns,
                )
                with lock:
                    accepted.append(i)
            except BudgetExceededError:
                pass

        run_concurrently(reserve, 10)

        assert len(accepted) == 3
        assert budget_store.get_state(cheap_budget.period_label)["active_count"] == 3

    def test_rejected_reservation_has_no_side_effects(self, budget_store, budget):
        """A rejection must leave the ledger byte-identical — a
        partially-applied reservation would leak budget permanently."""
        budget_store.ensure_budget(budget)
        budget_store.try_reserve(
            budget.period_label, Decimal("900.00"),
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        before = budget_store.get_state(budget.period_label)

        with pytest.raises(BudgetExceededError):
            budget_store.try_reserve(
                budget.period_label, Decimal("200.00"),
                budget.max_discount_spend, budget.max_concurrent_campaigns,
            )

        assert budget_store.get_state(budget.period_label) == before

    def test_boundary_exact_reservation_is_allowed(self, budget_store, budget):
        """Spending the budget to exactly zero is legal; one paisa
        beyond is not. Off-by-one here either blocks legitimate spend
        or permits an overspend."""
        budget_store.ensure_budget(budget)
        budget_store.try_reserve(
            budget.period_label, Decimal("1000.00"),
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        assert budget_store.remaining_budget(budget.period_label) == Decimal("0.00")

        with pytest.raises(BudgetExceededError):
            budget_store.try_reserve(
                budget.period_label, Decimal("0.01"),
                budget.max_discount_spend, budget.max_concurrent_campaigns,
            )


class TestCampaignRelease:
    def _reserve_and_record(self, store, budget, campaign_id, amount, expires_at):
        store.try_reserve(
            budget.period_label, amount,
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        store.record_campaign(
            CampaignRecord(
                campaign_id=campaign_id,
                period_label=budget.period_label,
                target_sku="SKU_MUG_002",
                reserved_amount=amount,
                expires_at=expires_at,
            )
        )

    def test_release_returns_budget_to_the_pool(self, budget_store, budget):
        budget_store.ensure_budget(budget)
        self._reserve_and_record(budget_store, budget, "camp_1", Decimal("400.00"), time.time() + 3600)
        assert budget_store.remaining_budget(budget.period_label) == Decimal("600.00")

        assert budget_store.release_campaign("camp_1") is True

        assert budget_store.remaining_budget(budget.period_label) == Decimal("1000.00")
        assert budget_store.get_state(budget.period_label)["active_count"] == 0

    def test_concurrent_releases_free_budget_exactly_once(self, budget_store, budget):
        """
        The double-release bug: several sweeps each see the campaign
        as ACTIVE and each hand back its reservation, freeing more
        than was ever reserved and leaving the ledger able to
        over-commit later.
        """
        budget_store.ensure_budget(budget)
        self._reserve_and_record(budget_store, budget, "camp_1", Decimal("400.00"), time.time() - 1)

        successes = []
        lock = threading.Lock()

        def release(i):
            if budget_store.release_campaign("camp_1"):
                with lock:
                    successes.append(i)

        run_concurrently(release, 8)

        assert len(successes) == 1, "campaign was released more than once"
        assert budget_store.get_state(budget.period_label)["committed_spend"] == Decimal("0.00")
        assert budget_store.get_state(budget.period_label)["active_count"] == 0

    def test_expiry_sweep_only_releases_campaigns_past_expiry(self, budget_store, budget):
        budget_store.ensure_budget(budget)
        now = time.time()
        self._reserve_and_record(budget_store, budget, "camp_expired", Decimal("300.00"), now - 10)
        self._reserve_and_record(budget_store, budget, "camp_live", Decimal("300.00"), now + 3600)

        expired = budget_store.expire_due_campaigns(now)

        assert [c.campaign_id for c in expired] == ["camp_expired"]
        assert budget_store.remaining_budget(budget.period_label) == Decimal("700.00")
        assert [c.campaign_id for c in budget_store.active_campaigns()] == ["camp_live"]

    def test_concurrent_expiry_sweeps_do_not_double_release(self, budget_store, budget):
        """Two cron ticks overlapping must not each release the same
        campaigns — the total released across both sweeps must equal
        the number of due campaigns, not double it."""
        budget_store.ensure_budget(budget)
        now = time.time()
        for i in range(4):
            self._reserve_and_record(budget_store, budget, f"camp_{i}", Decimal("100.00"), now - 10)

        released = []
        lock = threading.Lock()

        def sweep(i):
            result = budget_store.expire_due_campaigns(now)
            with lock:
                released.extend(c.campaign_id for c in result)

        run_concurrently(sweep, 6)

        assert sorted(released) == ["camp_0", "camp_1", "camp_2", "camp_3"]
        assert budget_store.get_state(budget.period_label)["committed_spend"] == Decimal("0.00")
        assert budget_store.get_state(budget.period_label)["active_count"] == 0

    def test_released_budget_is_reusable(self, budget_store, budget):
        """Releasing must genuinely restore headroom, not merely make
        the number look right — a reservation that failed before the
        release must succeed after it."""
        budget_store.ensure_budget(budget)
        self._reserve_and_record(budget_store, budget, "camp_1", Decimal("900.00"), time.time() - 1)

        with pytest.raises(BudgetExceededError):
            budget_store.try_reserve(
                budget.period_label, Decimal("500.00"),
                budget.max_discount_spend, budget.max_concurrent_campaigns,
            )

        budget_store.release_campaign("camp_1")

        budget_store.try_reserve(
            budget.period_label, Decimal("500.00"),
            budget.max_discount_spend, budget.max_concurrent_campaigns,
        )
        assert budget_store.remaining_budget(budget.period_label) == Decimal("500.00")


# ==================================================================
# Session / cart store
# ==================================================================
@pytest.fixture
def session_store():
    return SessionStore()


class TestSessionStoreConcurrency:
    def test_stale_write_is_rejected_and_does_not_clobber(self, session_store):
        cart = CartState()
        v1 = session_store.save_cart(cart)

        # A second writer advances the version behind the first's back.
        winner = session_store.get_cart(cart.cart_id)
        winner.status = CartStatus.COMPLETED
        session_store.save_cart(winner, expected_version=v1)

        # The first writer still holds v1 and tries to commit stale state.
        loser = CartState(cart_id=cart.cart_id, status=CartStatus.PAYMENT_FAILED)
        with pytest.raises(SessionConflictError):
            session_store.save_cart(loser, expected_version=v1)

        assert session_store.get_cart(cart.cart_id).status == CartStatus.COMPLETED

    def test_concurrent_double_submit_only_one_wins(self, session_store):
        """The double-submitted checkout: both requests read the same
        version, both try to commit. One must be told to retry rather
        than both silently succeeding."""
        cart = CartState()
        version = session_store.save_cart(cart)

        wins, conflicts = [], []
        lock = threading.Lock()

        def submit(i):
            try:
                session_store.save_cart(cart, expected_version=version)
                with lock:
                    wins.append(i)
            except SessionConflictError:
                with lock:
                    conflicts.append(i)

        run_concurrently(submit, 8)

        assert len(wins) == 1
        assert len(conflicts) == 7
        assert session_store.get_version(cart.cart_id) == version + 1

    def test_version_increments_exactly_once_per_successful_write(self, session_store):
        """Serial writes that each pass the correct expected_version
        must advance the version by exactly one each time — a skipped
        or repeated version breaks every later conflict check."""
        cart = CartState()
        version = session_store.save_cart(cart)
        for expected in range(1, 6):
            assert version == expected
            version = session_store.save_cart(cart, expected_version=version)

    def test_concurrent_writes_to_different_carts_do_not_interfere(self, session_store):
        """Per-cart isolation: contention on one cart must not cause
        spurious conflicts on another."""
        carts = [CartState() for _ in range(10)]
        for c in carts:
            session_store.save_cart(c)

        errors = []
        lock = threading.Lock()

        def write(i):
            try:
                session_store.save_cart(carts[i], expected_version=1)
            except SessionConflictError as e:
                with lock:
                    errors.append(e)

        run_concurrently(write, 10)

        assert not errors
        assert all(session_store.get_version(c.cart_id) == 2 for c in carts)
