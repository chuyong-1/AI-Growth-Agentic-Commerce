# ============================================================
# FILE: tests/test_durable_stores.py
# ============================================================
"""
Adversarial/concurrency tests for the durable DynamoDB-backed stores,
using `moto` to mock DynamoDB so these run with no real AWS account
and no network calls.

Install: pip install moto boto3 pytest
"""

from __future__ import annotations

import os
import threading
from decimal import Decimal

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

AWS_REGION = "us-east-1"
AUDIT_TABLE = "test-audit"
CAMPAIGN_TABLE = "test-campaigns"
SESSION_TABLE = "test-sessions"


# ------------------------------------------------------------------
# Table bootstrap helpers
# ------------------------------------------------------------------
def _create_audit_table(dynamodb):
    return dynamodb.create_table(
        TableName=AUDIT_TABLE,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "gsi1_pk", "AttributeType": "S"},
            {"AttributeName": "gsi1_sk", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "gsi1",
                "KeySchema": [
                    {"AttributeName": "gsi1_pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi1_sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
                "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            }
        ],
        ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
    )


def _create_simple_table(dynamodb, name):
    return dynamodb.create_table(
        TableName=name,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
    )


# ==================================================================
# 1a. Audit hash chain cannot fork under concurrent writers
# ==================================================================
@mock_aws
def test_audit_chain_never_forks_under_concurrent_writers():
    from audit_trail_dynamo import DynamoAuditTrail

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    _create_audit_table(dynamodb)

    trail = DynamoAuditTrail(table_name=AUDIT_TABLE, region_name=AWS_REGION)

    N_THREADS = 12
    ENTRIES_PER_THREAD = 5
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(thread_id: int):
        try:
            for i in range(ENTRIES_PER_THREAD):
                trail.log(
                    event_type="CONCURRENT_TEST_EVENT",
                    cart_id=f"cart_{thread_id}",
                    payload={"thread": thread_id, "i": i},
                )
        except Exception as e:  # pragma: no cover - captured for assertion
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"Unexpected errors during concurrent logging: {errors}"

    entries = trail.dump()
    expected_total = N_THREADS * ENTRIES_PER_THREAD
    assert len(entries) == expected_total

    # No two entries may share the same prev_hash — that would mean the
    # chain forked (two writers both built on the same tail).
    prev_hashes = [e["prev_hash"] for e in entries]
    assert len(prev_hashes) == len(set(prev_hashes)), (
        "Chain forked: multiple entries share the same prev_hash"
    )

    # Sequence numbers must be contiguous with no gaps or duplicates.
    seqs = sorted(e["seq"] for e in entries)
    assert seqs == list(range(expected_total))

    # The full chain must independently verify.
    assert trail.verify_integrity() is True


@mock_aws
def test_audit_chain_detects_tampering():
    from audit_trail_dynamo import DynamoAuditTrail

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    _create_audit_table(dynamodb)
    trail = DynamoAuditTrail(table_name=AUDIT_TABLE, region_name=AWS_REGION)

    trail.log("EVENT_A", "cart_1", {"x": 1})
    trail.log("EVENT_B", "cart_1", {"x": 2})
    assert trail.verify_integrity() is True

    table = dynamodb.Table(AUDIT_TABLE)
    table.update_item(
        Key={"PK": "ENTRY#000000000000", "SK": "META"},
        UpdateExpression="SET payload_json = :tampered",
        ExpressionAttributeValues={":tampered": '{"x": 9999}'},
    )

    assert trail.verify_integrity() is False


# ==================================================================
# 1b. Campaign budget cannot be overspent by racing reservations
# ==================================================================
@mock_aws
def test_campaign_budget_atomic_reservation_blocks_overspend():
    from campaign_store import CampaignBudgetStore, DurableCampaignBudget, BudgetExceededError

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    _create_simple_table(dynamodb, CAMPAIGN_TABLE)

    store = CampaignBudgetStore(table_name=CAMPAIGN_TABLE, region_name=AWS_REGION)
    budget = DurableCampaignBudget(
        period_label="2026-08",
        max_discount_spend=Decimal("1000.00"),
        max_concurrent_campaigns=5,
    )
    store.ensure_budget(budget)

    # Two reservations that are EACH individually affordable against a
    # stale read of remaining budget (1000 each fits under 1000 alone),
    # but which JOINTLY would blow the ceiling (1000 + 1000 > 1000).
    # try_reserve is atomic server-side, so this must not double-spend
    # even though both "look" affordable from a naive client-side read.
    amount = Decimal("700.00")

    results = {}
    errors = {}
    barrier = threading.Barrier(2)

    def reserve(tag):
        barrier.wait()  # maximize overlap window
        try:
            results[tag] = store.try_reserve(
                "2026-08", amount, budget.max_discount_spend, budget.max_concurrent_campaigns
            )
        except BudgetExceededError as e:
            errors[tag] = e

    t1 = threading.Thread(target=reserve, args=("a",))
    t2 = threading.Thread(target=reserve, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one must succeed and one must be rejected — 700 + 700 = 1400 > 1000.
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(list(errors.values())[0], BudgetExceededError)

    final_state = store.get_state("2026-08")
    committed = Decimal(final_state["committed_spend"])
    assert committed <= budget.max_discount_spend
    assert committed == amount  # only the winning reservation landed


@mock_aws
def test_campaign_budget_sequential_reservations_respect_ceiling():
    from campaign_store import CampaignBudgetStore, DurableCampaignBudget, BudgetExceededError

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    _create_simple_table(dynamodb, CAMPAIGN_TABLE)

    store = CampaignBudgetStore(table_name=CAMPAIGN_TABLE, region_name=AWS_REGION)
    budget = DurableCampaignBudget(
        period_label="2026-09",
        max_discount_spend=Decimal("500.00"),
        max_concurrent_campaigns=3,
    )
    store.ensure_budget(budget)

    store.try_reserve("2026-09", Decimal("300.00"), budget.max_discount_spend, budget.max_concurrent_campaigns)

    with pytest.raises(BudgetExceededError):
        store.try_reserve("2026-09", Decimal("300.00"), budget.max_discount_spend, budget.max_concurrent_campaigns)

    state = store.get_state("2026-09")
    assert Decimal(state["committed_spend"]) == Decimal("300.00")


@mock_aws
def test_campaign_budget_concurrent_count_ceiling_enforced():
    from campaign_store import CampaignBudgetStore, DurableCampaignBudget, BudgetExceededError

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    _create_simple_table(dynamodb, CAMPAIGN_TABLE)

    store = CampaignBudgetStore(table_name=CAMPAIGN_TABLE, region_name=AWS_REGION)
    budget = DurableCampaignBudget(
        period_label="2026-10",
        max_discount_spend=Decimal("100000.00"),
        max_concurrent_campaigns=1,
    )
    store.ensure_budget(budget)

    store.try_reserve("2026-10", Decimal("1.00"), budget.max_discount_spend, budget.max_concurrent_campaigns)

    with pytest.raises(BudgetExceededError):
        store.try_reserve("2026-10", Decimal("1.00"), budget.max_discount_spend, budget.max_concurrent_campaigns)


# ==================================================================
# 1c. Session store optimistic concurrency
# ==================================================================
@mock_aws
def test_session_store_stale_version_raises_conflict_and_does_not_clobber():
    from session_store import SessionStore, SessionConflictError
    from schema import CartState, CartStatus

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    _create_simple_table(dynamodb, SESSION_TABLE)

    store = SessionStore(table_name=SESSION_TABLE, region_name=AWS_REGION)

    cart = CartState(cart_id="cart_conflict_test")
    v0 = store.save_cart(cart, expected_version=None)  # initial write, version -> 1
    assert v0 == 1

    # Reader A and Reader B both fetch version 1.
    reader_a_version = store.get_version(cart.cart_id)
    reader_b_version = store.get_version(cart.cart_id)
    assert reader_a_version == reader_b_version == 1

    # Reader A writes first, successfully advancing to version 2.
    cart.status = CartStatus.PENDING_AUDIT
    v_a = store.save_cart(cart, expected_version=reader_a_version)
    assert v_a == 2

    # Reader B, still holding the stale version=1 read, tries to write —
    # must be rejected rather than silently overwriting A's update.
    cart.status = CartStatus.AUDIT_FAILED
    with pytest.raises(SessionConflictError):
        store.save_cart(cart, expected_version=reader_b_version)

    # The stored cart must reflect Reader A's write, not B's clobber attempt.
    stored = store.get_cart(cart.cart_id)
    assert stored.status == CartStatus.PENDING_AUDIT
    assert store.get_version(cart.cart_id) == 2


@mock_aws
def test_session_store_concurrent_double_submit_only_one_wins():
    from session_store import SessionStore, SessionConflictError
    from schema import CartState

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    _create_simple_table(dynamodb, SESSION_TABLE)

    store = SessionStore(table_name=SESSION_TABLE, region_name=AWS_REGION)
    cart = CartState(cart_id="cart_double_submit")
    store.save_cart(cart, expected_version=None)  # version 1

    current_version = store.get_version(cart.cart_id)

    results = {}
    errors = {}
    barrier = threading.Barrier(2)

    def submit(tag):
        barrier.wait()
        try:
            results[tag] = store.save_cart(cart, expected_version=current_version)
        except SessionConflictError as e:
            errors[tag] = e

    t1 = threading.Thread(target=submit, args=("first",))
    t2 = threading.Thread(target=submit, args=("second",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert store.get_version(cart.cart_id) == 2