# ============================================================
# FILE: campaign_store.py
# ============================================================
"""
Durable campaign budget store — DynamoDB-backed replacement for the
in-process CampaignBudget dataclass in campaign_orchestrator.py.

The whole point of a budget ceiling is that it holds under concurrent
writers. This uses DynamoDB's atomic ADD update (a native increment,
not read-modify-write from the client) for committing spend, so two
overlapping campaign cycles can never both "fit" against the same
remaining budget and jointly overspend it — the second writer's
increment is still atomic even though it's racing the first.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("agentictrade.campaign_store")

CAMPAIGN_TABLE = os.environ.get("CAMPAIGN_TABLE", "agentictrade-dev-campaigns")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


class BudgetExceededError(Exception):
    """Raised when an atomic reservation would push committed spend or
    active-campaign count past the configured ceiling — the caller
    should treat this exactly like a gatekeeper rejection."""


@dataclass
class DurableCampaignBudget:
    """Config-carrying handle. The actual mutable state (committed
    spend, active count) lives in DynamoDB, not on this object — this
    is intentionally NOT a cache, to avoid exactly the staleness bug
    that made the in-memory version race-unsafe."""
    period_label: str
    max_discount_spend: Decimal = Decimal("15000.00")
    max_concurrent_campaigns: int = 3
    max_single_campaign_discount_pct: Decimal = Decimal("25.0")


class CampaignBudgetStore:
    def __init__(self, table_name: str = CAMPAIGN_TABLE, region_name: str = AWS_REGION):
        self._table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)

    def _budget_key(self, period_label: str) -> dict:
        return {"PK": f"BUDGET#{period_label}", "SK": "STATE"}

    def ensure_budget(self, budget: DurableCampaignBudget) -> None:
        try:
            self._table.put_item(
                Item={
                    **self._budget_key(budget.period_label),
                    "period_label": budget.period_label,
                    "max_discount_spend": budget.max_discount_spend,
                    "max_concurrent_campaigns": budget.max_concurrent_campaigns,
                    "max_single_campaign_discount_pct": budget.max_single_campaign_discount_pct,
                    "committed_spend": Decimal("0.00"),
                    "active_count": 0,
                },
                ConditionExpression="attribute_not_exists(PK)",
            )
            logger.info("Initialized campaign budget for period=%s", budget.period_label)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise  # already exists — expected on every call after the first

    def get_state(self, period_label: str) -> dict:
        resp = self._table.get_item(Key=self._budget_key(period_label), ConsistentRead=True)
        item = resp.get("Item")
        if not item:
            raise KeyError(f"No budget initialized for period '{period_label}'")
        return item

    def remaining_budget(self, period_label: str) -> Decimal:
        item = self.get_state(period_label)
        return Decimal(item["max_discount_spend"]) - Decimal(item["committed_spend"])

    def try_reserve(
        self,
        period_label: str,
        amount: Decimal,
        max_discount_spend: Decimal,
        max_concurrent_campaigns: int,
    ) -> dict:
        """
        Atomically reserves `amount` against the budget AND increments
        the active-campaign counter, but ONLY if both remain within
        ceiling — enforced entirely inside a single conditional
        DynamoDB UpdateItem, so no other writer can slip in between
        the check and the write (the classic TOCTOU bug this design
        specifically avoids).
        """
        try:
            resp = self._table.update_item(
                Key=self._budget_key(period_label),
                UpdateExpression="ADD committed_spend :amt, active_count :one",
                ConditionExpression=(
                    "committed_spend + :amt <= :max_spend AND "
                    "active_count + :one <= :max_concurrent"
                ),
                ExpressionAttributeValues={
                    ":amt": amount,
                    ":one": 1,
                    ":max_spend": max_discount_spend,
                    ":max_concurrent": max_concurrent_campaigns,
                },
                ReturnValues="ALL_NEW",
            )
            return resp["Attributes"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise BudgetExceededError(
                    f"Reserving {amount} for period '{period_label}' would exceed "
                    f"budget ceiling or concurrent-campaign cap"
                ) from e
            raise

    def release(self, period_label: str, amount: Decimal) -> None:
        """Called when a campaign expires/cancels early — frees budget
        and decrements the active count. Floors at zero defensively."""
        self._table.update_item(
            Key=self._budget_key(period_label),
            UpdateExpression="ADD committed_spend :neg_amt, active_count :neg_one",
            ExpressionAttributeValues={":neg_amt": -amount, ":neg_one": -1},
        )


_CAMPAIGN_STORE_SINGLETON: Optional[CampaignBudgetStore] = None


def get_campaign_store() -> CampaignBudgetStore:
    global _CAMPAIGN_STORE_SINGLETON
    if _CAMPAIGN_STORE_SINGLETON is None:
        _CAMPAIGN_STORE_SINGLETON = CampaignBudgetStore()
    return _CAMPAIGN_STORE_SINGLETON