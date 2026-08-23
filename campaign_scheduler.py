# ============================================================
# FILE: campaign_scheduler.py
# ============================================================
"""
Scheduled housekeeping for the campaign budget system.

Intended trigger: an hourly EventBridge (CloudWatch Events) rule
invoking a small Lambda handler that calls `expire_stale_campaigns`.
This is intentionally decoupled from the campaign_orchestrator's
propose/gatekeeper path — it only ever RELEASES budget that is no
longer legitimately held, it never reserves or approves anything.

Data model assumption (matches the DynamoDB campaign table used by
campaign_store.CampaignBudgetStore, in the same table, alongside the
"BUDGET#<period>" / "STATE" item):

    PK = "CAMPAIGN#<campaign_id>"
    SK = "STATE"
    fields:
        campaign_id      (str)
        period_label     (str)   -- which budget period this campaign
                                     reserved against
        status           (str)   -- "ACTIVE" | "EXPIRED" | "REJECTED"
        reserved_amount  (Decimal)
        expires_at       (Decimal, epoch seconds)

Install: pip install boto3 tenacity
"""

from __future__ import annotations

import logging
import os
import time
from decimal import Decimal
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Attr

from campaign_store import CampaignBudgetStore, get_campaign_store
from audit_trail_dynamo import get_audit_trail

logger = logging.getLogger("agentictrade.campaign_scheduler")

CAMPAIGN_TABLE = os.environ.get("CAMPAIGN_TABLE", "agentictrade-dev-campaigns")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def _scan_active_expired_campaigns(table, now_epoch: float) -> list[dict]:
    """Scans for campaign items (PK begins with 'CAMPAIGN#') that are
    still ACTIVE but whose expires_at has passed. For production scale
    this should be a GSI on status+expires_at rather than a table scan;
    kept as a scan here for clarity, matching the note already left in
    audit_trail_dynamo.py about the same tradeoff."""
    items: list[dict] = []
    scan_kwargs = {
        "FilterExpression": (
            Attr("SK").eq("STATE")
            & Attr("status").eq("ACTIVE")
            & Attr("expires_at").lt(Decimal(str(now_epoch)))
            & Attr("PK").begins_with("CAMPAIGN#")
        )
    }
    resp = table.scan(**scan_kwargs)
    items.extend(resp.get("Items", []))
    while "LastEvaluatedKey" in resp:
        resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], **scan_kwargs)
        items.extend(resp.get("Items", []))
    return items


def expire_stale_campaigns(
    store: Optional[CampaignBudgetStore] = None,
    period_label: Optional[str] = None,
    now_epoch: Optional[float] = None,
) -> list[dict]:
    """
    Finds every ACTIVE campaign past its expires_at (optionally scoped
    to a single period_label) and releases its reserved budget back to
    the pool via store.release(), marking the campaign item EXPIRED.

    Returns the list of campaign records that were expired, for
    logging/observability by the caller (e.g. a Lambda handler).
    """
    store = store or get_campaign_store()
    audit = get_audit_trail()
    now_epoch = now_epoch if now_epoch is not None else time.time()

    table = store._table  # reuse the already-configured boto3 Table resource

    candidates = _scan_active_expired_campaigns(table, now_epoch)
    if period_label is not None:
        candidates = [c for c in candidates if c.get("period_label") == period_label]

    expired: list[dict] = []

    for item in candidates:
        campaign_id = item["campaign_id"]
        campaign_period = item["period_label"]
        reserved_amount = Decimal(item["reserved_amount"])

        try:
            store.release(campaign_period, reserved_amount)
        except Exception:
            logger.exception(
                "Failed to release budget for expired campaign_id=%s period=%s",
                campaign_id, campaign_period,
            )
            continue

        table.update_item(
            Key={"PK": item["PK"], "SK": item["SK"]},
            UpdateExpression="SET #s = :expired",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":expired": "EXPIRED"},
        )

        audit.log(
            "CAMPAIGN_EXPIRED_RELEASED",
            campaign_id,
            {
                "campaign_id": campaign_id,
                "period_label": campaign_period,
                "reserved_amount": reserved_amount,
                "expires_at": item["expires_at"],
                "released_at": Decimal(str(now_epoch)),
            },
        )

        logger.info(
            "Released expired campaign campaign_id=%s period=%s amount=%s",
            campaign_id, campaign_period, reserved_amount,
        )
        expired.append(item)

    return expired


def lambda_handler(event, context):
    """Entry point for the EventBridge-triggered Lambda."""
    expired = expire_stale_campaigns()
    return {
        "statusCode": 200,
        "expired_count": len(expired),
        "expired_campaign_ids": [c["campaign_id"] for c in expired],
    }