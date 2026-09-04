# ============================================================
# FILE: campaign_scheduler.py
# ============================================================
"""
Scheduled housekeeping for the campaign budget system.

This is intentionally decoupled from the campaign_orchestrator's
propose/gatekeeper path — it only ever RELEASES budget that is no
longer legitimately held. It never reserves, approves, or extends
anything, so a bug here can under-spend the budget but can never
over-spend it.

WHY THE SWEEP LIVES IN THE STORE, NOT HERE
--------------------------------------------
The obvious implementation is "list active campaigns, filter the
expired ones, release each". That reintroduces a TOCTOU window: two
sweeps (or a sweep racing a manual cancel) can both observe the same
campaign as ACTIVE and each hand its reservation back, freeing twice
what was ever reserved. So the select-and-release is delegated to
CampaignBudgetStore.expire_due_campaigns(), where the status check
and the ledger decrement happen under one lock. This module's job is
only to decide WHEN to sweep and to write the audit entries.

Trigger it from cron, an APScheduler job, or POST /api/campaigns/expire.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from audit_trail import AUDIT
from campaign_store import CampaignBudgetStore, get_campaign_store

logger = logging.getLogger("agentictrade.campaign_scheduler")


def expire_stale_campaigns(
    store: Optional[CampaignBudgetStore] = None,
    period_label: Optional[str] = None,
    now_epoch: Optional[float] = None,
) -> list[dict]:
    """
    Releases every ACTIVE campaign past its expires_at back into its
    budget period, optionally scoped to a single period_label.

    Returns one summary dict per campaign actually expired by THIS
    call — campaigns another concurrent sweep already claimed are not
    included, so the caller's count is a true count of work done.
    """
    store = store or get_campaign_store()
    now_epoch = now_epoch if now_epoch is not None else time.time()

    expired = store.expire_due_campaigns(now_epoch, period_label=period_label)

    summaries: list[dict] = []
    for campaign in expired:
        AUDIT.log(
            "CAMPAIGN_EXPIRED_RELEASED",
            "campaign_system",
            {
                "campaign_id": campaign.campaign_id,
                "period_label": campaign.period_label,
                "target_sku": campaign.target_sku,
                "reserved_amount": str(campaign.reserved_amount),
                "expires_at": campaign.expires_at,
                "released_at": now_epoch,
            },
        )
        logger.info(
            "Released expired campaign campaign_id=%s period=%s amount=%s",
            campaign.campaign_id, campaign.period_label, campaign.reserved_amount,
        )
        summaries.append(
            {
                "campaign_id": campaign.campaign_id,
                "period_label": campaign.period_label,
                "target_sku": campaign.target_sku,
                "reserved_amount": str(campaign.reserved_amount),
            }
        )

    return summaries
