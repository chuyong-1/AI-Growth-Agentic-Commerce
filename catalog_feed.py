# ============================================================
# FILE: catalog_feed.py
# ============================================================
"""
Agent-readable catalog feed.

This is the surface an EXTERNAL AI shopping agent (not our own
conversational UI) uses to discover what this merchant sells, at what
price, under what negotiable bounds, and how to transact — without
needing a human-facing chat turn first. Modeled loosely on the shape
of ACP (Agentic Commerce Protocol) product feeds and UAP-style agent
discovery documents; adjust field names if you're targeting a
specific published protocol version.

Design principle consistent with the rest of this repo: this feed
declares CEILINGS (max_discount_pct, bundle rules), never live
authorization. An AI buyer reading this feed still has to go through
propose -> gatekeeper -> audit exactly like a human-driven chat
session. The feed makes the merchant *discoverable and negotiable*;
it does not grant purchasing authority to whoever reads it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from schema import Catalog


MERCHANT_META = {
    "merchant_id": "agentictrade_demo_merchant",
    "display_name": "AgenticTrade Demo Store",
    "currency": "INR",
    "settlement_rail": "razorpay",
    "environment": "test",  # flip to "live" only with real keys + real review
}


def _item_to_feed_entry(item) -> dict:
    return {
        "sku": item.sku,
        "name": item.name,
        "price": {
            "amount": str(item.base_price),
            "currency": "INR",
        },
        "negotiation": {
            "discount_negotiable": item.max_discount_pct > 0,
            "max_discount_pct": str(item.max_discount_pct),
            "bundle_eligible": item.is_upsell_eligible,
        },
        "transact": {
            # What an external agent must POST, and where, to act on this SKU.
            # Every one of these still lands on ProposedAction -> gatekeeper.
            "propose_endpoint": "/api/agent/propose",
            "supported_actions": ["ADD_ITEM", "APPLY_DISCOUNT", "FINALIZE"],
            "requires_human_present": False,
        },
    }


def build_catalog_feed(catalog: Catalog) -> dict:
    """The core agent-readable feed. Every SKU's negotiable ceiling is
    published explicitly — an external agent should never need to
    guess or probe for the discount limit; it's declared up front,
    and it's the SAME number the gatekeeper enforces server-side."""
    return {
        "protocol": "agentic-commerce-feed",
        "version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "merchant": MERCHANT_META,
        "catalog": [_item_to_feed_entry(item) for item in catalog.items.values()],
        "policies": {
            "audit": "Every transaction is logged to an immutable, hash-chained "
                     "audit trail, independently retrievable at "
                     "/api/audit/{cart_id}.",
            "guarantee": "No discount beyond the published max_discount_pct per SKU "
                         "will ever be honored, regardless of what is proposed in "
                         "a negotiation message.",
            "max_transaction_ceiling": {"amount": "500000.00", "currency": "INR"},
        },
    }


def build_well_known_manifest(base_url: str) -> dict:
    """
    Served at /.well-known/agentic-commerce.json — the discovery
    entrypoint. An AI buyer agent that only knows this store's domain
    should be able to fetch this ONE url and learn everything it needs:
    where the catalog is, where to propose actions, where to audit.
    """
    return {
        "protocol": "agentic-commerce-manifest",
        "version": "0.1",
        "merchant": MERCHANT_META,
        "endpoints": {
            "catalog_feed": f"{base_url}/api/catalog/feed",
            "propose_action": f"{base_url}/api/agent/propose",
            "conversational_checkout": f"{base_url}/api/chat",
            "audit_trail": f"{base_url}/api/audit/{{cart_id}}",
            "payment_verification": f"{base_url}/api/checkout/verify",
        },
        "auth": {
            "type": "none",
            "note": "Test-mode demo merchant — no API key required to browse "
                     "the catalog or propose actions. Production deployments "
                     "should require signed agent identity per ACP/UAP spec.",
        },
        "guardrails_summary": [
            "Every proposal requires an explicit, logged rationale.",
            "Every discount is capped per-SKU and enforced server-side, independent of any agent's claim.",
            "Every money-moving action passes a deterministic, non-LLM gatekeeper before Razorpay is ever called.",
            "Every decision — proposal, verdict, payment event — is appended to a SHA-256 hash-chained, tamper-evident audit log.",
        ],
    }