# ============================================================
# FILE: audit_trail.py
# ============================================================
"""
Immutable, hash-chained audit trail.

Every state transition, every LLM rationale, and every gatekeeper
verdict is appended here. Entries are hash-linked (like a mini
blockchain) so any post-hoc tampering with a historical entry breaks
the chain and is detectable — this is the "completely explainable +
immutable audit trail" requirement from The Bar.

This is intentionally dependency-free (stdlib only) so it can be
swapped for an actual append-only store (WORM S3 bucket, ledger DB,
etc.) in production without changing calling code.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional


def _default(o: Any):
    if isinstance(o, Decimal):
        return str(o)
    raise TypeError(f"Not JSON serializable: {o!r}")


@dataclass
class AuditEntry:
    seq: int
    timestamp: float
    event_type: str          # e.g. "UPSELL_PROPOSED", "GATEKEEPER_VERDICT", "RAZORPAY_ORDER_CREATED"
    cart_id: str
    payload: dict
    prev_hash: str
    entry_hash: str = field(init=False)

    def __post_init__(self):
        self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        body = json.dumps(
            {
                "seq": self.seq,
                "timestamp": self.timestamp,
                "event_type": self.event_type,
                "cart_id": self.cart_id,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            default=_default,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "cart_id": self.cart_id,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


class AuditTrail:
    """Append-only, hash-chained log. `log()` is the only write path;
    there is deliberately no update/delete method."""

    GENESIS_HASH = "0" * 64

    def __init__(self):
        self._entries: list[AuditEntry] = []

    def log(self, event_type: str, cart_id: str, payload: dict) -> AuditEntry:
        prev_hash = self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH
        entry = AuditEntry(
            seq=len(self._entries),
            timestamp=time.time(),
            event_type=event_type,
            cart_id=cart_id,
            payload=payload,
            prev_hash=prev_hash,
        )
        self._entries.append(entry)
        return entry

    def verify_integrity(self) -> bool:
        """Walks the chain and confirms no entry has been tampered with."""
        prev_hash = self.GENESIS_HASH
        for entry in self._entries:
            if entry.prev_hash != prev_hash:
                return False
            if entry.entry_hash != entry._compute_hash():
                return False
            prev_hash = entry.entry_hash
        return True

    def history_for_cart(self, cart_id: str) -> list[dict]:
        return [e.to_dict() for e in self._entries if e.cart_id == cart_id]

    def dump(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    def pretty_print(self, cart_id: Optional[str] = None) -> str:
        entries = self.history_for_cart(cart_id) if cart_id else self.dump()
        lines = []
        for e in entries:
            lines.append(
                f"[{e['seq']:03d}] {e['event_type']:<24} "
                f"cart={e['cart_id']} hash={e['entry_hash'][:10]}… "
                f"payload={json.dumps(e['payload'], default=_default)}"
            )
        return "\n".join(lines)


# Module-level singleton used by the demo graph. In a real service this
# would be injected/DI'd rather than global.
AUDIT = AuditTrail()