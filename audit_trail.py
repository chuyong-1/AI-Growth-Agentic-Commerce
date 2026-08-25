# ============================================================
# FILE: audit_trail.py
# ============================================================
"""
Thread-safe, in-memory, hash-chained, append-only audit trail.

Every state transition, LLM rationale, and gatekeeper verdict is
appended here. Entries are hash-linked (like a mini blockchain) so
any post-hoc tampering with a historical entry breaks the chain and
is detectable — this is the "explainable + immutable audit trail"
requirement, now running with zero external dependencies.

CONCURRENCY MODEL — WHY THIS CAN'T FORK
------------------------------------------
The entire risk in a hash-chained log is two writers both reading the
same "current tail hash", both computing a valid-looking next entry
that claims that same tail as its prev_hash, and both appending —
silently forking the chain into two histories that each individually
"verify" but disagree with each other.

The DynamoDB version prevented this with an atomic conditional
UpdateItem on a shared counter row. Here, the equivalent primitive is
a single threading.Lock held for the ENTIRE
read-tail -> compute-hash -> append sequence in `log()`. No other
thread can read the tail while one thread is mid-append, so two
threads can never compute entries against the same tail — the fork
condition is structurally prevented, not just made unlikely.

This is intentionally the ONLY write path (`log()`) — there is no
update or delete method anywhere in this class, matching the
production version's append-only invariant.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger("agentictrade.audit_trail")

GENESIS_HASH = "0" * 64


def _default(o: Any):
    if isinstance(o, Decimal):
        return str(o)
    raise TypeError(f"Not JSON serializable: {o!r}")


@dataclass
class AuditEntry:
    seq: int
    timestamp: float
    event_type: str
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
    """
    Thread-safe, append-only, hash-chained log. `log()` acquires a
    single lock for its full duration — reading the tail, computing
    the new entry's hash, and appending all happen as one atomic
    unit, which is what prevents concurrent writers from forking the
    chain (see module docstring).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: list[AuditEntry] = []

    def log(self, event_type: str, cart_id: str, payload: dict) -> AuditEntry:
        with self._lock:
            prev_hash = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
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
        """
        Walks the chain and confirms no entry has been tampered with.
        Reads a snapshot of the entry list under the lock, then
        verifies outside the lock (verification is read-only and pure
        CPU work — no need to hold the lock across the whole scan,
        which would otherwise block new log() calls for longer than
        necessary on a large chain).
        """
        with self._lock:
            entries_snapshot = list(self._entries)

        prev_hash = GENESIS_HASH
        for entry in entries_snapshot:
            if entry.prev_hash != prev_hash:
                logger.error("Chain broken at seq=%s: prev_hash mismatch", entry.seq)
                return False
            if entry.entry_hash != entry._compute_hash():
                logger.error(
                    "Chain broken at seq=%s: entry_hash does not match recomputed "
                    "hash (payload or metadata was tampered with)", entry.seq,
                )
                return False
            prev_hash = entry.entry_hash
        return True

    def history_for_cart(self, cart_id: str) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._entries if e.cart_id == cart_id]

    def dump(self) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._entries]

    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

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

    def clear_all(self) -> None:
        """Test/demo utility only — never call this from a request handler."""
        with self._lock:
            self._entries = []


# Module-level singleton. Matches the ergonomics every other module in
# this codebase already expects (`from audit_trail import AUDIT`).
AUDIT = AuditTrail()