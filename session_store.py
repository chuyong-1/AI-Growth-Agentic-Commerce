# ============================================================
# FILE: session_store.py
# ============================================================
"""
In-memory session/cart store — thread-safe, zero-dependency
replacement for the DynamoDB-backed version.

WHY A LOCK PER OPERATION, NOT ONE GLOBAL LOCK
----------------------------------------------
A single global lock around every read/write would serialize all
traffic across every cart, which defeats the purpose of demonstrating
per-resource concurrency control. Instead, this uses ONE lock to
guard the dict itself (protecting Python's dict from concurrent
mutation, which is technically GIL-safe for single ops but not for
compound read-modify-write sequences), and the optimistic-versioning
logic below is what actually simulates DynamoDB's per-item
ConditionExpression semantics: read version -> compute -> compare-
and-swap under the lock.

This preserves the exact contract the DynamoDB version had:
  - save_cart(cart, expected_version) raises SessionConflictError if
    the stored version has moved since the caller last read it.
  - Nothing here silently overwrites a concurrent write.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

from schema import CartState

logger = logging.getLogger("agentictrade.session_store")


class SessionConflictError(Exception):
    """Raised when a write loses an optimistic-concurrency race — the
    caller should re-read and retry rather than silently overwrite."""


@dataclass
class _CartRecord:
    cart_json: str
    version: int


@dataclass
class _SessionState:
    carts: dict = field(default_factory=dict)              # cart_id -> _CartRecord
    history: dict = field(default_factory=dict)             # cart_id -> list[dict]
    gateway_state: dict = field(default_factory=dict)       # cart_id -> str | None


class SessionStore:
    """
    Thread-safe, in-process store. Every public method acquires the
    single internal lock for the duration of its critical section —
    short-lived (dict read/write only, no I/O), so contention is
    negligible even under concurrent request handling in a single
    uvicorn worker.

    NOTE ON SCOPE: this intentionally does NOT persist across process
    restarts — that's the tradeoff explicitly requested for a
    zero-friction local demo. In a real deployment this class's
    interface is what you'd re-implement against DynamoDB, Redis, or
    Postgres; nothing in api.py needs to change to swap it, since
    every caller only depends on this class's public method
    signatures, not its storage mechanism.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = _SessionState()

    # ---------------- cart state ----------------
    def get_cart(self, cart_id: str) -> Optional[CartState]:
        with self._lock:
            record = self._state.carts.get(cart_id)
        if record is None:
            return None
        return CartState.model_validate_json(record.cart_json)

    def get_version(self, cart_id: str) -> int:
        with self._lock:
            record = self._state.carts.get(cart_id)
        return record.version if record else 0

    def save_cart(self, cart: CartState, expected_version: Optional[int] = None) -> int:
        """
        Optimistic-locking write, mirroring DynamoDB's
        ConditionExpression semantics exactly:
          - expected_version is None -> unconditional write (first
            save of a brand-new cart).
          - expected_version is an int -> the write only succeeds if
            the CURRENT stored version equals expected_version. If a
            concurrent request already advanced the version (e.g. a
            double-submitted checkout), this raises
            SessionConflictError instead of silently clobbering that
            newer write — the same guarantee the DynamoDB
            ConditionExpression gave us, just enforced under a Python
            lock instead of a database-side condition.
        """
        with self._lock:
            existing = self._state.carts.get(cart.cart_id)
            current_version = existing.version if existing else 0

            if expected_version is not None and current_version != expected_version:
                raise SessionConflictError(
                    f"Cart {cart.cart_id} was modified concurrently "
                    f"(expected version {expected_version}, found {current_version})"
                )

            new_version = current_version + 1
            self._state.carts[cart.cart_id] = _CartRecord(
                cart_json=cart.model_dump_json(),
                version=new_version,
            )
            return new_version

    # ---------------- conversation history ----------------
    def get_history(self, cart_id: str) -> list[dict]:
        with self._lock:
            return list(self._state.history.get(cart_id, []))

    def save_history(self, cart_id: str, history_serializable: list[dict]) -> None:
        with self._lock:
            self._state.history[cart_id] = history_serializable

    # ---------------- test-hook: one-shot failure injection ----------------
    def get_failure_injection(self, cart_id: str) -> Optional[str]:
        with self._lock:
            mode = self._state.gateway_state.get(cart_id)
        return mode or None

    def set_failure_injection(self, cart_id: str, mode: Optional[str]) -> None:
        with self._lock:
            self._state.gateway_state[cart_id] = mode or ""

    # ---------------- housekeeping ----------------
    def cart_count(self) -> int:
        """Exposed for a lightweight /api/health memory-footprint signal —
        see the memory-bounding note in api.py's health endpoint."""
        with self._lock:
            return len(self._state.carts)

    def clear_all(self) -> None:
        """Test/demo utility only — never call this from a request handler."""
        with self._lock:
            self._state = _SessionState()


_SESSION_STORE_SINGLETON: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _SESSION_STORE_SINGLETON
    if _SESSION_STORE_SINGLETON is None:
        _SESSION_STORE_SINGLETON = SessionStore()
    return _SESSION_STORE_SINGLETON