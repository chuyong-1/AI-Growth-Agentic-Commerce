# ============================================================
# FILE: session_store.py
# ============================================================
"""
Durable session/cart store — DynamoDB-backed replacement for the
in-memory SESSIONS / GATEWAYS dicts in api.py.

Stores the full CartState (Pydantic-serialized) plus enough gateway
metadata to reconstruct a RazorpayGateway per request. LangChain
message history is stored separately as a JSON blob since it's
conversation-turn scoped, not part of the financial record.

Optimistic concurrency: every write includes the cart's current
`version` in a ConditionExpression, so two concurrent requests
operating on the same cart_id (e.g. a double-submit) can't silently
clobber each other — the loser gets a clean, catchable conflict
instead of a lost update.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from schema import CartState

logger = logging.getLogger("agentictrade.session_store")

SESSION_TABLE = os.environ.get("SESSION_TABLE", "agentictrade-dev-sessions")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


class SessionConflictError(Exception):
    """Raised when a write loses an optimistic-concurrency race — the
    caller should re-read and retry rather than silently overwrite."""


class SessionStore:
    def __init__(self, table_name: str = SESSION_TABLE, region_name: str = AWS_REGION):
        self._table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)

    def get_cart(self, cart_id: str) -> Optional[CartState]:
        resp = self._table.get_item(Key={"PK": f"CART#{cart_id}", "SK": "STATE"})
        item = resp.get("Item")
        if not item:
            return None
        return CartState.model_validate_json(item["cart_json"])

    def get_version(self, cart_id: str) -> int:
        resp = self._table.get_item(Key={"PK": f"CART#{cart_id}", "SK": "STATE"})
        item = resp.get("Item")
        return int(item["version"]) if item else 0

    def save_cart(self, cart: CartState, expected_version: Optional[int] = None) -> int:
        """Writes the cart. If expected_version is given, the write only
        succeeds if the stored version still matches — protects against
        two concurrent requests (e.g. a retried checkout) stomping on
        each other's cart mutations."""
        new_version = (expected_version or 0) + 1
        item = {
            "PK": f"CART#{cart.cart_id}",
            "SK": "STATE",
            "cart_id": cart.cart_id,
            "cart_json": cart.model_dump_json(),
            "status": cart.status.value,
            "version": new_version,
        }
        try:
            if expected_version is None:
                self._table.put_item(Item=item)
            else:
                self._table.put_item(
                    Item=item,
                    ConditionExpression="attribute_not_exists(version) OR version = :v",
                    ExpressionAttributeValues={":v": expected_version},
                )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise SessionConflictError(
                    f"Cart {cart.cart_id} was modified concurrently "
                    f"(expected version {expected_version})"
                ) from e
            raise
        return new_version

    def get_history(self, cart_id: str) -> list[dict]:
        resp = self._table.get_item(Key={"PK": f"CART#{cart_id}", "SK": "HISTORY"})
        item = resp.get("Item")
        if not item:
            return []
        return json.loads(item["history_json"])

    def save_history(self, cart_id: str, history_serializable: list[dict]) -> None:
        self._table.put_item(
            Item={
                "PK": f"CART#{cart_id}",
                "SK": "HISTORY",
                "cart_id": cart_id,
                "history_json": json.dumps(history_serializable),
            }
        )

    def get_failure_injection(self, cart_id: str) -> Optional[str]:
        resp = self._table.get_item(Key={"PK": f"CART#{cart_id}", "SK": "GATEWAY_STATE"})
        item = resp.get("Item")
        return item.get("force_failure_mode") if item else None

    def set_failure_injection(self, cart_id: str, mode: Optional[str]) -> None:
        self._table.put_item(
            Item={
                "PK": f"CART#{cart_id}",
                "SK": "GATEWAY_STATE",
                "cart_id": cart_id,
                "force_failure_mode": mode or "",
            }
        )


_SESSION_STORE_SINGLETON: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _SESSION_STORE_SINGLETON
    if _SESSION_STORE_SINGLETON is None:
        _SESSION_STORE_SINGLETON = SessionStore()
    return _SESSION_STORE_SINGLETON