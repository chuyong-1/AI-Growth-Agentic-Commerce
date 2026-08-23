# ============================================================
# FILE: audit_trail_dynamo.py
# ============================================================
"""
Durable, hash-chained, append-only audit trail — DynamoDB-backed.

Same external contract as audit_trail.AuditTrail (log, verify_integrity,
history_for_cart, dump, pretty_print) so it's a drop-in replacement
everywhere `AUDIT` is imported. The critical addition over the
in-memory version: this survives process restarts and is safe under
concurrent writers (multiple Lambda invocations appending at once).

Concurrency model
------------------
The chain's integrity depends on each entry knowing the true hash of
the entry immediately before it. Two concurrent writers naively doing
"read tail -> compute hash -> write" can both read the same tail and
each produce a valid-looking entry that both claim the same prev_hash
— silently forking the chain into two histories that both "verify."

This implementation prevents that with an atomic DynamoDB counter:
  1. A single item (PK="CHAIN#meta") holds the current `seq` and
     `tail_hash`, updated via UpdateItem with an atomic ADD on `seq`
     and a ConditionExpression that the OLD seq/tail_hash still match
     what this writer read.
  2. If another writer won the race, the ConditionalCheckFailedException
     causes an automatic retry against the new (correct) tail.
  3. Only after the counter update succeeds is the actual entry item
     written — so the counter is the single source of truth for chain
     ordering, and entries are append-only by construction (no update
     path exists in this module at all).

Table schema (see serverless.yml):
    PK (S)  — "CHAIN#meta" for the singleton counter, or "ENTRY#<seq>"
              for each audit entry
    SK (S)  — "META" for the counter, "META" for entries too (single-item
              partitions), with a GSI on cart_id for history_for_cart()

Install: pip install boto3 tenacity
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

logger = logging.getLogger("agentictrade.audit_trail_dynamo")

AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "agentictrade-dev-audit")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
GENESIS_HASH = "0" * 64
COUNTER_PK = "CHAIN#meta"
COUNTER_SK = "META"


def _default(o: Any):
    if isinstance(o, Decimal):
        return str(o)
    raise TypeError(f"Not JSON serializable: {o!r}")


class ChainWriteConflict(Exception):
    """Raised internally when a concurrent writer wins the race; the
    public log() call retries transparently and this should never
    escape to a caller under normal retry limits."""


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


class DynamoAuditTrail:
    """Durable, hash-chained, append-only audit log. `log()` is the
    only write path — no update/delete method exists in this class,
    by design, matching the in-memory version's invariant."""

    def __init__(self, table_name: str = AUDIT_TABLE, region_name: str = AWS_REGION):
        self._table_name = table_name
        self._resource = boto3.resource("dynamodb", region_name=region_name)
        self._table = self._resource.Table(table_name)
        self._ensure_counter_exists()

    def _ensure_counter_exists(self) -> None:
        try:
            self._table.put_item(
                Item={"PK": COUNTER_PK, "SK": COUNTER_SK, "seq": 0, "tail_hash": GENESIS_HASH},
                ConditionExpression="attribute_not_exists(PK)",
            )
            logger.info("Initialized audit chain counter in table=%s", self._table_name)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise  # already exists — expected on every call after the first

    @retry(
        retry=retry_if_exception_type(ChainWriteConflict),
        stop=stop_after_attempt(8),
        wait=wait_random_exponential(multiplier=0.02, max=0.5),
        reraise=True,
    )
    def log(self, event_type: str, cart_id: str, payload: dict) -> AuditEntry:
        # 1. Read the current tail atomically.
        counter_resp = self._table.get_item(Key={"PK": COUNTER_PK, "SK": COUNTER_SK}, ConsistentRead=True)
        counter_item = counter_resp.get("Item")
        if counter_item is None:
            self._ensure_counter_exists()
            counter_item = {"seq": 0, "tail_hash": GENESIS_HASH}

        current_seq = int(counter_item["seq"])
        current_tail_hash = str(counter_item["tail_hash"])

        entry = AuditEntry(
            seq=current_seq,
            timestamp=time.time(),
            event_type=event_type,
            cart_id=cart_id,
            payload=payload,
            prev_hash=current_tail_hash,
        )

        # 2. Atomically advance the counter — succeeds only if nobody else
        #    has moved the tail since we read it (optimistic concurrency).
        try:
            self._table.update_item(
                Key={"PK": COUNTER_PK, "SK": COUNTER_SK},
                UpdateExpression="SET seq = :new_seq, tail_hash = :new_hash",
                ConditionExpression="seq = :old_seq AND tail_hash = :old_hash",
                ExpressionAttributeValues={
                    ":new_seq": current_seq + 1,
                    ":new_hash": entry.entry_hash,
                    ":old_seq": current_seq,
                    ":old_hash": current_tail_hash,
                },
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.debug("Chain write conflict at seq=%s, retrying", current_seq)
                raise ChainWriteConflict() from e
            raise

        # 3. Only now persist the entry itself — the counter update above
        #    is what defines ordering; this write can't create a fork.
        self._table.put_item(
            Item={
                "PK": f"ENTRY#{entry.seq:012d}",
                "SK": "META",
                "cart_id": entry.cart_id,
                "gsi1_pk": f"CART#{entry.cart_id}",
                "gsi1_sk": f"SEQ#{entry.seq:012d}",
                "seq": entry.seq,
                "timestamp": Decimal(str(entry.timestamp)),
                "event_type": entry.event_type,
                "payload_json": json.dumps(entry.payload, default=_default),
                "prev_hash": entry.prev_hash,
                "entry_hash": entry.entry_hash,
            }
        )

        return entry

    def _load_all_entries_ordered(self) -> list[AuditEntry]:
        items = []
        scan_kwargs = {"FilterExpression": "SK = :meta", "ExpressionAttributeValues": {":meta": "META"}}
        # Scanning by PK prefix ENTRY# — for production scale, replace with
        # a GSI on a constant partition + seq range key rather than a scan.
        resp = self._table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        while "LastEvaluatedKey" in resp:
            resp = self._table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], **scan_kwargs)
            items.extend(resp.get("Items", []))

        items = [i for i in items if i["PK"] != COUNTER_PK]
        items.sort(key=lambda i: int(i["seq"]))

        entries = []
        for i in items:
            e = AuditEntry(
                seq=int(i["seq"]),
                timestamp=float(i["timestamp"]),
                event_type=i["event_type"],
                cart_id=i["cart_id"],
                payload=json.loads(i["payload_json"]),
                prev_hash=i["prev_hash"],
            )
            # Trust the stored hash for comparison in verify_integrity
            # rather than only the recomputed one, so tampering with the
            # stored entry_hash itself is also detectable.
            e.entry_hash = i["entry_hash"]
            entries.append(e)
        return entries

    def verify_integrity(self) -> bool:
        entries = self._load_all_entries_ordered()
        prev_hash = GENESIS_HASH
        for entry in entries:
            if entry.prev_hash != prev_hash:
                logger.error("Chain broken at seq=%s: prev_hash mismatch", entry.seq)
                return False
            if entry.entry_hash != entry._compute_hash():
                logger.error("Chain broken at seq=%s: entry_hash does not match recomputed hash "
                             "(payload or metadata was tampered with)", entry.seq)
                return False
            prev_hash = entry.entry_hash
        return True

    def history_for_cart(self, cart_id: str) -> list[dict]:
        resp = self._table.query(
            IndexName="gsi1",
            KeyConditionExpression=Key("gsi1_pk").eq(f"CART#{cart_id}"),
            ScanIndexForward=True,
        )
        items = resp.get("Items", [])
        while "LastEvaluatedKey" in resp:
            resp = self._table.query(
                IndexName="gsi1",
                KeyConditionExpression=Key("gsi1_pk").eq(f"CART#{cart_id}"),
                ScanIndexForward=True,
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
            items.extend(resp.get("Items", []))

        return [
            {
                "seq": int(i["seq"]),
                "timestamp": float(i["timestamp"]),
                "event_type": i["event_type"],
                "cart_id": i["cart_id"],
                "payload": json.loads(i["payload_json"]),
                "prev_hash": i["prev_hash"],
                "entry_hash": i["entry_hash"],
            }
            for i in items
        ]

    def dump(self) -> list[dict]:
        return [e.to_dict() for e in self._load_all_entries_ordered()]

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


# Lazily-constructed module-level singleton, mirroring the ergonomics of
# the in-memory `AUDIT` you already import elsewhere as `from audit_trail
# import AUDIT`. Constructed on first access so importing this module
# doesn't require live AWS credentials (useful for local unit tests that
# monkeypatch it out anyway).
_AUDIT_SINGLETON: Optional[DynamoAuditTrail] = None


def get_audit_trail() -> DynamoAuditTrail:
    global _AUDIT_SINGLETON
    if _AUDIT_SINGLETON is None:
        _AUDIT_SINGLETON = DynamoAuditTrail()
    return _AUDIT_SINGLETON