# ============================================================
# FILE: razorpay_client.py
# ============================================================
"""
Thin, defensive wrapper around the Razorpay Python SDK, scoped to
TEST MODE only. This module never receives an already-approved
CartState from anywhere other than the PaymentGatekeeper — that
invariant is enforced by the calling graph node, not here, but we
still re-validate defensively (defense in depth).

Install: pip install razorpay
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import razorpay
from razorpay.errors import SignatureVerificationError

from schema import CartState, to_paise


class RazorpayNetworkTimeout(Exception):
    """Simulated transport-level failure (not a real razorpay exception,
    but modeled the same way one would be handled)."""


class RazorpayOrderMismatchError(Exception):
    """Raised if the amount Razorpay echoes back doesn't match what we
    submitted — a last-mile sanity check before we ever tell the user
    'payment link ready'."""


@dataclass
class RazorpayOrderResult:
    order_id: str
    amount_paise: int
    currency: str
    status: str
    raw: dict


class RazorpayGateway:
    def __init__(self, key_id: Optional[str] = None, key_secret: Optional[str] = None):
        # TEST MODE credentials only. Never hardcode live keys.
        self.key_id = key_id or os.environ.get("RAZORPAY_TEST_KEY_ID", "rzp_test_xxxxxxxxxxxxxx")
        self.key_secret = key_secret or os.environ.get("RAZORPAY_TEST_KEY_SECRET", "test_secret_xxxxxxxxxxxx")
        self.client = razorpay.Client(auth=(self.key_id, self.key_secret))
        # Purely for the graceful-failure demo — lets us deterministically
        # trigger a failure without needing real flaky network conditions.
        self._force_failure_mode: Optional[str] = None

    def force_failure(self, mode: Optional[str]) -> None:
        """Test hook: 'timeout' | 'signature' | None"""
        self._force_failure_mode = mode

    def create_order(self, cart: CartState, receipt_prefix: str = "agentic_cart") -> RazorpayOrderResult:
        """
        Creates a Razorpay Order from an ALREADY AUDITED CartState.
        Callers must guarantee cart.status == CartStatus.AUDITED_OK
        before invoking this — the graph enforces that via the
        conditional edge out of PaymentGatekeeper.
        """
        if cart.computed_total <= 0:
            raise ValueError("Refusing to create a Razorpay order for a non-positive amount")

        amount_paise = to_paise(cart.computed_total)

        # ---- Simulated failure injection for the "graceful failure" demo ----
        if self._force_failure_mode == "timeout":
            raise RazorpayNetworkTimeout(
                "Simulated network timeout while calling Razorpay order.create"
            )
        if self._force_failure_mode == "signature":
            raise SignatureVerificationError(
                "Simulated signature mismatch on Razorpay response"
            )

        try:
            order = self.client.order.create(
                {
                    "amount": amount_paise,
                    "currency": cart.currency.value,
                    "receipt": f"{receipt_prefix}_{cart.cart_id}",
                    "notes": {
                        "cart_id": cart.cart_id,
                        "line_item_count": str(len(cart.line_items)),
                        "source": "agentic-upsell-checkout",
                    },
                    "payment_capture": 1,
                }
            )
        except SignatureVerificationError:
            raise
        except Exception as e:
            # razorpay SDK raises generic errors for transport issues;
            # normalize to our own exception type for the graph to catch.
            raise RazorpayNetworkTimeout(str(e)) from e

        if order.get("amount") != amount_paise:
            raise RazorpayOrderMismatchError(
                f"Razorpay echoed amount {order.get('amount')} != expected {amount_paise}"
            )

        return RazorpayOrderResult(
            order_id=order["id"],
            amount_paise=order["amount"],
            currency=order["currency"],
            status=order.get("status", "created"),
            raw=order,
        )

    def verify_payment_signature(self, params: dict) -> bool:
        """
        params must contain: razorpay_order_id, razorpay_payment_id,
        razorpay_signature. Used post-checkout on the client callback.
        """
        try:
            self.client.utility.verify_payment_signature(params)
            return True
        except SignatureVerificationError:
            return False