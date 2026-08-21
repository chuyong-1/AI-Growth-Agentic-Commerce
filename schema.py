# ============================================================
# FILE: schema.py
# ============================================================
"""
Pydantic schemas for the Agentic Commerce system.

Design principle: the CartState is the single source of truth that
flows through the graph. It is intentionally "over-typed" (strict
validators, Decimal for money, no floats) because this is the object
the PaymentGatekeeper will mathematically audit before any rupee
moves. Garbage-in/garbage-out is not acceptable when money is
involved.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ------------------------------------------------------------------
# Money is ALWAYS Decimal, never float. Floats cause cent/paisa drift
# which is unacceptable for a system whose entire mandate is
# mathematically bounded correctness.
# ------------------------------------------------------------------
def to_paise(rupees: Decimal) -> int:
    """Razorpay's API operates in the smallest currency unit (paise)."""
    return int((rupees * 100).to_integral_value(rounding=ROUND_HALF_UP))


class Currency(str, Enum):
    INR = "INR"


# ------------------------------------------------------------------
# Catalog: the agent-readable "menu" the Upsell Agent is allowed to
# operate over. max_discount_pct is a HARD CEILING enforced later
# by the PaymentGatekeeper, not a suggestion to the LLM.
# ------------------------------------------------------------------
class CatalogItem(BaseModel):
    sku: str
    name: str
    base_price: Decimal = Field(gt=0, description="Price in INR, exclusive of discount")
    max_discount_pct: Decimal = Field(
        ge=0, le=Decimal("50.0"),
        description="Hard ceiling on discount % the agent may ever apply to this SKU"
    )
    is_upsell_eligible: bool = True

    @field_validator("base_price", "max_discount_pct", mode="before")
    @classmethod
    def coerce_decimal(cls, v):
        return Decimal(str(v))

    model_config = {"frozen": True}  # catalog items are immutable once loaded


class Catalog(BaseModel):
    """In-memory catalog keyed by SKU. In production this would be
    backed by a merchant DB, but the schema/contract is what matters
    for the agent boundary."""
    items: dict[str, CatalogItem]

    def get(self, sku: str) -> CatalogItem:
        if sku not in self.items:
            raise KeyError(f"SKU '{sku}' does not exist in catalog")
        return self.items[sku]


# ------------------------------------------------------------------
# Line items & Cart
# ------------------------------------------------------------------
class LineItem(BaseModel):
    sku: str
    name: str
    quantity: int = Field(gt=0, le=99)
    unit_price: Decimal = Field(gt=0)
    discount_pct: Decimal = Field(default=Decimal("0.0"), ge=0, le=Decimal("100.0"))
    added_by_upsell: bool = False
    upsell_rationale: Optional[str] = Field(
        default=None, description="LLM's stated reason for adding/upselling this item"
    )

    @field_validator("unit_price", "discount_pct", mode="before")
    @classmethod
    def coerce_decimal(cls, v):
        return Decimal(str(v))

    @property
    def line_total(self) -> Decimal:
        gross = self.unit_price * self.quantity
        discount_amount = (gross * self.discount_pct / Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return (gross - discount_amount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class CartStatus(str, Enum):
    DRAFT = "DRAFT"                 # agent is still composing/upselling
    PENDING_AUDIT = "PENDING_AUDIT" # submitted to PaymentGatekeeper
    AUDIT_FAILED = "AUDIT_FAILED"   # gatekeeper rejected it
    AUDITED_OK = "AUDITED_OK"       # gatekeeper approved, ready for Razorpay
    ORDER_CREATED = "ORDER_CREATED"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    RECOVERED = "RECOVERED"
    COMPLETED = "COMPLETED"


class CartState(BaseModel):
    """
    The canonical, strictly-validated state object passed through
    every LangGraph node. `total` is NEVER trusted from upstream —
    it is always recomputed and cross-checked by the PaymentGatekeeper.
    """
    cart_id: str = Field(default_factory=lambda: f"cart_{uuid4().hex[:12]}")
    currency: Currency = Currency.INR
    line_items: list[LineItem] = Field(default_factory=list)
    status: CartStatus = CartStatus.DRAFT

    # Populated by the agent, MUST be validated against actual sum
    declared_total: Decimal = Field(default=Decimal("0.00"))

    # Populated post-audit / post-Razorpay
    razorpay_order_id: Optional[str] = None
    gatekeeper_notes: list[str] = Field(default_factory=list)

    @field_validator("declared_total", mode="before")
    @classmethod
    def coerce_decimal(cls, v):
        return Decimal(str(v))

    @property
    def computed_total(self) -> Decimal:
        """The ONLY value trusted for actual payment collection."""
        return sum((li.line_total for li in self.line_items), Decimal("0.00")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @model_validator(mode="after")
    def non_empty_cart_ids(self):
        if not self.cart_id:
            raise ValueError("cart_id cannot be empty")
        return self


# ------------------------------------------------------------------
# Agent action envelope — every proposed mutation to the cart is
# wrapped in this so the audit trail has an explicit record of what
# the LLM *intended* to do, before the gatekeeper decides.
# ------------------------------------------------------------------
class ProposedAction(BaseModel):
    action_type: str  # "ADD_ITEM" | "APPLY_DISCOUNT" | "FINALIZE"
    sku: Optional[str] = None
    discount_pct: Optional[Decimal] = None
    rationale: str = Field(description="LLM's natural-language justification, mandatory")

    @field_validator("discount_pct", mode="before")
    @classmethod
    def coerce_decimal(cls, v):
        return None if v is None else Decimal(str(v))