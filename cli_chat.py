# ============================================================
# FILE: cli_chat.py
# ============================================================
"""
Interactive terminal client for the Agentic Commerce backend.

Run with:
    python cli_chat.py
    (assumes `uvicorn api:app --reload` is running on localhost:8000)

Commands (typed instead of a chat message):
    /cart        show current cart contents & total
    /audit       show the hash-chained audit log for this cart
    /pay         create a real (test-mode) Razorpay order for the current cart
                 and print the checkout payload for manual verification
    /fail <mode> arm a simulated failure for the NEXT message
                 (mode = timeout | signature)
    /reset       start a brand-new cart/session
    /quit        exit
"""

from __future__ import annotations

import sys
from typing import Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

API_BASE = "http://localhost:8000"
console = Console()


class CliState:
    def __init__(self):
        self.cart_id: Optional[str] = None
        self.pending_failure: Optional[str] = None


def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]Agentic Commerce — Merchant Assistant[/bold cyan]\n"
            "[dim]Razorpay test-mode • LangGraph-audited checkout[/dim]\n\n"
            "Type naturally to chat. Commands: /cart /audit /pay /fail <mode> /reset /quit",
            border_style="cyan",
        )
    )


def render_cart(payload: dict):
    table = Table(title=f"Cart {payload['cart_id']}  —  status: {payload['cart_status']}", box=box.SIMPLE_HEAVY)
    table.add_column("SKU")
    table.add_column("Name")
    table.add_column("Qty", justify="right")
    table.add_column("Unit Price", justify="right")
    table.add_column("Discount %", justify="right")
    table.add_column("Upsell?", justify="center")

    for li in payload["line_items"]:
        table.add_row(
            li["sku"], li["name"], str(li["quantity"]), str(li["unit_price"]),
            str(li["discount_pct"]),
            "✓" if li.get("added_by_upsell") else "",
        )

    console.print(table)
    console.print(f"[bold]Computed total:[/bold] ₹{payload['computed_total']}")
    if payload.get("order_id"):
        console.print(f"[green]Razorpay order created:[/green] {payload['order_id']}")
    for note in payload.get("system_notes", []):
        console.print(Panel(note, title="System / Gatekeeper", border_style="yellow"))


def render_audit(entries: list[dict], chain_intact: bool):
    table = Table(title="Immutable Audit Trail", box=box.SIMPLE)
    table.add_column("#", justify="right")
    table.add_column("Event")
    table.add_column("Hash", overflow="fold")
    table.add_column("Payload", overflow="fold")

    for e in entries:
        table.add_row(
            str(e["seq"]),
            e["event_type"],
            e["entry_hash"][:12] + "…",
            str(e["payload"])[:120],
        )

    console.print(table)
    style = "green" if chain_intact else "bold red"
    label = "✓ chain intact" if chain_intact else "✗ TAMPERING DETECTED"
    console.print(f"[{style}]{label}[/{style}]")


def call_chat(client: httpx.Client, state: CliState, message: str) -> dict:
    body = {"cart_id": state.cart_id, "message": message}
    if state.pending_failure:
        body["simulate_failure"] = state.pending_failure
        console.print(f"[yellow]⚠ Simulated failure armed for this turn: {state.pending_failure}[/yellow]")
        state.pending_failure = None

    resp = client.post(f"{API_BASE}/api/chat", json=body, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    state.cart_id = payload["cart_id"]
    return payload


def call_audit(client: httpx.Client, cart_id: str):
    resp = client.get(f"{API_BASE}/api/audit/{cart_id}", timeout=30)
    if resp.status_code == 404:
        console.print("[dim]No audit history yet for this cart.[/dim]")
        return
    resp.raise_for_status()
    data = resp.json()
    render_audit(data["entries"], data["chain_intact"])


def call_pay(client: httpx.Client, state: CliState):
    """Drives the graph toward FINALIZE, which triggers PaymentGatekeeper
    -> Razorpay order.create if the cart is currently valid."""
    if not state.cart_id:
        console.print("[red]No active cart yet — say something first.[/red]")
        return
    payload = call_chat(client, state, "I'm ready to check out, please finalize my order.")
    render_cart(payload)
    if payload.get("order_id"):
        console.print(
            Panel(
                f"Order ID: [bold]{payload['order_id']}[/bold]\n"
                f"Amount: ₹{payload['computed_total']}\n\n"
                "[dim]In a real checkout, the Razorpay Checkout.js widget would open here "
                "using this order_id, collect payment, and return razorpay_payment_id + "
                "razorpay_signature to POST /api/checkout/verify.[/dim]",
                title="Test-mode Razorpay Order",
                border_style="green",
            )
        )


def main():
    print_banner()
    state = CliState()

    with httpx.Client() as client:
        try:
            client.get(f"{API_BASE}/api/health", timeout=5)
        except httpx.HTTPError:
            console.print(f"[bold red]Cannot reach API at {API_BASE}. Is `uvicorn api:app --reload` running?[/bold red]")
            sys.exit(1)

        while True:
            try:
                user_input = console.input("\n[bold green]you>[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]bye[/dim]")
                break

            if not user_input:
                continue

            if user_input == "/quit":
                console.print("[dim]bye[/dim]")
                break

            elif user_input == "/reset":
                state.cart_id = None
                state.pending_failure = None
                console.print("[dim]Started a new session.[/dim]")
                continue

            elif user_input == "/cart":
                if not state.cart_id:
                    console.print("[dim]No cart yet — chat first.[/dim]")
                    continue
                payload = call_chat(client, state, "Can you show me what's currently in my cart?")
                render_cart(payload)
                continue

            elif user_input == "/audit":
                if not state.cart_id:
                    console.print("[dim]No cart yet.[/dim]")
                    continue
                call_audit(client, state.cart_id)
                continue

            elif user_input == "/pay":
                call_pay(client, state)
                continue

            elif user_input.startswith("/fail"):
                parts = user_input.split()
                mode = parts[1] if len(parts) > 1 else "timeout"
                if mode not in ("timeout", "signature"):
                    console.print("[red]Usage: /fail timeout|signature[/red]")
                    continue
                state.pending_failure = mode
                console.print(f"[yellow]Armed simulated '{mode}' failure for your next message.[/yellow]")
                continue

            # Regular chat turn
            try:
                payload = call_chat(client, state, user_input)
            except httpx.HTTPStatusError as e:
                console.print(f"[bold red]API error:[/bold red] {e.response.status_code} {e.response.text}")
                continue
            except httpx.HTTPError as e:
                console.print(f"[bold red]Connection error:[/bold red] {e}")
                continue

            console.print(Panel(payload["assistant_message"], title="assistant", border_style="cyan"))
            if payload["line_items"]:
                render_cart(payload)


if __name__ == "__main__":
    main()