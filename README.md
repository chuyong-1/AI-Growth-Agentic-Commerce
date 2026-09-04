# Agentic Commerce — Razorpay Upsell, Checkout & Campaign Agent

A reference architecture for letting an LLM negotiate a sale — and run its own growth campaigns — without ever letting it touch money.

## Why this exists

Commerce is being rebuilt around autonomous agents. Protocols like UAP (Universal Agent Protocol) and ACP (Agentic Commerce Protocol) point at a near future where an AI assistant — ours, or an external AI shopping agent talking to our store over an API — browses a catalog, negotiates a bundle, applies a discount, and initiates payment on a user's behalf.

That future is only safe to ship if one principle holds absolutely: **a language model must never be the thing that authorizes a charge or a spend.** This project demonstrates that guardrail pattern end-to-end, for both a single checkout and a recurring, budget-constrained campaign process. The LLM proposes; a deterministic, non-LLM gatekeeper decides; every decision is written to a hash-chained audit trail before a rupee moves.

## The bar

| Property | What it means here |
|---|---|
| **Explainable** | Every cart mutation and campaign proposal carries a mandatory, human-readable rationale, recorded permanently. |
| **Bounded** | Every discount is checked against a per-SKU `max_discount_pct` ceiling; every campaign against a per-period spend ceiling and a concurrent-campaign cap — numbers the LLM can read but never write. |
| **Gated** | `PaymentGatekeeper` (checkout) and `CampaignBudgetStore.try_reserve` (campaigns) are the only code paths authorized to commit money or budget. |
| **Tamper-evident** | Every proposal, verdict and payment event is appended to a SHA-256 hash-chained log. Altering or deleting any historical entry breaks the chain and is detectable. |

## Architecture

Two parallel guardrail pipelines share one audit trail and one structural principle: an inert, one-way `Proposed*` object is the only channel between the untrusted LLM layer and the trusted deterministic layer.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          UNTRUSTED / LLM LAYER                            │
│                                                                           │
│   Human user (chat)              External AI buyer agent                  │
│        │                              │                                   │
│        ▼                              ▼                                   │
│  ConversationalAgent          POST /api/agent/propose                     │
│  (Groq or Anthropic           (same ProposedAction schema, same graph,    │
│   tool-calling, propose-only)  discovered via /.well-known + catalog feed)│
│        │                              │                                   │
│        └──────────────┬───────────────┘                                   │
│                       ▼  emits: list[ProposedAction]  (inert data)        │
└───────────────────────┼───────────────────────────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    DETERMINISTIC / TRUSTED LAYER (LangGraph)              │
│                                                                           │
│  UpsellAgent → PaymentGatekeeper → CreateOrderNode → Razorpay             │
│                       │                    │                              │
│                  rejected              SDK failure                        │
│                       ▼                    ▼                              │
│               RejectAndExplain      PaymentRecovery                       │
│                                                                           │
│  Every node writes to the hash-chained audit trail ──────────────────────►│
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                 CAMPAIGN / GROWTH LAYER (separate cadence)                │
│                                                                           │
│  CampaignOrchestrator (proposes) → campaign_gatekeeper_durable            │
│      → CampaignBudgetStore.try_reserve()  (atomic check-and-commit)       │
│      → APPROVED / REJECTED, written to the same audit trail               │
│                                                                           │
│  campaign_scheduler sweeps ACTIVE campaigns past expires_at and releases  │
│  their reservation, logging CAMPAIGN_EXPIRED_RELEASED                     │
└───────────────────────────────────────────────────────────────────────────┘
```

## Repository layout

| File | Purpose |
|---|---|
| `schema.py` | Pydantic models — Decimal-only money, strict validators, frozen catalog items |
| `audit_trail.py` | Thread-safe, hash-chained, append-only audit log |
| `razorpay_client.py` | Defensive wrapper around the Razorpay SDK (test mode / simulation) |
| `agent_graph.py` | LangGraph topology: UpsellAgent → PaymentGatekeeper → … |
| `session_store.py` | Cart/session store with optimistic concurrency on `save_cart()` |
| `campaign_store.py` | Campaign budget ledger — atomic `try_reserve()`, idempotent release |
| `campaign_scheduler.py` | Expiry sweep: releases budget held by campaigns past `expires_at` |
| `catalog_feed.py` | Agent-readable catalog feed + `/.well-known` discovery manifest |
| `agents/conversational_agent.py` | Tool-calling front end (propose-only tools); Groq or Anthropic, chosen at runtime |
| `agents/campaign_orchestrator.py` | Campaign proposer + deterministic campaign gatekeeper |
| `api.py` | FastAPI surface (see endpoints below) |
| `cli_chat.py` | Rich-powered terminal client |
| `static/index.html` | Browser UI: chat pane, cart/audit pane, campaign pane |
| `main.py` | Scripted demo covering six scenarios, fully offline |

## Storage model — read this before deploying

**Every store in this system is in-memory and single-process.** Carts, the audit chain, and campaign budgets live in Python objects guarded by locks. This is a deliberate tradeoff, chosen so the project runs with `pip install -r requirements.txt && python main.py` — no cloud account, no table provisioning, no environment variables, no network.

The consequences, stated plainly:

- **Nothing survives a process restart.** The audit trail is tamper-*evident*, not durable.
- **Correctness holds within one process only.** Under `uvicorn api:app` with one worker and many threads, every guarantee below is real. Run two workers and they will each hold their own budget ledger and their own audit chain, seeing none of each other's writes — the ceilings would no longer bind globally.

Each store's public interface is the seam for fixing that. Making this multi-instance means reimplementing the same three classes against a backend whose check-and-commit is atomic server-side (a DynamoDB conditional `UpdateItem`, a Postgres `UPDATE … WHERE committed + :amt <= ceiling`, or a Redis Lua script). No calling code changes, because every caller depends only on the method signatures.

## Concurrency

Three pieces of state cannot be allowed to corrupt under concurrent access. Each is protected by pushing the entire check-then-write into a single critical section, so there is no window between "read the current value" and "write the new value" for a second writer to slip into.

**Audit chain** (`audit_trail.py`) — each entry stores the hash of the entry before it. Two writers racing to append could otherwise both read the same tail and each produce an entry claiming to follow it, forking the chain into two histories that each verify on their own while disagreeing with each other. `log()` holds one lock across read-tail → compute-hash → append, so no two entries can ever be computed against the same tail.

**Campaign budget** (`campaign_store.py`) — two campaigns proposed at the same time could each look affordable against a stale read of remaining budget. That is the classic race that overspends a shared limit. `try_reserve()` performs the ceiling check and the increment inside one critical section, and raises `BudgetExceededError` with **no side effects at all** on rejection — a partially-applied reservation would leak budget permanently.

**Campaign release** (`campaign_store.py`) — the mirror-image bug. Two expiry sweeps could both see a campaign as ACTIVE and each hand its reservation back, freeing more than was ever reserved. `release_campaign()` checks status and decrements the ledger under one lock, so it is idempotent: the second caller gets `False` and changes nothing.

**Session/cart writes** (`session_store.py`) — carts use optimistic concurrency. Every write is conditioned on the version the caller last read, so a stale write (a double-submitted checkout, say) raises `SessionConflictError` rather than silently clobbering whatever a concurrent request already committed.

`tests/test_store_concurrency.py` proves all four with real, unserialized threads released from a barrier — not by inspection of the code.

## Testing strategy

Five suites, each attacking a different layer. **73 tests, no network calls, no cloud mocking.**

| Suite | What it attacks |
|---|---|
| `tests/test_adversarial.py` | Feeds malicious input directly into the checkout graph nodes — prompt-injected discounts, arithmetic tampering, hallucinated SKUs, fault injection |
| `tests/test_campaign_adversarial.py` | The same discipline on the campaign gatekeeper — over-ceiling discounts, over-cap concurrency, budget exhaustion, boundary-exact cases |
| `tests/test_store_concurrency.py` | Real threads against all three stores: chain forking, joint overspend, double release, stale cart writes |
| `tests/test_api_integration.py` | The full HTTP surface via `httpx.AsyncClient`, mocking only the two genuine third-party I/O boundaries |
| `tests/test_llm_provider.py` | Provider selection and lazy model construction — that a missing key degrades one endpoint, not the app |

The common thread: **every suite bypasses the layer that is supposed to behave well and attacks the layer that is supposed to enforce.** The adversarial suites construct hostile `ProposedAction` / `CampaignProposal` objects by hand rather than coaxing them out of a model, because the proposer is exactly the component assumed to be compromised. A test that went through the orchestrator would only prove the orchestrator proposes sensible numbers, which is not a security property.

In the integration suite only `api.AGENT.run_turn` (Anthropic) and `RazorpayGateway.create_order` (Razorpay) are mocked. The gatekeeper, the LangGraph topology, cart persistence with its version checks, and the real audit trail all execute.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the scripted demo — works offline, no configuration
python main.py

# 3. Run the test suite
pytest

# 4. Run the API server
uvicorn api:app --reload
#   → open http://127.0.0.1:8000/static/index.html
#   → or: python cli_chat.py
```

Nothing above requires credentials. These environment variables change behaviour:

| Variable | Effect if unset |
|---|---|
| `GROQ_API_KEY` or `ANTHROPIC_API_KEY` | Required only for the live conversational agent (`/api/chat`, `cli_chat.py`). Everything else — the gatekeeper, campaigns, the audit trail, `/api/agent/propose` — runs with no model at all. |
| `LLM_PROVIDER` | Auto-detected from whichever key is present (`groq` wins if both are). Set to `groq` or `anthropic` to force one. |
| `LLM_MODEL` | Defaults to `llama-3.3-70b-versatile` on Groq, `claude-sonnet-4-6` on Anthropic. |
| `RAZORPAY_TEST_KEY_ID` / `_SECRET` | Razorpay runs in **simulation mode**: orders are fabricated locally, no HTTP request is made. `GET /api/health` reports which mode is active, and simulated orders are flagged as such. |

### On swapping the model

The proposing model is a runtime choice, not an architectural one, and the code treats it that way — the agent builds its client lazily, so a missing key degrades exactly one endpoint instead of failing at import.

This is worth stating plainly because it is the thesis, not a convenience: **nothing downstream depends on which model proposes.** The gatekeeper re-derives every number from the catalog regardless of what produced the proposal. Running a smaller open model here is therefore a fair test of that claim rather than a compromise of it — a weaker proposer emits *more* invalid proposals, and the same ceiling rejects them. `GET /api/health` reports the active provider and model.

Simulation mode exists because the alternative is worse. Falling through to a live HTTP call with placeholder keys would make every "successful checkout" demo fail with an auth error dressed up as a transport failure — the demo would appear to exercise the recovery path while really only proving the keys were missing.

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | Conversational checkout turn |
| `POST /api/agent/propose` | Machine-to-machine entry point for an external AI buyer agent |
| `POST /api/checkout/verify` | Razorpay signature verification (fails closed) |
| `GET /api/audit/{cart_id}` | Audit history for one cart, with chain-integrity check |
| `POST /api/campaigns/run` | Run one campaign proposal/gatekeeper cycle |
| `GET /api/campaigns/status` | Budget ledger state for a period |
| `POST /api/campaigns/expire` | Release budget held by expired campaigns |
| `GET /api/campaigns/audit` | Campaign-scoped audit history |
| `GET /api/catalog/feed` | Agent-readable catalog, including discount ceilings |
| `GET /.well-known/agentic-commerce.json` | Discovery manifest |
| `GET /api/health` | Storage mode, Razorpay mode, cart/audit counts |

The catalog feed publishes each SKU's discount ceiling deliberately: an external agent should be able to propose a compliant discount rather than guess and be rejected. Publishing it costs nothing, because the gatekeeper enforces it regardless of what the agent believes.

## Key files to read first

1. **`agent_graph.py`** and **`agents/campaign_orchestrator.py`** — the two deterministic gatekeepers. This is the whole point of the project.
2. **`campaign_store.py`** — how "atomic under concurrency" is actually implemented, not just claimed.
3. **`schema.py`** — money is `Decimal` everywhere, never `float`; catalog items are frozen once loaded.
4. **`tests/test_store_concurrency.py`** — the sharpest evidence that the guarantees hold under concurrent attack rather than only on the happy path.
