Agentic Commerce — Razorpay Upsell, Checkout & Campaign Agent

A reference architecture for letting an LLM negotiate a sale — and run its own growth campaigns — without ever letting it touch money.

Why this exists

Commerce is being rebuilt around autonomous agents. Protocols like UAP (Universal Agent Protocol) and ACP (Agentic Commerce Protocol) point at a near future where an AI assistant — ours, or an external AI shopping agent talking to our store over an API — browses a catalog, negotiates a bundle, applies a discount, and initiates payment on a user's behalf.

That future is only safe to ship if one principle holds absolutely: a language model must never be the thing that authorizes a charge or a spend. This project demonstrates that guardrail pattern end-to-end, for both a single checkout and a recurring, budget-constrained campaign process: the LLM proposes, a deterministic, non-LLM gatekeeper decides, and every decision is written to an immutable, hash-chained audit trail — durably, in DynamoDB, safe under concurrent writers — before a rupee moves.

"The Bar"
Property	What it means here
Explainable	Every cart mutation and every campaign proposal carries a mandatory, human-readable rationale, recorded permanently.
Bounded	Every discount is checked against a per-SKU max_discount_pct ceiling; every campaign is checked against a per-period spend ceiling and concurrent-campaign cap — numbers the LLM can read but never write.
Gated	PaymentGatekeeper (checkout) and the campaign store's atomic try_reserve (campaigns) are the only code paths authorized to commit money or budget.
Immutable, durable audit trail	Every proposal, verdict, and payment/campaign event is appended to a SHA-256 hash-chained log backed by DynamoDB, so it survives process restarts and stays tamper-evident under concurrent writers.
Architecture

Two parallel guardrail pipelines share the same audit trail and the same structural principle: an inert, one-way Proposed* object is the only channel between the untrusted LLM layer and the trusted deterministic layer.

┌───────────────────────────────────────────────────────────────────────────┐
│                          UNTRUSTED / LLM LAYER                            │
│                                                                             │
│   Human user (chat)              External AI buyer agent                  │
│        │                              │                                   │
│        ▼                              ▼                                   │
│  ConversationalAgent          /api/agent/propose                          │
│  (agents/conversational_agent.py)   (same ProposedAction schema,          │
│   Anthropic tool-calling,           same graph, discovered via            │
│   propose-only tools                /.well-known/agentic-commerce.json    │
│                                      + catalog_feed.py)                   │
│        │                              │                                   │
│        └──────────────┬───────────────┘                                  │
│                        ▼  emits: list[ProposedAction] (inert data)        │
└────────────────────────┼──────────────────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    DETERMINISTIC / TRUSTED LAYER (LangGraph)              │
│                                                                             │
│  UpsellAgent → PaymentGatekeeper → CreateOrderNode → Razorpay (test mode) │
│                       │                    │                              │
│                  rejected              SDK failure                        │
│                       ▼                    ▼                              │
│               RejectAndExplain      PaymentRecovery                       │
│                                                                             │
│  Every node writes to the durable audit trail (audit_trail_dynamo.py) ───►│
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                 CAMPAIGN / GROWTH LAYER (separate cadence)                │
│                                                                             │
│  campaign_orchestrator (LLM proposes campaigns) → gatekeeper node          │
│      → CampaignBudgetStore.try_reserve() (atomic DynamoDB UpdateItem,     │
│        ADD committed_spend, ADD active_count, single conditional write)   │
│      → APPROVED / REJECTED, written to the same durable audit trail       │
│                                                                             │
│  campaign_scheduler.py (hourly EventBridge) sweeps ACTIVE campaigns past  │
│  expires_at and calls store.release(), logging CAMPAIGN_EXPIRED_RELEASED  │
└───────────────────────────────────────────────────────────────────────────┘

Durability was added to every stateful component: sessions/carts, the audit chain, and campaign budgets all moved from in-process Python objects to DynamoDB-backed stores, each with an explicit concurrency-safety mechanism (see below).

Repository layout
schema.py                     Pydantic models — Decimal-only money, strict validators
audit_trail.py                In-memory hash-chained audit log (used by main.py demo, stdlib only)
audit_trail_dynamo.py         Durable, hash-chained audit log — DynamoDB-backed, atomic chain writes
razorpay_client.py            Defensive wrapper around the Razorpay SDK (test mode)
agent_graph.py                LangGraph topology: UpsellAgent → PaymentGatekeeper → …
session_store.py              Durable cart/session store — optimistic concurrency on save_cart()
campaign_store.py             Durable campaign budget store — atomic reservation via DynamoDB ADD
campaign_scheduler.py         Hourly EventBridge job: expires stale ACTIVE campaigns, releases budget
catalog_feed.py                Agent-readable catalog feed + /.well-known discovery manifest
agents/conversational_agent.py Anthropic tool-calling front end (propose-only tools)
agents/campaign_orchestrator.py LLM-driven campaign proposer + deterministic campaign gatekeeper
api.py                        FastAPI: /api/chat, /api/checkout/verify, /api/audit, /api/agent/propose,
                               /api/campaigns/status, /api/campaigns/run
cli_chat.py                   Rich-powered terminal client for manual exploration
static/index.html             Browser UI: chat pane, cart/audit pane, campaign status pane
main.py                       Scripted demo: checkout success/reject/failure, external agent propose,
                               a mixed-outcome campaign cycle, and a budget-depletion cycle
tests/test_adversarial.py     Attacks the checkout graph nodes directly
tests/test_campaign_adversarial.py  Attacks the campaign gatekeeper directly
tests/test_api_integration.py Attacks the full HTTP surface, LLM + Razorpay mocked
tests/test_durable_stores.py  moto-mocked DynamoDB: chain-fork, budget-overspend, and
                               stale-write-conflict tests against the durable stores
Concurrency & Durability

Two places in this system hold state that absolutely cannot be corrupted by two things happening "at the same time" — the audit chain and the campaign budget. Both are solved the same way: push the entire check-then-write into a single atomic DynamoDB operation, so there's no window between "read the current value" and "write the new value" for a second writer to sneak into.

Audit chain (audit_trail_dynamo.py): each new entry needs to know the hash of the entry before it. Two writers racing to append could otherwise both read the same "current tail" and each produce an entry that claims to follow it — forking the chain into two histories that each look valid on their own. This is prevented with a single counter item whose UpdateItem call only succeeds if the sequence number and tail hash still match what this writer read; if another writer got there first, the condition fails, and the writer retries against the new, correct tail. The actual audit entry is only written after that counter update succeeds, so ordering is decided once, atomically, before anything else happens.
Campaign budget (campaign_store.py): two campaigns proposed "at the same time" could each look affordable if they only ever see a stale read of remaining budget — that's the classic race that lets you overspend a shared limit. Instead of read-check-write from the client, try_reserve() does the increment and the ceiling check inside one conditional UpdateItem: DynamoDB only applies the ADD committed_spend if committed_spend + amount <= max_spend is still true at the moment of the write, atomically. If two reservations race, DynamoDB serializes them; whichever lands second sees the already-incremented value and its condition fails, so it comes back as a clean BudgetExceededError instead of a silent overspend.
Session/cart writes (session_store.py): carts use optimistic concurrency — every write is conditioned on the version the caller last read. A stale write (e.g. a double-submitted checkout) fails with SessionConflictError rather than silently clobbering whatever a concurrent request already committed.

tests/test_durable_stores.py proves all three of these under real (moto-mocked) concurrent access, not just by inspection of the code.

Testing strategy

Three complementary suites:

tests/test_adversarial.py — feeds malicious/malformed input directly into the checkout graph nodes (prompt-injected discounts, arithmetic tampering, hallucinated SKUs, fault injection), bypassing the conversational layer to prove the gatekeeper — not model good behavior — is what blocks the attack.
tests/test_campaign_adversarial.py — the same discipline applied to the campaign gatekeeper: over-ceiling campaigns, over-cap concurrent campaigns, and boundary-exact cases.
tests/test_api_integration.py — drives the real FastAPI app over httpx.AsyncClient, mocking only the genuine I/O boundaries (ChatAnthropic.invoke, Razorpay's order.create), so tool-call parsing, the live gatekeeper, and the real audit trail all execute for real.
tests/test_durable_stores.py — moto-mocked DynamoDB tests proving the audit chain cannot fork, the campaign budget cannot be jointly overspent by racing reservations, and stale session writes are rejected rather than silently applied.
Quick start
bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
export ANTHROPIC_API_KEY=sk-ant-...
export RAZORPAY_TEST_KEY_ID=rzp_test_...
export RAZORPAY_TEST_KEY_SECRET=...
export AUDIT_TABLE=agentictrade-dev-audit
export CAMPAIGN_TABLE=agentictrade-dev-campaigns
export SESSION_TABLE=agentictrade-dev-sessions
export AWS_REGION=us-east-1

# 3. Run the scripted demo (checkout, external agent, campaign cycles)
python main.py

# 4. Run the API server
uvicorn api:app --reload
#   → open static/index.html, or:
python cli_chat.py

# 5. Run the full test suite
pytest tests/ -v

No live network calls are required to run the test suite — checkout/campaign adversarial tests never touch the LLM or Razorpay SDK, the integration suite mocks both at their I/O boundary, and test_durable_stores.py mocks DynamoDB with moto.

Key files to read first
agent_graph.py and agents/campaign_orchestrator.py — the two deterministic gatekeepers. This is the whole point of the project.
campaign_store.py and audit_trail_dynamo.py — how "atomic under concurrency" is actually implemented, not just claimed.
schema.py — money is Decimal everywhere, never float; catalog items are frozen once loaded.
tests/test_durable_stores.py — the sharpest evidence that the durability layer holds under concurrent attack, not just sequential happy-path calls.