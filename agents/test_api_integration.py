 # ============================================================
# ADD TO: tests/test_api_integration.py
# ============================================================

# ==================================================================
# /api/agent/propose — direct machine-to-machine entrypoint
# ------------------------------------------------------------------
# Proves an EXTERNAL AI buyer agent (bypassing our own chat/LLM layer
# entirely) still can't get anything past the same PaymentGatekeeper
# that guards the conversational path. No LLM mocking needed here —
# this endpoint takes ProposedAction objects directly.
# ==================================================================
class TestExternalAgentProposeEndpoint:
    @pytest.mark.asyncio
    async def test_valid_external_proposal_reaches_razorpay(
        self, client, mocked_razorpay_order_create, fresh_audit
    ):
        resp = await client.post(
            "/api/agent/propose",
            json={
                "cart_id": None,
                "agent_identity": "test-shopping-agent-v1",
                "actions": [
                    {
                        "action_type": "ADD_ITEM",
                        "sku": "SKU_COFFEE_001",
                        "rationale": "External agent purchasing on behalf of its principal.",
                    },
                    {
                        "action_type": "FINALIZE",
                        "rationale": "External agent confirms purchase intent.",
                    },
                ],
            },
        )

        assert resp.status_code == 200
        payload = resp.json()

        assert payload["cart_status"] == "ORDER_CREATED"
        assert payload["order_id"] == "order_MOCKtest123456"
        assert payload["computed_total"] == "399.00"

        mocked_razorpay_order_create.order.create.assert_called_once()
        call_kwargs = mocked_razorpay_order_create.order.create.call_args[0][0]
        assert call_kwargs["amount"] == 39900

        # Audit trail records the external agent's identity, not just
        # a generic event — explainability extends to WHO proposed it.
        audit_resp = await client.get(f"/api/audit/{payload['cart_id']}")
        audit_payload = audit_resp.json()
        assert audit_payload["chain_intact"] is True
        proposal_entries = [
            e for e in audit_payload["entries"]
            if e["event_type"] == "EXTERNAL_AGENT_PROPOSAL_RECEIVED"
        ]
        assert len(proposal_entries) == 1
        assert proposal_entries[0]["payload"]["agent_identity"] == "test-shopping-agent-v1"

    @pytest.mark.asyncio
    async def test_external_agent_excessive_discount_is_blocked(
        self, client, mocked_razorpay_order_create, fresh_audit
    ):
        """An external agent is just as untrusted as our own LLM layer —
        this proves the gatekeeper doesn't implicitly trust the
        machine-to-machine entrypoint any more than the chat one."""
        resp = await client.post(
            "/api/agent/propose",
            json={
                "cart_id": None,
                "agent_identity": "malicious-or-buggy-agent",
                "actions": [
                    {
                        "action_type": "ADD_ITEM",
                        "sku": "SKU_GRINDER_003",
                        "rationale": "External agent request.",
                    },
                    {
                        "action_type": "APPLY_DISCOUNT",
                        "sku": "SKU_GRINDER_003",
                        "discount_pct": 60.0,  # catalog ceiling is 5.0%
                        "rationale": "External agent demands maximum discount.",
                    },
                    {
                        "action_type": "FINALIZE",
                        "rationale": "External agent finalizes despite policy.",
                    },
                ],
            },
        )

        assert resp.status_code == 200
        payload = resp.json()

        assert payload["cart_status"] == "AUDIT_FAILED"
        assert payload["order_id"] is None
        assert any("exceeds catalog ceiling" in note for note in payload["system_notes"])
        mocked_razorpay_order_create.order.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_actions_list_is_rejected_with_400(self, client, fresh_audit):
        resp = await client.post(
            "/api/agent/propose",
            json={"cart_id": None, "agent_identity": "empty-agent", "actions": []},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_hallucinated_sku_from_external_agent_is_rejected_not_crashed(
        self, client, mocked_razorpay_order_create, fresh_audit
    ):
        resp = await client.post(
            "/api/agent/propose",
            json={
                "cart_id": None,
                "agent_identity": "confused-agent",
                "actions": [
                    {
                        "action_type": "ADD_ITEM",
                        "sku": "SKU_DOES_NOT_EXIST",
                        "rationale": "Agent hallucinated a SKU.",
                    },
                ],
            },
        )

        # Must not 500 — the API layer should surface a clean error path.
        assert resp.status_code in (200, 400, 404, 422)
        mocked_razorpay_order_create.order.create.assert_not_called()


# ==================================================================
# Discovery surface — /.well-known manifest and /api/catalog/feed
# ==================================================================
class TestAgentDiscoverySurface:
    @pytest.mark.asyncio
    async def test_well_known_manifest_lists_all_core_endpoints(self, client):
        resp = await client.get("/.well-known/agentic-commerce.json")
        assert resp.status_code == 200
        manifest = resp.json()

        assert manifest["protocol"] == "agentic-commerce-manifest"
        endpoints = manifest["endpoints"]
        assert "catalog_feed" in endpoints
        assert "propose_action" in endpoints
        assert "conversational_checkout" in endpoints
        assert "audit_trail" in endpoints
        assert "payment_verification" in endpoints
        assert len(manifest["guardrails_summary"]) >= 4

    @pytest.mark.asyncio
    async def test_catalog_feed_declares_the_same_ceilings_gatekeeper_enforces(self, client):
        resp = await client.get("/api/catalog/feed")
        assert resp.status_code == 200
        feed = resp.json()

        grinder = next(i for i in feed["catalog"] if i["sku"] == "SKU_GRINDER_003")
        # This must match the catalog fixture's max_discount_pct exactly —
        # a feed that lies about ceilings is worse than no feed at all.
        assert grinder["negotiation"]["max_discount_pct"] == "5.0"
        assert grinder["transact"]["propose_endpoint"] == "/api/agent/propose"


# ==================================================================
# Campaign endpoints, exercised over the real HTTP surface
# ==================================================================
class TestCampaignEndpointsThroughAPI:
    @pytest.mark.asyncio
    async def test_run_campaigns_endpoint_returns_decided_campaigns(self, client, fresh_audit):
        import api as api_module
        from decimal import Decimal
        from agents.campaign_orchestrator import CampaignBudget

        # Isolate this test's budget so it doesn't inherit committed
        # spend from any other test hitting the same module global.
        api_module.CAMPAIGN_BUDGET = CampaignBudget(period_label="api-test-period")

        resp = await client.post("/api/campaigns/run")
        assert resp.status_code == 200
        payload = resp.json()

        assert payload["period"] == "api-test-period"
        assert isinstance(payload["campaigns"], list)
        assert len(payload["campaigns"]) > 0

    @pytest.mark.asyncio
    async def test_campaign_audit_endpoint_reflects_run(self, client, fresh_audit):
        import api as api_module
        from agents.campaign_orchestrator import CampaignBudget

        api_module.CAMPAIGN_BUDGET = CampaignBudget(period_label="api-audit-test-period")

        await client.post("/api/campaigns/run")
        resp = await client.get("/api/campaigns/audit")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["chain_intact"] is True
        assert payload["entry_count"] > 0

    @pytest.mark.asyncio
    async def test_campaign_audit_404s_when_no_campaigns_have_run(self, client, fresh_audit):
        resp = await client.get("/api/campaigns/audit")
        assert resp.status_code == 404