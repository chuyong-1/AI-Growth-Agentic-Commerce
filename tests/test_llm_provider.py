# ============================================================
# FILE: tests/test_llm_provider.py
# ============================================================
"""
Tests for LLM provider selection.

The point of these is not that the model works — that needs a live key
and a network call. It is that the SYSTEM works when the model does
not: the provider is a runtime detail, a missing key degrades exactly
one endpoint, and nothing about the guardrails depends on which vendor
is answering.
"""

from __future__ import annotations

import pytest

from agents.conversational_agent import (
    DEFAULT_MODELS,
    ConversationalAgent,
    resolve_provider,
)
from schema import Catalog, CatalogItem
from decimal import Decimal


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts with no provider configured, so a real key in
    the developer's shell can't silently change what is being tested."""
    for var in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(
        items={
            "SKU_MUG_002": CatalogItem(
                sku="SKU_MUG_002", name="Ceramic Mug",
                base_price=Decimal("249.00"), max_discount_pct=Decimal("15.0"),
            ),
        }
    )


class TestProviderResolution:
    def test_groq_key_alone_selects_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        assert resolve_provider() == "groq"

    def test_anthropic_key_alone_selects_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert resolve_provider() == "anthropic"

    def test_groq_wins_when_both_keys_are_present(self, monkeypatch):
        """Documents the tie-break: setting GROQ_API_KEY is sufficient
        to switch, with no second variable to remember."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert resolve_provider() == "groq"

    def test_explicit_provider_overrides_available_keys(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert resolve_provider() == "anthropic"

    def test_provider_name_is_case_and_space_insensitive(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "  GROQ  ")
        assert resolve_provider() == "groq"

    def test_unknown_provider_names_the_valid_options(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with pytest.raises(ValueError) as exc:
            resolve_provider()
        # An error that lists the options is the difference between a
        # 10-second fix and a trip to the source.
        assert "groq" in str(exc.value) and "anthropic" in str(exc.value)

    def test_no_credentials_raises_with_actionable_guidance(self):
        with pytest.raises(RuntimeError) as exc:
            resolve_provider()
        message = str(exc.value)
        assert "GROQ_API_KEY" in message
        assert "ANTHROPIC_API_KEY" in message
        # Must also say what still works, so a missing key doesn't read
        # as "the whole app is broken".
        assert "/api/agent/propose" in message


class TestLazyModelConstruction:
    def test_agent_constructs_with_no_credentials(self, catalog):
        """The guarantee api.py depends on: importing the app must not
        require an LLM key, because only one endpoint uses a model."""
        agent = ConversationalAgent(catalog=catalog)
        assert agent._llm_with_tools is None

    def test_model_is_only_built_on_first_use(self, catalog):
        agent = ConversationalAgent(catalog=catalog)
        with pytest.raises(RuntimeError):
            _ = agent.llm_with_tools

    def test_injected_model_bypasses_env_resolution(self, catalog):
        """Lets tests and embedders supply their own client without
        setting environment variables."""
        class FakeLLM:
            def bind_tools(self, tools):
                return f"bound:{len(tools)}"

        agent = ConversationalAgent(catalog=catalog, llm=FakeLLM())
        assert agent.llm_with_tools == "bound:4"


class TestMalformedToolCallRetry:
    """Open models intermittently emit tool calls the provider cannot
    parse. That is a generation glitch, so the same payload is retried;
    real errors are not."""

    def _agent_raising(self, catalog, errors: list):
        """Builds an agent whose model raises the given errors in turn,
        then returns a plain message."""
        from langchain_core.messages import AIMessage

        calls = {"n": 0}

        class FlakyBinding:
            def invoke(self, messages):
                i = calls["n"]
                calls["n"] += 1
                if i < len(errors):
                    raise errors[i]
                return AIMessage(content="done")

        class FlakyLLM:
            def bind_tools(self, tools):
                return FlakyBinding()

        agent = ConversationalAgent(catalog=catalog, llm=FlakyLLM())
        return agent, calls

    def test_malformed_tool_call_is_retried_and_succeeds(self, catalog):
        err = Exception("Error code: 400 - tool_use_failed: Failed to parse tool call arguments as JSON")
        agent, calls = self._agent_raising(catalog, [err, err])

        text, actions, _ = agent.run_turn(history=[], user_message="hi")

        assert text == "done"
        assert calls["n"] == 3  # two failures, then success

    def test_retry_gives_up_and_reraises_after_the_limit(self, catalog):
        err = Exception("tool_use_failed")
        agent, _ = self._agent_raising(catalog, [err] * 10)

        with pytest.raises(Exception, match="tool_use_failed"):
            agent.run_turn(history=[], user_message="hi")

    def test_auth_errors_are_not_retried(self, catalog):
        """Retrying a bad key just delays the message the user needs."""
        agent, calls = self._agent_raising(
            catalog, [Exception("Error code: 401 - invalid_api_key")]
        )

        with pytest.raises(Exception, match="invalid_api_key"):
            agent.run_turn(history=[], user_message="hi")
        assert calls["n"] == 1


class TestEmptyContentFallback:
    """Reasoning models can end a turn with empty `content`, having put
    their prose in a separate reasoning field that must not be shown."""

    def test_empty_reply_is_replaced_with_a_summary_of_the_proposals(self, catalog):
        from langchain_core.messages import AIMessage
        from schema import ProposedAction

        class Binding:
            def invoke(self, messages):
                return AIMessage(content="   ")

        class LLM:
            def bind_tools(self, tools):
                return Binding()

        agent = ConversationalAgent(catalog=catalog, llm=LLM())
        text, _, _ = agent.run_turn(history=[], user_message="hi")

        assert text.strip()
        assert "rephrase" in text  # no actions were proposed

    def test_summary_describes_only_actions_that_were_emitted(self):
        from agents.conversational_agent import _summarize_proposals
        from schema import ProposedAction

        summary = _summarize_proposals([
            ProposedAction(action_type="ADD_ITEM", sku="SKU_MUG_002", rationale="r"),
            ProposedAction(action_type="APPLY_DISCOUNT", sku="SKU_MUG_002",
                           discount_pct=Decimal("10"), rationale="r"),
        ])

        assert "SKU_MUG_002" in summary
        assert "10" in summary
        # Must not claim the discount was granted — nothing has passed
        # the gatekeeper at the point this text is written.
        assert "approved" not in summary.lower()
        assert "checkout" not in summary.lower()


class TestDefaults:
    def test_every_provider_has_a_default_model(self):
        assert set(DEFAULT_MODELS) == {"anthropic", "groq"}
        assert all(v for v in DEFAULT_MODELS.values())

    def test_llm_model_env_var_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
        monkeypatch.setenv("LLM_MODEL", "some-other-model")
        import api
        assert api._llm_status() == "groq:some-other-model"

    def test_health_reports_when_no_provider_is_configured(self):
        import api
        assert "not configured" in api._llm_status()
