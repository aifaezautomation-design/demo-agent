"""The LLM speed settings: one shared OpenAI client, the warm-up while the greeting plays, the stuck-request retry and the
per-reply latency log line."""

import asyncio

from pipecat.processors.aggregators.llm_context import LLMContext

from app import bot, config
from app.prompts import build_system_prompt
from app.tools import CallSession
from tests.test_agent import make_tenant


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warning(self, msg):
        self.lines.append(("warning", msg))


def _context(tenant):
    session = CallSession(tenant, "c1", "+15125550199")
    return LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(tenant, "+15125550199")}],
        tools=session.tools_schema(),
    )


def test_every_call_shares_one_openai_client():
    _, llm_a, _ = bot.build_services()
    _, llm_b, _ = bot.build_services()
    assert llm_a._client is bot._openai and llm_b._client is bot._openai


def test_stuck_request_is_retried():
    _, llm, _ = bot.build_services()
    assert llm._retry_on_timeout is (config.OPENAI_RETRY_SECS > 0)
    assert llm._retry_timeout_secs == config.OPENAI_RETRY_SECS
    assert config.OPENAI_RETRY_SECS < config.STALL_SECS  # the retry comes before the "say something" re-prompt


def test_warm_up_sends_exactly_the_prompt_and_tools_the_first_reply_will_use(monkeypatch):
    tenant = make_tenant()
    _, llm, _ = bot.build_services()
    context = _context(tenant)
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return object()

    monkeypatch.setattr(bot._openai.chat.completions, "create", create)
    log = _Log()
    asyncio.run(bot._warm_up_llm(llm, context, log))

    real = llm.build_chat_completion_params(
        llm.get_llm_adapter().get_llm_invocation_params(context, convert_developer_to_user=not llm.supports_developer_role)
    )
    assert sent["max_tokens"] == 1 and sent["model"] == config.OPENAI_MODEL
    # OpenAI caches by prefix: the warm-up must match the real request's tools and messages exactly.
    assert sent["messages"] == real["messages"] and sent["tools"] == real["tools"]
    assert any("warm-up done" in m for _, m in log.lines)


def test_warm_up_failure_is_only_logged(monkeypatch):
    _, llm, _ = bot.build_services()

    async def create(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(bot._openai.chat.completions, "create", create)
    log = _Log()
    asyncio.run(bot._warm_up_llm(llm, _context(make_tenant()), log))  # must not raise
    assert log.lines and log.lines[0][0] == "warning"


def test_latency_line_names_where_the_time_went():
    from pipecat.observers.user_bot_latency_observer import (
        LatencyBreakdown,
        MeasuredFrom,
    )

    breakdown = LatencyBreakdown.model_validate({
        "total_secs": 1.31,
        "measured_from": MeasuredFrom.USER_SILENCE,
        "contributions": [
            {"key": "a", "label": "endpointing wait", "owner": "config: VAD stop_secs", "owner_kind": "setting", "start_time": 0, "duration_secs": 0.2},
            {"key": "b", "label": "LLM inference", "owner": "OpenAILLMService#0", "owner_kind": "service", "start_time": 0.2, "duration_secs": 0.65},
            {"key": "c", "label": "frame hop", "owner": "pipeline", "owner_kind": "pipeline", "start_time": 0.85, "duration_secs": 0.001},
        ],
    })
    log = _Log()
    bot._log_latency(log, breakdown)
    line = log.lines[0][1]
    assert line.startswith("reply latency 1.31s") and "LLM inference 0.65s" in line and "frame hop" not in line
