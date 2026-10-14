"""Builds the real per-call pipeline (transport, VAD, Smart Turn, aggregators,
event handlers) with only the network run stubbed, so Pipecat API drift is
caught at test time rather than on a live call."""

import asyncio
from unittest.mock import MagicMock

from pipecat.runner.types import TelnyxCallData

from app import bot
from tests.test_agent import make_tenant


def test_run_bot_assembles_and_bills(monkeypatch):
    finalized = {}

    created = {}

    async def create_call_log(*args, call_id=None):
        created["call_id"] = call_id
        return call_id

    async def finalize_call(call_id, duration, transcript, summary, booked):
        finalized.update(call_id=call_id, duration=duration, booked=booked)
        return 0.0

    async def fake_run(self, worker, **kwargs):
        # Event handlers must have registered against real event names.
        assert worker is not None

    monkeypatch.setattr(bot.db, "create_call_log", create_call_log)
    monkeypatch.setattr(bot.db, "finalize_call", finalize_call)
    monkeypatch.setattr(bot.WorkerRunner, "run", fake_run)

    call_data = TelnyxCallData.model_validate(
        {"stream_id": "s1", "call_id": "cc1", "outbound_encoding": "PCMU",
         "from": "+15125550199", "to": "+15125550100"}
    )
    asyncio.run(bot.run_bot(MagicMock(), call_data, make_tenant(), "CA1"))
    # The id is chosen up front so the insert can run in the background.
    assert finalized["call_id"] == created["call_id"] and finalized["booked"] is False


def _graph(monkeypatch, multilingual):
    monkeypatch.setattr(bot.config, "MULTILINGUAL_ENABLED", multilingual)
    seen = {}

    async def create_call_log(*args, call_id=None):
        return call_id

    async def finalize_call(*args):
        return 0.0

    async def fake_run(self, worker, **kwargs):
        return None

    class SpyPipeline(bot.Pipeline):
        def __init__(self, processors, *args, **kwargs):
            seen["processors"] = [type(p).__name__ for p in processors]
            super().__init__(processors, *args, **kwargs)

    monkeypatch.setattr(bot, "Pipeline", SpyPipeline)
    monkeypatch.setattr(bot.db, "create_call_log", create_call_log)
    monkeypatch.setattr(bot.db, "finalize_call", finalize_call)
    monkeypatch.setattr(bot.WorkerRunner, "run", fake_run)
    call_data = TelnyxCallData.model_validate(
        {"stream_id": "s1", "call_id": "cc1", "outbound_encoding": "PCMU", "from": "+15125550199", "to": "+15125550100"}
    )
    asyncio.run(bot.run_bot(MagicMock(), call_data, make_tenant(), "CA1"))
    return seen


def test_english_graph_has_no_language_processors(monkeypatch):
    seen = _graph(monkeypatch, multilingual=False)
    assert seen["processors"], "the graph was not captured"
    assert "LanguageDetector" not in seen["processors"] and "ServiceSwitcher" not in seen["processors"]
    assert "DeepgramSTTService" in seen["processors"]


def test_multilingual_graph_adds_detector_and_switcher(monkeypatch):
    seen = _graph(monkeypatch, multilingual=True)
    assert "LanguageDetector" in seen["processors"] and "ServiceSwitcher" in seen["processors"]
    assert "DeepgramSTTService" not in seen["processors"]  # it sits inside the switcher instead
