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
