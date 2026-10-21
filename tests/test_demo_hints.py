"""Demo-line notes for price and call-transfer questions (app/demo_hints.py)."""

import asyncio

import pytest
from pipecat.frames.frames import LLMMessagesAppendFrame, TranscriptionFrame

from app import bot
from app.demo_hints import PRICE_NOTE, TRANSFER_NOTE, DemoHints, hint_for
from app.language import LanguageState

PRODUCT_PRICE = [
    "How much does this cost?",
    "What's your price?",
    "How much do you charge per month?",
    "Is this expensive?",
    "What does the service cost?",
    "How much is Mia?",
    "What's the pricing for the AI receptionist?",
    "Can a small office afford you?",
]
PATIENT_PRICE = [
    "How much is a cleaning?",
    "How much do you charge for a filling?",
    "What does a root canal cost?",
    "How much is teeth whitening?",
    "What's the price of a crown?",
    "How much does an exam cost with insurance?",
]
TRANSFER = [
    "Can you transfer me to a person?",
    "Can you transfer calls to our front desk?",
    "Could you put me through to someone?",
    "Can you connect me to a real person?",
]
NEITHER = [
    "Are you a real person?",
    "How long does setup take?",
    "How many minutes are included in the plan?",
    "I'd like to book a cleaning for Tuesday.",
    "Do you text patients?",
]


@pytest.mark.parametrize("text", PRODUCT_PRICE)
def test_product_price_questions_get_the_price_note(text):
    assert hint_for(text) == "price"


@pytest.mark.parametrize("text", PATIENT_PRICE)
def test_dental_price_questions_are_left_alone(text):
    assert hint_for(text) is None


@pytest.mark.parametrize("text", TRANSFER)
def test_transfer_questions_get_the_transfer_note(text):
    assert hint_for(text) == "transfer"


@pytest.mark.parametrize("text", NEITHER)
def test_other_questions_get_no_note(text):
    assert hint_for(text) is None


def test_notes_carry_the_facts_and_no_numbers():
    assert "monthly plan" in PRICE_NOTE and "$" not in PRICE_NOTE
    assert "transfer calls to the front desk" in TRANSFER_NOTE and "Never say you can't transfer" in TRANSFER_NOTE


def test_note_goes_in_before_the_transcript_and_only_once_per_call():
    from pipecat.tests.utils import run_test

    said = ["How much does this cost?", "Can you transfer me to a person?", "Just tell me how much this costs."]
    frames = [TranscriptionFrame(text=t, user_id="u", timestamp="now") for t in said]
    down, _ = asyncio.run(run_test(
        DemoHints(LanguageState()),
        frames_to_send=frames,
        expected_down_frames=[LLMMessagesAppendFrame, TranscriptionFrame, LLMMessagesAppendFrame, TranscriptionFrame,
                              TranscriptionFrame],
    ))
    price, transfer = down[0], down[2]
    assert price.run_llm is False and price.messages[0] == {"role": "developer", "content": PRICE_NOTE}
    assert transfer.messages[0]["content"] == TRANSFER_NOTE
    assert [f.text for f in down if isinstance(f, TranscriptionFrame)] == said


def test_no_notes_off_english():
    state = LanguageState()
    state.current = "es"
    assert DemoHints(state).note_for("How much does this cost?") is None


def _processors(monkeypatch, demo_mode):
    from unittest.mock import MagicMock

    from pipecat.runner.types import TelnyxCallData

    from tests.test_agent import make_tenant

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
    monkeypatch.setattr(bot.notify, "notify_demo_call", lambda *a, **k: False)
    call_data = TelnyxCallData.model_validate(
        {"stream_id": "s1", "call_id": "cc1", "outbound_encoding": "PCMU", "from": "+15125550199", "to": "+15125550100"}
    )
    asyncio.run(bot.run_bot(MagicMock(), call_data, make_tenant(demo_mode=demo_mode), "CA1"))
    return seen["processors"]


def test_hints_run_on_the_demo_line_only(monkeypatch):
    demo = _processors(monkeypatch, demo_mode=True)
    assert demo.index("EmergencyTriage") < demo.index("DemoHints") < demo.index("LLMUserAggregator")
    assert "DemoHints" not in _processors(monkeypatch, demo_mode=False)


@pytest.mark.parametrize("text", ["How do I get started?", "How do I sign up?", "I'm interested.", "We're really interested", "What's the next step?"])
def test_get_started_questions_get_the_full_next_step_note(text):
    from app.demo_facts import DEMO_FACTS
    from app.demo_hints import GET_STARTED_NOTE

    assert hint_for(text) == "get_started"
    assert DEMO_FACTS["next_step"] in GET_STARTED_NOTE and "ANSWER, not an offer" in GET_STARTED_NOTE


@pytest.mark.parametrize("text,other", [
    ("I'm interested, can someone call me?", None), ("We're not interested", "opt_out"),
    ("How much does this cost? I'm interested", "price"), ("I'm interested, can you transfer me to a person?", "transfer"),
])
def test_get_started_does_not_hijack_other_questions(text, other):
    assert hint_for(text) == other
