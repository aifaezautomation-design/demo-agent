"""Mia never stays silent: a reply that ends with no words and no tool call (OpenAI out of credits on the 2026-10-23 call,
an outage, an empty reply) gets the fixed "trouble" line, once per caller turn."""

import asyncio

from pipecat.frames.frames import (
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)

from app import bot


def _replies(tracker: bot._SpokeTracker, *replies: list) -> int:
    """Runs each reply (a list of frames between start and end) through the tracker; returns how often on_empty fired."""
    fired = []

    async def on_empty():
        fired.append(1)

    async def go():
        tracker.on_empty = on_empty
        for frames in replies:
            for frame in [LLMFullResponseStartFrame(), *frames, LLMFullResponseEndFrame()]:
                tracker._track(frame)
            await asyncio.sleep(0)

    asyncio.run(go())
    return len(fired)


def test_a_reply_with_nothing_in_it_speaks_the_fallback():
    assert _replies(bot._SpokeTracker(), []) == 1


def test_a_spoken_reply_does_not():
    assert _replies(bot._SpokeTracker(), [LLMTextFrame("Yes, Faez from Boltix AI set this demo up.")]) == 0


def test_a_tool_call_does_not():
    assert _replies(bot._SpokeTracker(), [FunctionCallsStartedFrame(function_calls=[])]) == 0


def test_whitespace_only_counts_as_nothing():
    assert _replies(bot._SpokeTracker(), [LLMTextFrame("  ")]) == 1


def test_the_fallback_is_spoken_once_per_caller_turn():
    tracker = bot._SpokeTracker()
    assert _replies(tracker, [], [], []) == 1  # a re-prompt that also fails stays quiet
    tracker.reset()  # the caller speaks again
    assert _replies(tracker, []) == 1


def test_the_start_of_a_reply_is_not_activity():
    # Pipecat sends the start frame before the OpenAI request; counting it hid the failed request from the stall guard.
    tracker = bot._SpokeTracker()
    before = tracker.last_activity
    tracker._track(LLMFullResponseStartFrame())
    assert tracker.last_activity == before
