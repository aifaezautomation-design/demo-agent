"""The demo line's filter for stock closing offers (app/trimmer.py)."""
import asyncio

import pytest
from pipecat.frames.frames import LLMTextFrame

from app.trimmer import OfferPolicy, OfferTrimmer, could_become_offer


@pytest.mark.parametrize("closing", [
    "How can I assist you today?", "How can I help you today?", "What else can I help you with?",
    "Is there anything else I can help you with?", "Is there anything else you'd like to know?",
    "Is there something else you'd like to ask?", "Do you have any other questions?", "Anything I can help you book?",
    "If you have more questions, just let me know!", "Let me know if you need anything else.", "Feel free to ask me anything.",
])
def test_generic_closing_offers_are_dropped(closing):
    p = OfferPolicy()
    assert p.trim(f"The team gives an exact quote on a call. {closing}") == "The team gives an exact quote on a call. "


@pytest.mark.parametrize("needed", [
    "Which day works for you?", "Would you like to book that time?", "Is that correct?", "Is the number you're calling from the best one to reach you?",
    "Can I have your full name, please?", "Want me to have the team call you with a quote?", "Would you like me to take your details?",
    "Are you a new or returning patient?", "Is that for you right now, or are you asking how I'd handle it?",
])
def test_questions_the_conversation_needs_are_never_dropped(needed):
    assert OfferPolicy().trim(needed) == needed
    assert OfferPolicy().trim(f"Sure. {needed}") == f"Sure. {needed}"


def test_a_reply_is_never_left_empty():
    assert OfferPolicy().trim("How can I assist you today?") == "How can I assist you today?"
    assert OfferPolicy().trim("Is there anything else? How can I help?") == "How can I help?"


def test_test_booking_offers_at_most_twice_and_never_twice_in_a_row():
    p = OfferPolicy()
    offer = "Would you like to try a test booking?"
    answers = []
    for n in range(6):
        answers.append(p.trim(f"We answer around the clock. {offer}"))
    kept = [a.endswith(offer) for a in answers]
    assert kept == [True, False, True, False, False, False], kept  # twice in total, never back to back
    assert all(a.startswith("We answer around the clock.") for a in answers)


def test_streamed_sentences_are_held_only_while_they_look_like_an_offer():
    assert could_become_offer("") and could_become_offer("Would") and could_become_offer("Would you li")
    assert could_become_offer("How can I assist you today")
    assert not could_become_offer("The team") and not could_become_offer("I'm Mia") and not could_become_offer("Yes, we")


def test_streaming_filter_drops_the_offer_and_passes_everything_else_at_once():
    out = []

    async def go():
        t = OfferTrimmer(enabled=True)

        async def push(frame, direction=None):
            out.append(frame)

        t.push_frame = push
        for token in ["Yes", ",", " the", " practice", " keeps", " its", " number", ".", " How", " can", " I", " assist", " you", " today", "?"]:
            await t._text(token)
            if token == " number":
                # the first sentence streamed straight through, token by token, before it ended
                assert "".join(f.text for f in out) == "Yes, the practice keeps its number"
        await t._flush_sentence()
        return "".join(f.text for f in out if isinstance(f, LLMTextFrame))

    assert asyncio.run(go()) == "Yes, the practice keeps its number."


def test_streaming_filter_keeps_a_lone_offer_so_the_caller_is_not_met_with_silence():
    out = []

    async def go():
        t = OfferTrimmer(enabled=True)

        async def push(frame, direction=None):
            out.append(frame)

        t.push_frame = push
        await t._text("How can I help you today?")
        await t._flush_sentence()
        # what the end-of-reply step does when everything was dropped and no tool was called
        if t._dropped and not t._pushed and not t._tool:
            await t._release(t._dropped[-1])

    asyncio.run(go())
    assert "".join(f.text for f in out) == "How can I help you today?"


def test_statement_style_test_offers_count_and_are_dropped_over_the_limit():
    offer = "If you're interested, I can help set up a test booking."
    p = OfferPolicy()
    # a factual answer that mentions a test booking stays, but uses up one of the two offers
    assert p.trim("This is a demo practice, so feel free to book a test appointment.").endswith("test appointment.")
    # the next reply follows an offer, so its offer is dropped (never twice in a row)
    assert p.trim(f"We answer all day. {offer}") == "We answer all day. "
    # a reply later on may offer again: that is the second and last one
    assert p.trim(f"Yes, a practice can choose my name. {offer}").endswith(offer)
    assert p.trim("Fine, no problem.") == "Fine, no problem."
    assert p.trim(f"Sure. {offer}") == "Sure. "  # the limit of two is used up for the rest of the call
