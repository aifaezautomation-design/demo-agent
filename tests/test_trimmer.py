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


def test_a_next_step_offer_is_not_repeated_in_the_very_next_reply():
    p = OfferPolicy()
    first = p.trim("The team can go through that on a call. Would you like to set up a time for that?")
    assert first.endswith("set up a time for that?")
    second = p.trim("Every call is saved with a summary. Would you like me to have the team call you?")
    assert second == "Every call is saved with a summary. "  # the previous reply already ended on an offer
    third = p.trim("The plan is based on call minutes. Want me to have the team call you with a quote?")
    assert third.endswith("with a quote?")  # a reply later on may offer again


def test_a_long_answer_loses_its_closing_offer():
    p = OfferPolicy()
    long_answer = ("The plan is based on how many call minutes the practice needs, plus a one-time setup fee. "
                   "The Boltix AI team can give you an exact quote on a short call. ")
    offer = "Would you like me to have them reach out to you?"
    assert len((long_answer + offer).split()) > 38
    assert p.trim(long_answer + offer) == long_answer
    assert OfferPolicy().trim("It depends on call volume. " + offer).endswith(offer)  # a short answer keeps it


@pytest.mark.parametrize("offer", [
    "Would you like to know more about how I can assist?", "Would you like to book a call with them?",
    "Would you like me to arrange that?", "Would you like to set up a time for that?", "Would you like me to have them reach out to you?",
    "Would you like to hear more?", "Can I set that up for you?", "Would you prefer to hear about pricing?",
])
def test_closing_offers_are_kept_at_most_twice_per_call_and_never_twice_in_a_row(offer):
    from app.trimmer import is_offer

    assert is_offer(offer), offer
    p = OfferPolicy()
    replies = [p.trim(f"The team can go through that on a call. {offer}") for _ in range(8)]
    kept = [r.endswith(offer) for r in replies]
    assert sum(kept) == 2, kept  # at most twice in the call
    assert not any(a and b for a, b in zip(kept, kept[1:])), kept  # never back to back
    assert all(r.startswith("The team can go through that on a call.") for r in replies)  # the answer itself is never touched


@pytest.mark.parametrize("needed", [
    "Can I take your name and practice?", "Can I get your full name?", "Is the number you're calling from the best one to reach you?",
    "Would you like to come in at noon or one PM?", "Would you like to book that time?", "Would you like Tuesday or Wednesday?",
    "Which day works for you?", "Is that correct?", "Can I have your name, please?",
])
def test_questions_the_flows_need_are_not_offers_and_are_never_counted(needed):
    from app.trimmer import is_offer

    assert not is_offer(needed), needed
    p = OfferPolicy()
    assert all(p.trim(f"Sure. {needed}").endswith(needed) for _ in range(6))
    assert p.offers == 0


def test_a_flow_question_does_not_use_up_the_offers():
    p = OfferPolicy()
    p.trim("Sure. Can I take your name and practice?")
    p.trim("Thanks. Is the number you're calling from the best one to reach you?")
    assert p.trim("Okay. Would you like me to have them reach out?").endswith("reach out?")  # still the first offer


def test_the_get_started_answer_is_never_cut_and_never_counted():
    from app.demo_facts import DEMO_FACTS
    from app.trimmer import is_next_step_answer, is_offer

    answer = f"You can {DEMO_FACTS['next_step']}."
    assert "Faez's email" in answer and "boltix A I dot com" in answer and "take your details" in answer
    assert is_next_step_answer(answer) and not is_offer(answer)
    p = OfferPolicy()
    for _ in range(6):  # more often than the two-offer cap, and in consecutive replies
        assert p.trim(answer) == answer
    assert p.offers == 0
    # the cap still applies to ordinary offers afterwards
    offer = "Would you like to hear more?"
    assert p.trim("Fine. " + offer).endswith(offer) and p.trim("Fine. " + offer) == "Fine. "
    # a follow-up question completing the same answer is kept too
    q = OfferPolicy()
    out = q.trim("You can reply to Faez's email or book a call at boltix A I dot com. Would you like me to take your details?")
    assert out.endswith("take your details?") and q.offers == 0
    # the same question without the answer in front is an ordinary offer
    assert is_offer("Would you like me to take your details?")
