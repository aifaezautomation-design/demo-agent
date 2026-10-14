import asyncio
import json

import pytest

from app import emergency
from app.emergency import classify, knocked_out

EMERGENCIES = [
    "I have really severe tooth pain, like a nine out of ten.",
    "My cheek is swollen and my tooth hurts a lot.",
    "I just knocked out my front tooth!",
    "I think I have an abscess, it's really painful and there's a bump on my gum.",
    "My tooth broke in half and my mouth is bleeding.",
    "This is an emergency, my tooth is killing me.",
    "I've had a toothache since last night and the pain is a 9.",
    "My son fell off his bike and his tooth got knocked out.",
    "I can't sleep because of the pain in my tooth, it's throbbing.",
    "I have a really bad toothache and I need to be seen today.",
    "My face is swollen on one side.",
    "The bleeding won't stop after I had a tooth pulled.",
    "Hi, I cracked my tooth and it hurts.",
    "I've got an infection in my gum and my jaw hurts.",
    "It's an emergency, I need a dentist right now.",
    "The pain is an eight out of ten and it won't go away.",
    "I have unbearable pain in my back tooth.",
    "my tooth fell out this morning",
    "I have excruciating pain in my jaw",
    "I'm having a dental emergency",
]

AIRWAY = [
    "My face is swelling up fast and I'm having trouble breathing.",
    "I can't swallow and my throat feels like it's closing.",
    "It's hard to breathe and my tongue is swollen.",
    "I'm struggling to breathe, my face is puffed up.",
    "I have trouble swallowing and my neck is swollen",
]

NOT_EMERGENCIES = [
    "I'd like to book a routine cleaning for next Thursday.",
    "Do you handle emergencies?",
    "I don't have any pain, I just want a checkup.",
    "No swelling, no pain, I just need a cleaning.",
    "What happens if my tooth gets knocked out?",
    "My name is John Smith and my number is five one two five five five zero one nine nine.",
    "How much is a root canal?",
    "Is it an emergency if my tooth is chipped?",
    "Thursday at ten works for me.",
    "I had a toothache last year but it's fine now.",
    "Do you treat swelling?",
    "Can I move my appointment to Friday morning?",
    "I'm a new patient and I'd like a cleaning and whitening.",
    "What are your hours on Saturday?",
    "Do you accept Delta Dental?",
    "I'm not in any pain, I want to schedule a filling.",
    "Yes, that works for me.",
    "I want to cancel my appointment.",
]


@pytest.mark.parametrize("text", EMERGENCIES)
def test_emergencies_are_detected(text):
    assert classify(text) == "emergency", text


@pytest.mark.parametrize("text", AIRWAY)
def test_airway_trouble_is_detected_and_takes_priority(text):
    assert classify(text) == "airway", text


@pytest.mark.parametrize("text", NOT_EMERGENCIES)
def test_routine_requests_and_questions_are_left_alone(text):
    assert classify(text) is None, text


def test_knocked_out_tooth_gets_first_aid():
    assert knocked_out("I just knocked out my front tooth")
    assert not knocked_out("I have severe tooth pain")
    line = emergency.EmergencyTriage._line(True)
    assert "hold it by the crown" in line and line.index("milk") < line.index("911")  # first aid before the safety line
    assert "hold it by the crown" not in emergency.EmergencyTriage._line(False)


def test_spoken_lines_never_send_the_caller_elsewhere():
    for line in (emergency.EmergencyTriage._line(True), emergency.EmergencyTriage._line(False)):
        for banned in ("urgent care", "emergency room", "another dentist", "hospital"):
            assert banned not in line.lower()
        assert "911" in line  # only as the conditional safety line


class _Tenant:
    name = "Apex Dental Care"


class _Session:
    call_id = "t"
    tenant = _Tenant()

    def __init__(self, result):
        self.result, self.flagged = result, []

    async def emergency_search(self):
        return self.result

    async def flag_urgent_callback(self, reason):
        self.flagged.append(reason)
        return {"status": "flagged"}


class _Lang:
    current = "en"


def _run_triage(text, result):
    """Returns (triage, frames_before_the_transcript, frames_after_it) for what a caller just said."""
    from pipecat.frames.frames import TranscriptionFrame

    triage = emergency.EmergencyTriage(_Session(result), _Lang())
    frame = TranscriptionFrame(text, "u", "2030-01-01T00:00:00Z")

    async def go():
        kind = classify(frame.text)
        if kind == "airway":
            return await triage._airway(frame)
        if kind == "emergency":
            return await triage._emergency(frame)
        return [], []

    before, after = asyncio.run(go())
    return triage, before, after


def test_emergency_hands_over_the_note_first_and_speaks_only_after_the_transcript():
    from pipecat.frames.frames import LLMMessagesAppendFrame, TTSSpeakFrame

    result = {"options": [{"slot_id": "x", "spoken": "Wednesday, January 9 at 11 AM"}], "window": "within_two_hours"}
    triage, before, after = _run_triage("I knocked out my tooth and it hurts", result)
    # Speaking before the transcript is released made the transcript interrupt the bot's own line (seen live).
    assert [type(f) for f in before] == [LLMMessagesAppendFrame] and before[0].run_llm is False
    assert [type(f) for f in after] == [TTSSpeakFrame, TTSSpeakFrame]
    caring, offer = after
    assert "milk" in caring.text and "sorry you're hurting" in caring.text and caring.append_to_context is False
    assert caring.interruptible is False and offer.interruptible is False  # a later transcript fragment must not cut them off
    assert caring.text.index("milk") < caring.text.index("911")
    assert offer.text == "The earliest opening I have is today at 11 AM."
    note = before[0].messages[0]["content"]
    assert json.dumps(result) in note and "Do NOT search again" in note


def test_emergency_with_an_empty_calendar_is_told_by_code_with_no_time_to_invent():
    from pipecat.frames.frames import LLMMessagesAppendFrame, TTSSpeakFrame

    triage, before, after = _run_triage("I have severe tooth pain", {"options": [], "window": "none"})
    assert [type(f) for f in before] == [LLMMessagesAppendFrame] and [type(f) for f in after] == [TTSSpeakFrame, TTSSpeakFrame]
    assert "will call you back very shortly" in after[1].text and not any(ch.isdigit() for ch in after[1].text)
    assert "do not invent one" in before[0].messages[0]["content"]


def test_if_the_search_fails_the_caller_still_hears_the_caring_line_and_the_llm_searches():
    from pipecat.frames.frames import TTSSpeakFrame

    class _Broken(_Session):
        async def emergency_search(self):
            raise RuntimeError("calendar down")

    from pipecat.frames.frames import TranscriptionFrame

    triage = emergency.EmergencyTriage(_Broken({}), _Lang())
    before, after = asyncio.run(triage._emergency(TranscriptionFrame("I have severe pain", "u", "2030-01-01T00:00:00Z")))
    assert before == [] and [type(f) for f in after] == [TTSSpeakFrame]


def test_airway_flags_a_callback_and_speaks_911_after_the_transcript():
    from pipecat.frames.frames import LLMMessagesAppendFrame, TTSSpeakFrame

    triage, before, after = _run_triage("My face is swelling and I can't breathe", {})
    assert [type(f) for f in before] == [LLMMessagesAppendFrame] and [type(f) for f in after] == [TTSSpeakFrame]
    assert "call 911 right now" in after[0].text
    assert triage._session.flagged and "breathing" in triage._session.flagged[0]
    assert "Do not offer an appointment" in before[0].messages[0]["content"]


def test_plan_states_the_facts_in_code_and_never_leaves_room_to_invent_a_slot():
    same_day = {"options": [{"slot_id": "x", "spoken": "Wednesday, January 9 at 11 AM"}], "window": "within_two_hours"}
    line, note = emergency.plan(same_day, False, "Apex Dental Care")
    assert line == "The earliest opening I have is today at 11 AM."          # spoken by code, not invented by the model
    assert "asking for their name" in note and "Do NOT search again" in note and "do not repeat the time" in note

    tomorrow = {"options": [{"slot_id": "y", "spoken": "Thursday, January 10 at 8 AM"}], "window": "next_available_not_today",
                "urgent_callback": "flagged"}
    line, note = emergency.plan(tomorrow, False, "Apex Dental Care")
    assert "a team member from Apex Dental Care will call you back very shortly" in line
    assert line.endswith("the earliest opening I have is Thursday, January 10 at 8 AM.")
    assert "asking for their name" in note

    line, note = emergency.plan({"options": [], "window": "none", "urgent_callback": "flagged"}, False, "Apex Dental Care")
    assert line == "I've marked this as urgent, and a team member from Apex Dental Care will call you back very shortly to get you in."
    assert "do not mention any time or slot" in note and "do not invent one" in note

    for result in (same_day, tomorrow, {"options": []}):
        for aid in (True, False):
            _, n = emergency.plan(result, aid, "Apex Dental Care")
            assert "Never send the caller to another dentist" in n
