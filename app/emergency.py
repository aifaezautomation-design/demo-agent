"""Deterministic emergency triage.

Why this exists: left to the LLM alone, an emergency was handled correctly only about three times in four. In
testing it sometimes said "hold on while I check availability" and never called the tool, and for a knocked-out
tooth it gave first aid and stopped. For the one situation where a mistake matters most, the first steps are
therefore done in code, before the LLM runs:

  * airway trouble (can't breathe or swallow): the 911 instruction is spoken, a callback is flagged, and the LLM
    is told not to offer an appointment;
  * any other emergency (severe pain, swelling, bleeding, a knocked-out or broken tooth, an abscess): a caring line
    (plus first aid when it applies) is spoken at once while the calendar is searched for the soonest same-day
    slot, and the LLM gets the result so it can offer it without another round trip.

Wording that the detector misses still reaches the LLM and its prompt rules; this layer only adds certainty.
"""

import json
import re

from loguru import logger
from pipecat.frames.frames import EagerTranscriptionFrame, Frame, LLMMessagesAppendFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Phrases like "no pain", "not an emergency", "doesn't have any swelling" must not count.
_NEGATED = re.compile(
    r"\b(no|not|without|zero|nothing|never|don'?t have|do not have|doesn'?t have|isn'?t|aren'?t|wasn'?t|haven'?t)\b"
    r"\s+(\w+\s+){0,3}(pain|swelling|swollen|bleeding|emergency|urgent|abscess|infection)\b",
    re.I,
)
# A hypothetical or a question about the service is not a report ("what if my tooth gets knocked out?").
_HYPOTHETICAL = re.compile(r"\b(if|what happens|what would|in case|hypothetical|someday|in general|do you (treat|handle|do|see))\b", re.I)
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|my|me|we|our|he|she|his|her|him|my (son|daughter|kid|child|wife|husband|mom|dad|mother|father))\b", re.I)

_AIRWAY = re.compile(
    r"(trouble|difficulty|hard|harder|can'?t|cannot|unable|struggl\w*|short(ness)? of|closing up|closing)\s+(\w+\s+){0,2}"
    r"(breath\w*|swallow\w*)|throat\s+(is\s+)?(closing|swelling|swollen)|(breath\w*|swallow\w*)\s+(is\s+)?(hard|difficult|getting hard)",
    re.I,
)
_KNOCKED_OUT = re.compile(
    r"knocked\s+(out|loose)|tooth\s+(fell|came|popped|got\s+knocked)\s+out|lost\s+(a|my|the)\s+(front\s+|back\s+)?tooth|"
    r"(tooth|teeth)\s+(is|are|was|were)\s+(completely\s+)?(out|loose)",
    re.I,
)
_EMERGENCY = [
    _KNOCKED_OUT,
    re.compile(r"\b(severe|extreme|excruciating|unbearable|terrible|horrible|intense|awful|worst|agonizing)\b[\w\s,]{0,25}\b(tooth\s*ache|toothache|pain|hurts?)\b", re.I),
    re.compile(r"\b(tooth\s*ache|toothache|pain|hurts?|hurting)\b[\w\s,]{0,30}\b(8|9|10|eight|nine|ten)\b(\s+(out\s+of|/)\s+(10|ten))?", re.I),
    re.compile(r"\b(8|9|10|eight|nine|ten)\s*(out\s+of|/)\s*(10|ten)\b", re.I),
    re.compile(r"\b(swollen|swelling|swells|puffed\s+up)\b", re.I),
    re.compile(r"\babscess\b|\bpus\b|\b(infected|infection)\b[\w\s]{0,20}\b(tooth|gum|jaw|mouth)\b", re.I),
    re.compile(r"bleeding\s+(won'?t|will\s+not|isn'?t|is\s+not|does\s*n'?t|doesn'?t)\s+stop|bleeding\s+(a\s+lot|heavily|badly|really\s+bad)|(mouth|gum|gums)\s+(is|are)\s+bleeding", re.I),
    re.compile(r"\b(broke|broken|cracked|snapped|split)\b[\w\s]{0,15}\b(tooth|teeth)\b|\b(tooth|teeth)\b[\w\s]{0,15}\b(broke|broken|cracked|snapped|split)\b", re.I),
    re.compile(r"\b(really|very|extremely|so|pretty)\s+(bad|painful|sore)\b[\w\s]{0,20}\b(tooth|toothache|pain|jaw|gum)\b|\b(tooth|toothache|jaw|gum)\b[\w\s]{0,25}\b(really|very|extremely|so)\s+(bad|painful)\b", re.I),
    re.compile(r"killing\s+me|can'?t\s+(sleep|eat|chew|take\s+it|stand\s+it)\b[\w\s]{0,25}\b(pain|tooth|hurt)|throbbing", re.I),
    re.compile(r"(it'?s|this\s+is|i\s+have|i'?m\s+having|i\s+need|we\s+have)\s+(an?\s+)?(dental\s+)?emergency", re.I),
]


def classify(text: str) -> str | None:
    """'airway', 'emergency', or None for what a caller just said."""
    t = " ".join(text.split())
    cleaned = _NEGATED.sub(" ", t)
    if _HYPOTHETICAL.search(cleaned):
        return None
    if _AIRWAY.search(cleaned) and (_FIRST_PERSON.search(cleaned) or re.search(r"swell|swollen|face|throat|tongue", cleaned, re.I)):
        return "airway"
    body = r"\b(severe|extreme|excruciating|unbearable|emergency|tooth|teeth|toothache|pain|jaw|gum|gums|mouth|face|cheek)\b"
    if not _FIRST_PERSON.search(cleaned) and not re.search(body, cleaned, re.I):
        return None
    if any(p.search(cleaned) for p in _EMERGENCY):
        return "emergency"
    return None


def knocked_out(text: str) -> bool:
    return bool(_KNOCKED_OUT.search(text))


CARING = "I'm so sorry you're hurting. Let me get you in right away."
SAFETY = "If it's ever hard to breathe or swallow, call 911."
FIRST_AID = "If the tooth is out, hold it by the crown, don't scrub it, and keep it in milk or inside your cheek."
AIRWAY_LINE = (
    "That sounds serious. Please hang up and call 911 right now: trouble breathing or swallowing needs emergency "
    "care immediately. Our team will follow up with you afterward."
)

_NEVER_ELSEWHERE = (
    "Never send the caller to another dentist, clinic, urgent care, or emergency room, and do not mention 911 yourself: "
    "the system already told them what to do if breathing gets hard. Do not call any tool in this reply: any callback "
    "is already flagged, and booking waits until they give their name."
)
_NOTE_SLOT = (
    "EMERGENCY, handled by the system. The caller reported {kind}. The system already spoke a caring line{aid}, searched "
    "the practice's calendar, and told the caller: '{said}'\n"
    "Calendar result: {result}\n"
    "Do NOT search again, do not repeat the time, do not say 'one moment', and do not apologise or sympathise again (the system already did). Your reply is one short sentence asking "
    "for their name so you can book that opening (the caller ID is their number). Call book_appointment with is_emergency "
    "true and that slot only after they tell you their name. " + _NEVER_ELSEWHERE
)
_NOTE_CALLBACK_SLOT = (
    "EMERGENCY, handled by the system. The caller reported {kind}. The system already spoke a caring line{aid}, searched "
    "the calendar, flagged an urgent callback, and told the caller: '{said}'\n"
    "Calendar result: {result}\n"
    "Do NOT search again, do not repeat what was said, and do not apologise or sympathise again (the system already did). Your reply is one short sentence asking for their name so you "
    "can book that earliest opening (the caller ID is their number). Call book_appointment with is_emergency true and that "
    "slot only after they tell you their name. " + _NEVER_ELSEWHERE
)
_NOTE_NOTHING = (
    "EMERGENCY, handled by the system. The caller reported {kind}. The calendar has NO openings at all. The system "
    "already told the caller that the urgent request is flagged and a team member will call them back very shortly. "
    "There is no appointment to offer: do not mention any time or slot, do not invent one, and do not search. Reply "
    "with one short sentence asking whether there is anything else you can help with. " + _NEVER_ELSEWHERE
)
_AIRWAY_NOTE = (
    "EMERGENCY, handled by the system. The caller described trouble breathing or swallowing. The system already told "
    "them to call 911 and flagged an urgent callback for the team. Reply with at most one short caring sentence that "
    "repeats it (call 911 now; the team will follow up). Do not offer an appointment, do not ask questions, and do not call any tool (the callback is already flagged)."
)


def _earliest(result: dict) -> str:
    """'today at 11 AM' for a same-day slot, otherwise the full day and time ('Thursday, January 10 at 8 AM')."""
    spoken = result["options"][0]["spoken"]
    if result.get("window") in ("within_two_hours", "later_today") and " at " in spoken:
        return "today at " + spoken.split(" at ", 1)[1]
    return spoken


def plan(result: dict, knocked_out_tooth: bool, practice: str) -> tuple[str | None, str]:
    """What to say after the calendar search: (a line for the code to speak, the note for the LLM).

    The code states the facts (the time, the callback) so none of them depends on the LLM remembering or inventing
    them; the LLM's part is to ask for the name and book.
    """
    kind = "a knocked-out tooth" if knocked_out_tooth else "severe pain, swelling, bleeding, or a damaged tooth"
    aid = " and the first-aid instructions" if knocked_out_tooth else ""
    callback = f"I've marked this as urgent, and a team member from {practice} will call you back very shortly"
    if not result.get("options"):
        # Nothing to present, so the code says it all rather than leave room to invent a time.
        return f"{callback} to get you in.", _NOTE_NOTHING.format(kind=kind)
    earliest = _earliest(result)
    if result.get("urgent_callback") == "flagged":
        line = f"{callback}. In the meantime, the earliest opening I have is {earliest}."
        return line, _NOTE_CALLBACK_SLOT.format(kind=kind, aid=aid, result=json.dumps(result), said=line)
    line = f"The earliest opening I have is {earliest}."
    return line, _NOTE_SLOT.format(kind=kind, aid=aid, result=json.dumps(result), said=line)


def _speak(text: str) -> TTSSpeakFrame:
    """A line spoken by code. Not interruptible: a later fragment of the same sentence from the speech recognizer
    must not cut off first aid or the 911 instruction (it did, in a live test)."""
    frame = TTSSpeakFrame(text, append_to_context=False)
    frame.interruptible = False
    return frame


class EmergencyTriage(FrameProcessor):
    """Sits between speech-to-text and the user aggregator and runs the emergency steps above.

    Order matters. The caller's transcript must reach the aggregator while nothing is playing: a transcript that
    arrives while the bot is speaking counts as the caller talking over it and interrupts the bot, which cut the
    first-aid and safety line off after two seconds when it was spoken first. So the note for the LLM goes in first,
    then the transcript, and only then the spoken lines.
    """

    def __init__(self, session, language_state):
        super().__init__()
        self._session = session
        self._language = language_state
        self._handled_emergency = False
        self._handled_airway = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        before: list[Frame] = []
        after: list[Frame] = []
        if direction == FrameDirection.DOWNSTREAM:
            before, after = await self.triage(frame, lambda f: self.push_frame(f, direction))
        for f in before:
            await self.push_frame(f, direction)
        await self.push_frame(frame, direction)
        for f in after:
            await self.push_frame(f, direction)

    async def triage(self, frame: Frame, emit=None) -> tuple[list[Frame], list[Frame]]:
        """(frames to send before this frame, frames to send after it) for what the caller just said.

        Flux speech recognition announces an EAGER end of turn: a provisional transcript that starts the LLM before the
        turn is final, and whose response is kept if the final transcript agrees. Two consequences:
          * triage must act on the eager transcript, ahead of it: acting only on the final one meant the model had
            already answered without the emergency note (it offered a next-day slot and promised a callback that was
            never flagged);
          * the spoken lines must go out BEFORE the eager transcript, through `emit`: once speculation has started, the
            speculation gate holds every later frame and releases it behind the model's reply, so the caller heard the
            model's "may I have your name?" before the first-aid advice. (Speaking early is safe: these lines cannot be
            interrupted.) With a final-transcript-only speech engine the lines follow the transcript instead.
        """
        if not isinstance(frame, (TranscriptionFrame, EagerTranscriptionFrame)) or self._language.current != "en":
            return [], []
        speak_now = emit if isinstance(frame, EagerTranscriptionFrame) else None
        kind = classify(frame.text)
        if kind == "airway" and not self._handled_airway:
            self._handled_airway = True
            return await self._airway(frame, speak_now)
        if kind == "emergency" and not self._handled_emergency:
            self._handled_emergency = True
            return await self._emergency(frame, speak_now)
        return [], []

    async def _airway(self, frame: TranscriptionFrame, speak_now=None) -> tuple[list[Frame], list[Frame]]:
        """(frames to send before the transcript, frames to send after it). With `speak_now`, the 911 line is said at once."""
        logger.info(f"[{self._session.call_id}] triage: AIRWAY ({frame.text[:60]!r})")
        line = _speak(AIRWAY_LINE)
        if speak_now:
            await speak_now(line)  # before anything else, even the callback write
        try:
            await self._session.flag_urgent_callback("Caller reported trouble breathing or swallowing (told to call 911)")
        except Exception:
            logger.exception("could not flag the airway callback")
        note = LLMMessagesAppendFrame([{"role": "developer", "content": _AIRWAY_NOTE}], run_llm=False)
        return [note], ([] if speak_now else [line])

    async def _emergency(self, frame: TranscriptionFrame, speak_now=None) -> tuple[list[Frame], list[Frame]]:
        logger.info(f"[{self._session.call_id}] triage: EMERGENCY ({frame.text[:60]!r})")
        aid = knocked_out(frame.text)
        spoken: list[Frame] = []

        async def say(f: Frame) -> None:
            if speak_now:
                await speak_now(f)  # straight away: the caring line does not wait for the calendar search below
            else:
                spoken.append(f)

        await say(_speak(self._line(aid)))
        try:
            result = await self._session.emergency_search()
        except Exception:
            # No note: the LLM falls back to its own prompt rule and searches itself.
            logger.exception("emergency search failed; the LLM will search instead")
            return [], spoken
        extra_line, note = plan(result, aid, self._session.tenant.name)
        if extra_line:
            await say(_speak(extra_line))
        return [LLMMessagesAppendFrame([{"role": "developer", "content": note}], run_llm=False)], spoken

    @staticmethod
    def _line(first_aid: bool) -> str:
        return " ".join([CARING, FIRST_AID, SAFETY] if first_aid else [CARING, SAFETY])
