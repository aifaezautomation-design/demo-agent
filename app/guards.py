"""Checks on what the model SAID against what it DID, so a caller is never left waiting on words with no action behind them.

Each failure below was seen in a real call or in scripts/conversation_eval.py:
  promise     "Let me check for availability this Friday." and no check_availability call (20 s of silence)
  unsaved     "You're all set!" and no book_appointment call (nothing booked, caller told otherwise)
  unsearched  "3 PM isn't available" from a sample of three times, with no search for 3 PM
A guard only reads text and flags; the pipeline decides what to do (it re-prompts the model once per caller turn).
"""
import re

from app.tools import is_readback

# "Let me check...", "one moment": a promise to look something up or book it, which only a tool call can keep.
_PROMISE = re.compile(
    r"\b(let me (check|see|look|pull|book|find|search|grab)|i'?ll (check|look|book|pull|find|search)|"
    r"i('?m| am) (checking|looking|booking|pulling|searching)|one moment|just a (moment|second)|hold on|bear with me)\b",
    re.I,
)
# "You're all set", "I've booked ...", "your appointment is confirmed": a statement that something was saved.
_SAVED_CLAIM = re.compile(
    r"(you'?re (all set|booked|confirmed)|you are (all set|booked|confirmed)|(i'?ve|i have) (booked|scheduled|rescheduled|cancel+ed|moved)|"
    r"(appointment|visit|booking) (is|has been) (confirmed|booked|scheduled|set|cancel+ed|rescheduled)|"
    r"(is|are) (now )?(confirmed|booked)\b|we look forward to seeing you)",
    re.I,
)
# A read-back says the same words before asking: "Just to confirm, I have you booked for Friday at nine."
_READBACK_WORDING = re.compile(r"((i have|i've got) you (booked|scheduled|down|set))", re.I)

# A claim about the calendar made without searching: the model only saw a sample of the day.
_UNAVAILABLE = re.compile(
    r"(not available|isn'?t available|aren'?t available|unavailable|no (openings|availability|slots)|fully booked|"
    r"nothing (is )?(open|available)|only (have|found|see) (morning|afternoon|a few|these))",
    re.I,
)

NOTES = {
    "goodbye": (
        "SYSTEM NOTE: the caller has now been rude or abusive twice. Say one polite goodbye and call end_call now. "
        "Do not argue and do not offer anything else."
    ),
    "promise": (
        "SYSTEM NOTE: you told the caller you would check or book, but you called no tool, so they are waiting in silence. "
        "Call the right tool now (check_availability for a day or time; book_appointment only when name, phone, reason and "
        "slot are all confirmed). Say nothing before the call."
    ),
    "unsaved": (
        "SYSTEM NOTE: nothing has been booked, rescheduled or cancelled yet, because no tool was called. If the caller just "
        "said yes to your read-back, call book_appointment (or reschedule_appointment / cancel_appointment) now with nothing "
        "spoken first; otherwise ignore this note and answer the caller normally. Never say it is done before the tool returns."
    ),
    "unsearched": (
        "SYSTEM NOTE: you told the caller something is unavailable without searching for it. Call check_availability now "
        "for the time or part of day they asked for (preferred_time or time_of_day), then answer from the result."
    ),
    "stall": "SYSTEM NOTE: the caller is waiting for your reply. Answer their last message now, in one short sentence.",
}


def broke_promise(spoken: str, called_tool: bool, saved: bool = False) -> str | None:
    """'promise' when the model said it would check or book, or 'unsaved' when it said a booking was made, without calling a
    tool; None otherwise. A question to the caller is never a broken promise, since then the caller is the one expected to act."""
    spoken = spoken.strip()
    if not spoken or called_tool or spoken.endswith("?"):
        return None
    if not saved:
        asked_to_confirm = re.search(r"\bconfirm\b", spoken, re.I)
        if _SAVED_CLAIM.search(spoken) or (_READBACK_WORDING.search(spoken) and not asked_to_confirm):
            return "unsaved"
    if _PROMISE.search(spoken):
        return "promise"
    return None


def claimed_unavailable_unsearched(spoken: str, searched_this_turn: bool) -> bool:
    return not searched_this_turn and bool(_UNAVAILABLE.search(spoken))


class PromiseGuard:
    """Judges each model reply in order. After a read-back ("Just to confirm ... Is that correct?") the next reply must
    be a tool call or a question: a bare "Great, see you then" means the model skipped the booking."""

    def __init__(self):
        self.after_readback = False

    def check(self, spoken: str, called_tool: bool, writes: int, searched: bool = True, abuse: int = 0) -> str | None:
        spoken = spoken.strip()
        pending, self.after_readback = self.after_readback, False
        if abuse >= 2 and not called_tool and spoken:
            return "goodbye"  # the call must end; the model is asked once per caller turn, like every other guard
        if called_tool or not spoken:
            return None
        if is_readback(spoken):
            self.after_readback = True
            return None
        kind = broke_promise(spoken, called_tool, saved=writes > 0)
        if kind:
            return kind
        if claimed_unavailable_unsearched(spoken, searched):
            return "unsearched"
        if pending and writes == 0 and not spoken.endswith("?"):
            return "unsaved"
        return None
