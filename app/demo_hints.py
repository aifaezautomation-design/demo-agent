"""Demo line: a short note for the model when the caller asks about Mia's price or about call transfer.

These are the first things a practice manager asks, and gpt-4o-mini answered both against its prompt in rehearsal
(tests/demo_rehearsal.md): "How much does this cost?" got the practice's cleaning price, and "Can you transfer me to a
person?" got "I can't transfer calls directly". As with emergencies (app/emergency.py), code recognizes the question
and puts the right fact in front of the model just before the caller's words. It adds no delay: a regex on the
transcript, no extra model call.
"""

import re

from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    Frame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.demo_facts import DEMO_FACTS
from app.tools import is_opt_out

_PRICE = re.compile(r"\b(how much|cost|costs|costing|price|prices|pricing|priced|charge|charges|fee|fees|expensive|cheap|afford)\b", re.I)
# Words that point at Mia or the service rather than at a dental treatment.
_PRODUCT = re.compile(
    r"\b(this|you|your|yours|mia|boltix|service|system|product|software|receptionist|ai|a\.i\.|plan|plans|package|"
    r"subscription|monthly|per month|a month|setup|set up)\b",
    re.I,
)
# A dental treatment or visit named in the same sentence: then it is a patient's price question, answered from the
# practice's prices as usual.
_DENTAL = re.compile(
    r"\b(clean\w*|fill\w*|crown\w*|root canal\w*|whiten\w*|exam\w*|x-?rays?|implant\w*|extract\w*|braces|invisalign|"
    r"check-?ups?|consult\w*|veneer\w*|denture\w*|bridge\w*|sealant\w*|tooth|teeth|dental work|procedure\w*|"
    r"treatment\w*|insurance|copay\w*|deductible\w*|visit|appointment\w*)\b",
    re.I,
)
_TRANSFER = re.compile(
    r"\btransfer\w*\b|\bput (me|them|callers?|patients?|calls?) through\b|"
    r"\bconnect (me|them|callers?|patients?|calls?) (to|with) (a|an|the|our|your)?\s*(real )?"
    r"(person|human|someone|receptionist|front desk|staff|manager|team|office)\b",
    re.I,
)

_FAEZ = re.compile(r"\bfaez\b|\b(email|e-mail|message) from (boltix|boltics|bolt ?x)\b|\bgot (an|your) (email|e-mail)\b", re.I)
_OUTREACH = re.compile(
    r"\b(did|has|have) (someone|somebody|anyone|you|boltix|faez)\b[^.?!]*\b(call|called|calling|contact|contacted|reach|reached)\b|"
    r"\bhow did you (get|find) (my|our|the) (e-?mail|number|name|info|information|address)\b|\bwhy (are you|did you) (calling|call|email)\w*\b",
    re.I,
)
_INFO_EMAIL = re.compile(
    r"\b(send|email|e-mail|mail)\b[^.?!]*\b(more )?(info|information|details|brochure|pricing|something)\b[^.?!]*\b(by )?(e-?mail)?\b|"
    r"\bmore (info|information|details)\b[^.?!]*\be-?mail\b",
    re.I,
)
_CAPABILITIES = re.compile(r"\bwhat (can|do) you (do|offer|handle)\b|\bwhat (are you|is this) (able|capable)\b|\bwhat does (this|it|mia) do\b", re.I)

OPT_OUT_NOTE = (
    "SYSTEM NOTE: the caller is asking not to be contacted again. Say exactly one polite line, for example "
    f'"{DEMO_FACTS["opt_out"]}" Nothing else: no question, no offer, no next step. Then stop.'
)
FAEZ_NOTE = (
    "SYSTEM NOTE: the caller mentions an email, or Faez. Faez is " + DEMO_FACTS["founder"] + '. If they say they got an email '
    f'from Faez or Boltix AI, answer: "{DEMO_FACTS["faez_email"]}" If they ask who Faez is, say he is the founder of Boltix AI. '
    "Say nothing else about Faez; never invent his background, title history, location, or plans."
)
OUTREACH_NOTE = (
    f'SYSTEM NOTE: the caller asks if someone called their office or how you got their email. Answer: "{DEMO_FACTS["outreach"]}" '
    "Do not argue, guess, or explain how contact details were found."
)
INFO_EMAIL_NOTE = (
    f'SYSTEM NOTE: the caller wants more information by email. Say exactly: "{DEMO_FACTS["info_by_email"]}" Nothing more. '
    "You have no email address for them, so never promise to send, or have the team send, an email or any information. "
    'Only if they choose the call, use the request_boltix_callback flow (name, practice, number, read-back, yes) and put '
    '"wants info by email" in notes.'
)
CAPABILITIES_NOTE = (
    "SYSTEM NOTE: the caller asks what you can do. Name at most TWO highlights from the What you do fact (for example answering "
    "every call day or night, and booking appointments) in one or two short sentences, then stop. Never read the whole list."
)

_GET_STARTED = re.compile(
    r"\b(how (do|can|would|should) (i|we) (get started|start|sign up|begin|join|get going|move forward|proceed)|"
    r"(get|getting) started|sign(ing)? up|(i'?m|i am|we'?re|we are)( very| really| quite)? interested|"
    r"what('?s| is) the next step|next steps?|how (does|do) (it|this|we) (start|begin))\b",
    re.I,
)
_WANTS_A_CALL = re.compile(r"\b(call (me|us)|give (me|us) a (call|ring)|call ?back|call back|someone call|have (someone|the team) call)\b", re.I)

GET_STARTED_NOTE = (
    "SYSTEM NOTE: the caller asks how to get started or sign up, or says they are interested. This is an ANSWER, not an offer: "
    f"give the whole next step in one or two short sentences, keeping all three ways: \"You can {DEMO_FACTS['next_step']}.\" "
    "Do not shorten it to only one way and do not turn it into a question. If they already asked for a call, skip the list and "
    "start the callback flow instead (ask their name and practice)."
)

PRICE_NOTE = (
    "SYSTEM NOTE: if the caller's next message asks what you or the service cost (not the price of a dental "
    f"treatment), answer only from the Price of Mia fact: {DEMO_FACTS['pricing']}. Say no number and no dental prices. "
    'Use this short wording: "It\'s a monthly plan based on call minutes, plus a one-time setup fee, and the Boltix AI team '
    'gives an exact quote." Then offer: "Want me to have the team call you with a quote?"'
)
TRANSFER_NOTE = (
    "SYSTEM NOTE: the caller's next message is about transferring a call. Answer from the Call transfer fact: "
    f"{DEMO_FACTS['call_transfer']}. Never say you can't transfer calls. Use this wording: \"For a practice, "
    "I can be set up to transfer calls to the front desk. In this demo there's no one to transfer to, but I can have the "
    'team call you back."'
)


def hint_for(text: str) -> str | None:
    """'price', 'transfer', or None for what the caller just said."""
    t = " ".join((text or "").split())
    if is_opt_out(t):
        return "opt_out"
    if _OUTREACH.search(t):
        return "outreach"
    if _FAEZ.search(t):
        return "faez"
    if _INFO_EMAIL.search(t) and not _PRICE.search(t):
        return "info_email"
    if _CAPABILITIES.search(t):
        return "capabilities"
    if _TRANSFER.search(t):
        return "transfer"
    if _PRICE.search(t) and _PRODUCT.search(t) and not _DENTAL.search(t):
        return "price"
    if _GET_STARTED.search(t) and not _WANTS_A_CALL.search(t):
        return "get_started"
    return None


NOTES = {
    "price": PRICE_NOTE, "transfer": TRANSFER_NOTE, "opt_out": OPT_OUT_NOTE, "faez": FAEZ_NOTE, "outreach": OUTREACH_NOTE,
    "info_email": INFO_EMAIL_NOTE, "capabilities": CAPABILITIES_NOTE, "get_started": GET_STARTED_NOTE,
}


class DemoHints(FrameProcessor):
    """Sits between speech-to-text and the user aggregator on the demo line. Each note is given at most once per call."""

    def __init__(self, language_state=None):
        super().__init__()
        self._language = language_state
        self._given: set[str] = set()

    def note_for(self, text: str) -> LLMMessagesAppendFrame | None:
        """The note to add ahead of this transcript, or None. Also used by scripts/demo_rehearsal.py."""
        if self._language is not None and self._language.current != "en":
            return None
        kind = hint_for(text)
        if not kind or kind in self._given:
            return None
        self._given.add(kind)
        return LLMMessagesAppendFrame([{"role": "developer", "content": NOTES[kind]}], run_llm=False)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (TranscriptionFrame, EagerTranscriptionFrame)):
            note = self.note_for(frame.text)
            if note:
                await self.push_frame(note, direction)  # into the context before the caller's message
        await self.push_frame(frame, direction)
