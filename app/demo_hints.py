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

PRICE_NOTE = (
    "SYSTEM NOTE: if the caller's next message asks what you or the service cost (not the price of a dental "
    f"treatment), answer only with the Price of Mia fact: {DEMO_FACTS['pricing']}. Say no number and no dental prices. "
    'Then offer: "Want me to have the team call you with a quote?"'
)
TRANSFER_NOTE = (
    "SYSTEM NOTE: the caller's next message is about transferring a call. Answer from the Call transfer fact: "
    f"{DEMO_FACTS['call_transfer']}. Never say you can't transfer calls. Offer to have the team call them back."
)


def hint_for(text: str) -> str | None:
    """'price', 'transfer', or None for what the caller just said."""
    t = " ".join((text or "").split())
    if _TRANSFER.search(t):
        return "transfer"
    if _PRICE.search(t) and _PRODUCT.search(t) and not _DENTAL.search(t):
        return "price"
    return None


NOTES = {"price": PRICE_NOTE, "transfer": TRANSFER_NOTE}


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
