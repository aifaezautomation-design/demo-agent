"""Weekday-versus-date check for every call: the model is told the exact weekday and date a caller names.

A caller says "Friday the 23rd" and the model, asked whether those go together, did the calendar arithmetic itself and told
a caller that Friday, October 23 was a Sunday. So the arithmetic is done here, in code (`date_note` in app/tools.py), and
the answer is put in front of the model, like the emergency and demo notes: a regex on the transcript, no extra model call
and no added delay.
"""

from datetime import datetime

from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    Frame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.tools import date_note


class DateCheck(FrameProcessor):
    """Sits between speech-to-text and the user aggregator. Each distinct note is given once per call."""

    def __init__(self, session, language_state=None):
        super().__init__()
        self._session = session
        self._language = language_state
        self._given: set[str] = set()

    def note_for(self, text: str) -> LLMMessagesAppendFrame | None:
        """The note to add ahead of this transcript, or None. Also used by the test scripts."""
        if self._language is not None and self._language.current != "en":
            return None
        today = datetime.now(self._session.tz).date()
        note = date_note(text, today)
        if not note or note in self._given:
            return None
        self._given.add(note)
        return LLMMessagesAppendFrame([{"role": "developer", "content": note}], run_llm=False)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (TranscriptionFrame, EagerTranscriptionFrame)):
            note = self.note_for(frame.text)
            if note:
                await self.push_frame(note, direction)  # into the context before the caller's message
        await self.push_frame(frame, direction)
