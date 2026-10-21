"""Keeps Mia from ending every reply with a stock offer of help (demo line only).

The prompt asks for it, but gpt-4o-mini swaps one closing question for another ("What else can I help you with?",
"Is there anything else you'd like to know?") and offers a test booking in most replies, which sounds robotic and
pushy. So the rule is enforced in code, on the sentences the model is about to speak:

  * a generic closing offer ("How can I assist you today?", "Is there anything else...?", "Do you have any other
    questions?") is dropped;
  * an offer of a test booking is allowed at most twice per call and never in two replies in a row.

Only those sentences are ever dropped. A question the conversation needs ("Which day works?", "Would you like to book
that time?", "Is that correct?") is never touched, and a reply is never left empty: if everything in it would be
dropped, the last sentence is kept.
"""
import re

from pipecat.frames.frames import (
    CancelFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Sentences that open like a closing offer. A streamed sentence is held back only while its first words still match one
# of these, so every other sentence reaches the voice with no added delay.
_STARTS = (
    "how can i", "how may i", "what else", "what can i", "is there anything", "is there something", "anything else",
    "anything i can", "do you have any other", "do you have any more", "do you have any further", "can i help you with anything",
    "can i assist", "would you like", "want to", "do you want", "if you have more", "if you have any", "feel free",
    "let me know if", "is there anything else", "care to", "should i", "could i", "may i", "would you prefer", "would it help", "shall we", "if you", "you can", "i can", "let's", "why not", "how about",
)

_GENERIC = re.compile(
    r"^(how (can|may) i (assist|help)( you)?( with)?( today| further)?|"
    r"what (else|more) can i (help|do|assist)|what can i (help|assist) you with( today)?|"
    r"(is there )?(anything|something) else( i can (help|assist) (you )?with| you('d| would) (like|want) (to )?(know|ask|hear|discuss)| i can do)?|"
    r"do you have (any|some)( other| more| further)? questions|"
    r"anything i can (help|assist) you (with|book)|can i (help|assist) you with anything else|"
    r"if you have (any )?(more |other |further )?questions[^.?!]*|feel free to (ask|reach out)[^.?!]*|"
    r"let me know if (there's|there is|you need|you have)[^.?!]*)\b[^.?!]*[.?!]?\s*$",
    re.I,
)
_TEST_WORDS = re.compile(r"\b(test|demo|trial)\s+(booking|appointment|visit)\b", re.I)
# Statement-style offers: "If you're interested, I can help set up a test booking."
_TEST_OFFER = re.compile(
    r"^(would you like|want|do you want|care|shall we|if you('re| are| would|'d)|feel free|you can|i can|let's|why not|how about)\b[^.?!]*[.?!]?\s*$",
    re.I,
)
# An offer of a call with the team ("Would you like to set up a time?", "Want me to have them reach out?").
_NEXT_STEP = re.compile(
    r"^(would you like|want|do you want|shall i|can i|care to)\b[^.?!]*\b(set up a time|schedule (a|that|this)|book a call|"
    r"have (them|the team|someone) (reach|call|contact)|reach out to you|call you|take your (details|name))\b[^.?!]*[.?!]?\s*$",
    re.I,
)
# Any question that offers something: "Would you like to know more about how I can assist?", "Would you like to book a call with
# them?", "Would you like me to arrange that?", "Want me to have the team call you with a quote?".
_QUESTION_OFFER = re.compile(
    r"^(would you like|would you prefer|would it help|do you want|want|shall i|should i|can i|could i|may i|care to|how about)\b"
    r"[^.?!]*\?\s*$",
    re.I,
)
# Questions the conversation needs, which are never offers: the callback and booking flows ask for a name, a practice, a
# number, a yes, a day or a time.
_NEEDED = re.compile(
    r"\b(your (full )?name|your practice|the name of your|full name|best one to reach|the number|your number|is that correct|"
    r"is that right|new or returning|reason for|which (day|time|one|works)|what (day|time)|works (best|better)|what's the name|"
    r"go ahead and book|book that|that time|those times|that slot|proceed|noon|a\.?m\.?|p\.?m\.?|morning|afternoon|evening|"
    r"tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\d",
    re.I,
)


# The answer to "how do I get started?" names the ways to do it. It is an answer, not an offer: never cut, never counted.
_NEXT_STEP_ANSWER = re.compile(r"dot com|faez'?s email|reply to (faez|the email)|email from boltix", re.I)


def is_next_step_answer(text: str) -> bool:
    return bool(_NEXT_STEP_ANSWER.search(text.replace("’", "'")))


def is_offer(text: str) -> bool:
    """True when a sentence is a closing offer: it offers something (a test booking, a call, more detail) rather than asking for
    what the conversation needs."""
    t = text.strip()
    if is_next_step_answer(t):
        return False
    if _TEST_WORDS.search(t) and _TEST_OFFER.match(t):
        return True
    if _NEEDED.search(t):
        return False
    return bool(_NEXT_STEP.match(t) or _QUESTION_OFFER.match(t))


MAX_OFFERS = 2  # closing offers kept per call
MAX_REPLY_WORDS = 38  # a closing offer that would take a reply past this is dropped


def could_become_offer(prefix: str) -> bool:
    """True while the start of a sentence still looks like the start of a closing offer."""
    p = re.sub(r"\s+", " ", prefix.lstrip()).lower()
    if not p:
        return True
    return any(s.startswith(p) or p.startswith(s) for s in _STARTS)


class OfferPolicy:
    """Decides, sentence by sentence, whether a closing offer may be spoken. One per call."""

    def __init__(self):
        self.offers = 0  # closing offers kept so far in the call
        self._words = 0  # words kept so far in the reply being judged
        self._next_step_reply = False  # this reply already named the ways to get started
        self._offered_this_reply = False
        self._offered_last_reply = False

    def keep(self, sentence: str) -> bool:
        kept = self._keep(sentence)
        if kept:
            self._words += len(sentence.split())
        return kept

    def _keep(self, sentence: str) -> bool:
        text = sentence.strip()
        if not text:
            return True
        if _GENERIC.match(text):
            return False
        if is_next_step_answer(text):
            self._next_step_reply = True
            return True  # the way to get started is the answer: kept, and not one of the offers
        if self._next_step_reply and _QUESTION_OFFER.match(text) and re.search(r"details|call you|reach out", text, re.I):
            return True  # "...or would you like me to take your details?" completing the same answer
        if is_offer(text):
            if (self._words + len(text.split()) > MAX_REPLY_WORDS  # a long answer loses its closing offer
                    or self._offered_last_reply  # never in two replies in a row
                    or self.offers >= MAX_OFFERS):  # and at most twice in the whole call
                return False
            self.offers += 1
            self._offered_this_reply = True
            return True
        if _TEST_WORDS.search(text):
            # A factual answer that mentions a test booking ("This is a demo practice, so feel free to book a test
            # appointment") stays, but it still uses up one of the offers.
            self.offers += 1
            self._offered_this_reply = True
        return True

    def end_reply(self) -> None:
        self._words = 0
        self._next_step_reply = False
        self._offered_last_reply, self._offered_this_reply = self._offered_this_reply, False

    def trim(self, reply: str) -> str:
        """The whole reply at once (for scripts and tests). Never returns an empty reply."""
        sentences = [s for s in re.findall(r"[^.!?]*[.!?]+\s*|[^.!?]+$", reply) if s.strip()]
        kept = [s for s in sentences if self.keep(s)]
        self.end_reply()
        return "".join(kept) if kept else (sentences[-1] if sentences else reply)


class OfferTrimmer(FrameProcessor):
    """Streaming version of OfferPolicy: sits between the LLM and the voice."""

    def __init__(self, enabled: bool = True):
        super().__init__()
        self._enabled = enabled
        self._policy = OfferPolicy()
        self._reset()

    def _reset(self) -> None:
        self._sentence = ""  # the sentence being held back while it still looks like an offer
        self._plain = False  # the current sentence is plainly not an offer: text passes straight through
        self._pushed = False  # something was spoken in this reply
        self._dropped: list[str] = []
        self._tool = False  # a tool call was made in this reply

    async def _release(self, text: str) -> None:
        if text:
            self._pushed = True
            await self.push_frame(LLMTextFrame(text))

    async def _sentence_done(self, sentence: str) -> None:
        if self._policy.keep(sentence):
            await self._release(sentence)
        else:
            self._dropped.append(sentence)

    async def _text(self, text: str) -> None:
        for piece in re.findall(r"[^.!?]*[.!?]+\s*|[^.!?]+$", text):
            ends = bool(re.search(r"[.!?]\s*$", piece))
            if self._plain:
                await self._release(piece)
            else:
                self._sentence += piece
                if not could_become_offer(self._sentence) and not ends:
                    self._plain = True
                    await self._release(self._sentence)
                    self._sentence = ""
                    continue
                if not ends:
                    continue
            if ends:
                sentence, self._sentence, was_plain, self._plain = self._sentence, "", self._plain, False
                if sentence and not was_plain:
                    await self._sentence_done(sentence)

    async def _flush_sentence(self) -> None:
        if self._sentence:  # the model stopped mid-sentence
            sentence, self._sentence = self._sentence, ""
            await self._sentence_done(sentence)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not self._enabled or direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame):
            await self._text(frame.text)
        elif isinstance(frame, FunctionCallsStartedFrame):
            await self._flush_sentence()
            self._tool = True  # a tool call follows: an empty spoken reply is fine
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush_sentence()
            if self._dropped and not self._pushed and not self._tool:
                await self._release(self._dropped[-1])  # never leave the caller with silence
            self._policy.end_reply()
            self._reset()
            await self.push_frame(frame, direction)
        elif isinstance(frame, (InterruptionFrame, CancelFrame, LLMFullResponseStartFrame)):
            self._reset()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
