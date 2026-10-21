"""LLM tools for one call. State lives on CallSession so handlers can enforce
guardrails in code: only slots offered this call can be booked, and only
appointments looked up this call can be changed."""

import asyncio
import difflib
import re
import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    EndWorkerFrame,
    FunctionCallResultProperties,
    TTSSpeakFrame,
)
from pipecat.services.llm_service import FunctionCallParams, LLMService

from app import config, db, notify
from app import language as language_module
from app.calcom import CalCom, CalComError
from app.db import Tenant
from app.emergency import classify

_MAX_OPTIONS = 12  # the day's times, so the model never claims a time is taken because it saw only a sample; the prompt tells it to offer two
_SEARCH_DAYS = 7
_EMERGENCY_WINDOW = timedelta(hours=2)
_EMERGENCY_SEARCH = timedelta(hours=48)
_MIN_LEAD_TIME = timedelta(minutes=30)
# Openings are loaded once while the greeting plays, so the first calendar question skips the Cal.com round trip
# (about 0.8 s). The copy is used for up to two minutes, and dropped after any booking, change, or cancellation.
_PREFETCH_DAYS = 14
_PREFETCH_MAX_AGE_SECS = 120.0


def phone_matches_caller_id(phone: str | None, caller: str | None) -> bool:
    return bool(phone and caller and _digits(phone)[-10:] == _digits(caller)[-10:])


def normalize_phone(raw: str | None) -> str | None:
    """Best-effort E.164 for North American numbers; other inputs pass through as +digits."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}" if digits else None


def spoken_time(dt: datetime) -> str:
    """'Tuesday, October 27 at 1 PM'. Midday is spoken as people say it: 'noon', and 12:30 as '12:30 in the afternoon'."""
    day = f"{dt:%A, %B} {dt.day}"
    if dt.hour == 12:
        return f"{day} at noon" if dt.minute == 0 else f"{day} at 12:{dt.minute:02d} in the afternoon"
    hour = dt.hour % 12 or 12
    minutes = "" if dt.minute == 0 else f":{dt.minute:02d}"
    return f"{day} at {hour}{minutes} {'AM' if dt.hour < 12 else 'PM'}"


_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())} | {"sept": 9})
_NUMBER_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def _add_months_day(today: date, day: int, month: int | None) -> date | None:
    """The next occurrence of `day` (of `month`, or of this/next month when none is given)."""
    for offset in range(0, 14 if month is None else 2):
        if month is None:
            y, m = today.year + (today.month - 1 + offset) // 12, (today.month - 1 + offset) % 12 + 1
        else:
            y, m = today.year + offset, month
        try:
            candidate = date(y, m, day)
        except ValueError:
            continue
        if candidate >= today:
            return candidate
    return None


def resolve_day(text: str | None, today: date) -> tuple[date, bool]:
    """Turns what the LLM passes for the day into a real date, in code, because models are bad at
    calendar arithmetic ("next week Thursday" used to become a Tuesday).

    Accepts YYYY-MM-DD, today, tomorrow, a weekday ("friday", "this friday": the coming one),
    "next <weekday>" (that weekday in next week, weeks starting Monday), "next week" (its Monday),
    "the week after next [weekday]", "in N days/weeks", and calendar dates ("the 26th", "october 26").
    Returns (date, understood).
    """
    s = (text or "").strip().lower().replace(",", " ")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return max(datetime.strptime(s, "%Y-%m-%d").date(), today), True
        except ValueError:
            return today, False
    if s in ("", "today", "now", "asap", "this afternoon", "this morning", "tonight"):
        return today, True
    if s in ("tomorrow", "tmrw"):
        return today + timedelta(days=1), True
    if s == "day after tomorrow":
        return today + timedelta(days=2), True

    words = s.split()
    m = re.fullmatch(r"in (\w+) (day|days|week|weeks)", s)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS.get(m.group(1))
        if n:
            return today + timedelta(days=n * (7 if m.group(2).startswith("week") else 1)), True

    next_monday = today + timedelta(days=7 - today.weekday())
    weekday = next((_WEEKDAYS[w] for w in words if w in _WEEKDAYS), None)
    if "after next" in s and "week" in s:  # "the week after next [weekday]"
        return next_monday + timedelta(days=7 + (weekday or 0)), True
    if weekday is not None:
        if "next" in words:
            return next_monday + timedelta(days=weekday), True
        return today + timedelta(days=(weekday - today.weekday()) % 7), True
    if s in ("next week", "following week"):
        return next_monday, True

    month = next((_MONTHS[w] for w in words if w in _MONTHS), None)
    num = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", s)
    if num and 1 <= int(num.group(1)) <= 31:
        found = _add_months_day(today, int(num.group(1)), month)
        if found:
            return found, True
    return today, False


def parse_time_pref(text: str | None) -> time | None:
    """'14:00', '2pm', '2:30 pm', '2 PM' -> time; None when absent or unreadable."""
    if not text:
        return None
    m = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*", text.strip().lower())
    if not m:
        return None
    hour, minute, suffix = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "")
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    return time(hour, minute) if hour < 24 and minute < 60 else None


_CALLER_TIME = re.compile(
    r"(?<![\w:])(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![\w])|(?<![\w:])at\s+(\d{1,2})(?::(\d{2}))?(?![\w:/])", re.I
)


def time_from_caller(text: str | None) -> str | None:
    """A clock time the caller clearly named ('9 AM', '2:30 pm', 'at 9'), as text parse_time_pref reads; else None.
    A bare 'at 9' is taken as business hours: 1-6 means afternoon, 7-11 morning, 12 noon."""
    if not text:
        return None
    m = None
    for m in _CALLER_TIME.finditer(text):
        pass  # the last time mentioned is the one the caller means now
    if not m:
        return None
    if m.group(3):
        hour, minute, suffix = int(m.group(1)), m.group(2) or "00", m.group(3).replace(".", "").lower()
    else:
        hour, minute, suffix = int(m.group(4)), m.group(5) or "00", ""
        suffix = "pm" if 1 <= hour <= 6 else "am"
    return f"{hour}:{minute} {suffix}" if 1 <= hour <= 12 else None


_DAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def slot_in_hours(kb: dict, slot_local: datetime) -> bool:
    """True when an appointment starting at `slot_local` (practice time) fits inside the practice's opening hours.

    Cal.com decides what is bookable, and its schedule can disagree with the practice's real hours (it did:
    9-8 Mon-Sat in Eastern time against 8-6 / 9-2 in Central). This check makes sure a slot outside the hours
    in the knowledge base is never offered or booked. Without `hours_schedule` in the knowledge base it allows
    everything, so tenants that haven't configured it behave as before.
    """
    schedule = kb.get("hours_schedule")
    if not schedule:
        return True
    window = schedule.get(_DAY_KEYS[slot_local.weekday()])
    if not window:
        return False  # closed that day
    opens, closes = (time.fromisoformat(t) for t in window)
    end = slot_local + timedelta(minutes=int(kb.get("appointment_minutes", 60)))
    return opens <= slot_local.time() and end.date() == slot_local.date() and end.time() <= closes


_PLACEHOLDER_NAMES = {
    "unknown", "caller", "patient", "anonymous", "unnamed", "na", "n/a", "none", "test", "name", "john doe", "jane doe",
    "new patient", "the caller", "not provided", "no name",
}


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


_READBACK = re.compile(r"(confirm|correct|that right|sound(s)? (good|right)|does that work|shall i (book|schedule)|"
                       r"should i (book|schedule)|would you like me to (book|schedule)|go ahead and book)", re.I)
_YES = re.compile(r"\b(yes|yeah|yep|yup|yea|correct|right|sure|okay|ok|perfect|absolutely|definitely|please|"
                  r"go ahead|sounds good|that works|that's right|that is correct|uh-huh|mm-hmm)\b", re.I)
_NO = re.compile(r"^\W*(no|nope|nah|wait|actually|hold on|not)\b", re.I)


def is_readback(said: str) -> bool:
    """Mia's line was a confirmation question (\"Just to confirm ... Is that correct?\")."""
    return said.strip().endswith("?") and bool(_READBACK.search(said))


def is_affirmative(heard: str) -> bool:
    return bool(_YES.search(heard)) and not _NO.search(heard.strip())


# Insults aimed at Mia. A patient who swears about pain, a tooth, a dentist or themselves ("this stupid tooth is killing
# me", "the pain is shit", "I feel so dumb") must never be counted, or Mia could hang up on an emergency. So an insult
# word counts only when it sits within a couple of words of a word that points at her, or is a direct attack by itself.
_INSULT = r"(stupid|idiot|moron|useless|worthless|garbage|trash|pathetic|dumb|shit+y?|crap+y?|bitch|asshole|bastard|suck|sucks)"
_AT_MIA = r"(you|you're|youre|your|ur|u|robot|bot|ai|mia|machine)"
_GAP = r"(?:[\w']+[\s,]+){0,2}"
_AIMED = re.compile(
    rf"(?<![\w'])(?:{_AT_MIA}[\s,]+{_GAP}{_INSULT}|{_INSULT}[\s,]+{_GAP}{_AT_MIA})(?![\w'])", re.I
)
_DIRECT = re.compile(r"(?<![\w'])(shut up|screw you|f+u+c+k+ (you|off|u)|go to hell|piece of shit,? you)", re.I)
_THANKS = re.compile(r"(thank you|thanks to you|bless you|love you)", re.I)


def is_abusive(text: str) -> bool:
    """True when a sentence insults Mia herself. Sentences are judged one at a time."""
    for sentence in re.split(r"[.!?;]+", text):
        sentence = _THANKS.sub(" ", sentence).strip()
        if sentence and (_DIRECT.search(sentence) or _AIMED.search(" " + sentence + " ")):
            return True
    return False


def spoken_day(d: date) -> str:
    return f"{d:%A, %B} {d.day}"


def _slot_id(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _matches_time_of_day(dt: datetime, time_of_day: str) -> bool:
    if time_of_day == "morning":
        return dt.hour < 12
    if time_of_day == "afternoon":
        return 12 <= dt.hour < 17
    if time_of_day == "evening":
        return dt.hour >= 17
    return True


class CallSession:
    def __init__(self, tenant: Tenant, call_id: str, caller_phone: str | None, call_log_ready: asyncio.Future | None = None):
        self.tenant = tenant
        self.call_id = call_id
        self._tools_running = 0
        self.said: list[str] = []  # what Mia has said, in order; the booking guard reads the last line
        self.turn_searched = False
        self._rejected: dict[tuple[str, str], int] = {}
        # The call_logs insert runs in the background so the greeting isn't delayed;
        # anything that references it (appointments FK, billing) awaits this first.
        self._call_log_ready = call_log_ready
        self.language_state = language_module.LanguageState()
        # Set by the pipeline: queues the settings frames that move STT and the voice to a new language.
        self.apply_language = None
        self._pending: set[asyncio.Task] = set()
        self.caller_phone = normalize_phone(caller_phone)
        self.tz = ZoneInfo(tenant.timezone)
        self.calcom = CalCom(tenant.cal_com_api_key) if tenant.cal_com_api_key else None
        self.offered_slots: dict[str, datetime] = {}
        self.found_appointments: set[str] = set()
        # Rows returned by find_appointments, kept for the call so changing an appointment needs no second database read.
        self._appt_cache: dict[str, dict] = {}
        # Everything the caller has said, so a booking can be checked against it (see _name_is_from_caller).
        self.heard: list[str] = []
        self.appointment_booked = False
        self.writes = 0  # bookings, reschedules and cancellations really saved this call
        # Demo line only: the callback request the caller made, if any (see request_boltix_callback).
        self.callback: dict | None = None
        self.abuse_strikes = 0  # caller messages that insulted Mia herself (demo line: the second one ends the call)
        self.emergency_seen = False  # an emergency/airway triage happened in this call: strikes stop and never end the call
        # (fetched_at, event_type_id, start, end, task) from prefetch_slots, or None.
        self._prefetch: tuple[float, int, datetime, datetime, asyncio.Task] | None = None

    # ------------------------------------------------------------------ calendar cache

    def prefetch_slots(self) -> None:
        """Starts loading the next two weeks of openings in the background (called when the call starts)."""
        if not self.calcom or not self.tenant.cal_com_event_type_id:
            return
        now = datetime.now(self.tz)
        end = datetime.combine(now.date() + timedelta(days=_PREFETCH_DAYS), time.min, self.tz)
        event_type = self.tenant.cal_com_event_type_id
        task = asyncio.ensure_future(self.calcom.get_slots(event_type, now, end, self.tenant.timezone))
        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())  # no "never retrieved" warning
        self._prefetch = (time_module.monotonic(), event_type, now, end, task)

    def _drop_prefetch(self) -> None:
        self._prefetch = None

    async def _get_slots(self, event_type: int, start: datetime, end: datetime) -> list[datetime]:
        """Openings from the prefetched copy when it covers the request and is fresh; otherwise straight from Cal.com."""
        calcom = self._require_calendar()
        pf = self._prefetch
        if pf:
            fetched_at, pf_type, pf_start, pf_end, task = pf
            if (pf_type == event_type and start >= pf_start and end <= pf_end
                    and time_module.monotonic() - fetched_at < _PREFETCH_MAX_AGE_SECS):
                try:
                    slots = await asyncio.shield(task)
                    return [s for s in slots if start <= s < end]
                except Exception as e:
                    logger.warning(f"[{self.call_id}] prefetched slots unusable ({e!r}); asking Cal.com directly")
                    self._drop_prefetch()
        return await calcom.get_slots(event_type, start, end, self.tenant.timezone)

    # ------------------------------------------------------------------ setup

    def tools_schema(self) -> ToolsSchema:
        slot_id = {"type": "string", "description": "slot_id from check_availability."}
        phone_last4 = {
            "type": "string",
            "description": "Last 4 digits of the appointment phone; needed if found by name or the number differs from caller ID.",
        }
        return ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="check_availability",
                    description="Find open slots. Call before offering any time.",
                    properties={
                        "date": {
                            "type": "string",
                            "description": "The caller's own words for the day, complete and in English: today, tomorrow, friday, next thursday, friday next week. The system works out the exact date; never compute it. YYYY-MM-DD only if the caller gave a calendar date.",
                        },
                        "time_of_day": {"type": "string", "enum": ["morning", "afternoon", "evening", "any"]},
                        "is_emergency": {"type": "boolean", "description": "True: soonest same-day slots."},
                        "preferred_time": {"type": "string", "description": "A time the caller named, e.g. 14:00 or 2pm."},
                    },
                    required=["date", "time_of_day", "is_emergency"],
                ),
                FunctionSchema(
                    name="book_appointment",
                    description="Book a slot after the caller confirmed the read-back.",
                    properties={
                        "patient_name": {"type": "string", "description": "Full name in Latin letters (transliterate if the caller spoke another script)."},
                        "phone_number": {"type": "string", "description": "Contact number, digits."},
                        "visit_reason": {"type": "string", "description": "e.g. Routine Cleaning, Toothache, Crown/Filling, Cosmetic."},
                        "is_new_patient": {"type": "boolean"},
                        "is_emergency": {"type": "boolean"},
                        "slot_id": slot_id,
                    },
                    required=["patient_name", "phone_number", "visit_reason", "is_new_patient", "is_emergency", "slot_id"],
                ),
                FunctionSchema(
                    name="find_appointments",
                    description="Find upcoming appointments by phone (preferred) or full name.",
                    properties={
                        "phone_number": {"type": "string"},
                        "patient_name": {"type": "string"},
                    },
                    required=[],
                ),
                FunctionSchema(
                    name="reschedule_appointment",
                    description="Move an appointment from find_appointments to a new slot.",
                    properties={
                        "appointment_id": {"type": "string"},
                        "slot_id": slot_id,
                        "phone_last4": phone_last4,
                    },
                    required=["appointment_id", "slot_id"],
                ),
                FunctionSchema(
                    name="cancel_appointment",
                    description="Cancel an appointment from find_appointments, after confirmation.",
                    properties={
                        "appointment_id": {"type": "string"},
                        "reason": {"type": "string"},
                        "phone_last4": phone_last4,
                    },
                    required=["appointment_id", "reason"],
                ),
                FunctionSchema(
                    name="flag_urgent_callback",
                    description="Flag an urgent staff callback when no emergency slot exists within two hours.",
                    properties={"reason": {"type": "string", "description": "Short emergency description."}},
                    required=["reason"],
                ),
                *self._language_tool(),
                *self._demo_tools(),
                FunctionSchema(
                    name="end_call",
                    description="Say goodbye and hang up.",
                    properties={"farewell": {"type": "string", "description": "One short goodbye."}},
                    required=["farewell"],
                ),
            ]
        )

    def _demo_tools(self) -> list[FunctionSchema]:
        """Tools that exist only on the demo line. A production practice never sees them."""
        if not self.tenant.demo_mode:
            return []
        return [
            FunctionSchema(
                name="request_boltix_callback",
                description=(
                    "Demo line only: save a request for the Boltix AI team to call this caller back. "
                    "Call it only after the caller said yes to the read-back of their name, practice and number."
                ),
                properties={
                    "name": {"type": "string", "description": "The caller's name, exactly as they said it."},
                    "practice_name": {"type": "string", "description": "The name of the caller's practice."},
                    "phone_number": {"type": "string", "description": "Number to call back: the caller ID or one the caller said."},
                    "best_time": {"type": "string", "description": "When they prefer a call, in their words. Optional."},
                    "notes": {"type": "string", "description": "What they are interested in. Optional."},
                },
                required=["name", "practice_name", "phone_number"],
            )
        ]

    def _language_tool(self) -> list[FunctionSchema]:
        if not config.MULTILINGUAL_ENABLED:
            return []
        return [
            FunctionSchema(
                name="set_language",
                description="Switch the voice and speech recognition to the caller's language. Needed for Urdu; other languages switch automatically.",
                properties={"language": {"type": "string", "enum": list(language_module.SUPPORTED)}},
                required=["language"],
            )
        ]

    def register(self, llm: LLMService) -> None:
        handlers = {
            "check_availability": self.check_availability,
            "book_appointment": self.book_appointment,
            "find_appointments": self.find_appointments,
            "reschedule_appointment": self.reschedule_appointment,
            "cancel_appointment": self.cancel_appointment,
            "flag_urgent_callback": self.flag_urgent_callback,
        }
        if config.MULTILINGUAL_ENABLED:
            handlers["set_language"] = self.set_language
        if self.tenant.demo_mode:
            handlers["request_boltix_callback"] = self.request_boltix_callback
        for name, handler in handlers.items():
            llm.register_function(name, self._wrap(name, handler), timeout_secs=12)
        llm.register_function("end_call", self.end_call)

    @property
    def wrote(self) -> bool:
        return self.writes > 0

    @property
    def abuse_count(self) -> int:
        """Strikes that count toward ending the call: none once an emergency has been reported."""
        return 0 if self.emergency_seen else self.abuse_strikes

    @property
    def busy(self) -> bool:
        """True while a tool call is in flight, so the idle timer never mistakes our own work for the caller's silence."""
        return self._tools_running > 0

    def _wrap(self, name: str, handler):
        # Errors go back to the LLM as data so Mia can recover in conversation
        # instead of the pipeline logging a stack trace and going silent.
        async def run(params: FunctionCallParams):
            started = time_module.monotonic()
            self._tools_running += 1
            try:
                result = await handler(**dict(params.arguments))
            except CalComError as e:
                logger.warning(f"[{self.call_id}] {name} Cal.com failure: {e}")
                if name in ("check_availability", "book_appointment", "reschedule_appointment", "cancel_appointment"):
                    self._flag_urgent(f"Calendar unavailable during {name}; the caller needs a callback")
                result = {"error": "calendar_unavailable", "say": "Apologize and offer to have the team call back."}
            except Exception:
                logger.exception(f"[{self.call_id}] {name} failed")
                result = {"error": "internal_error", "say": "Apologize and offer to have the team call back."}
            finally:
                self._tools_running -= 1
            logger.info(f"[{self.call_id}] tool {name} took {(time_module.monotonic() - started) * 1000:.0f} ms")
            await params.result_callback(result)

        return run

    # ---------------------------------------------------------------- background writes

    def _background(self, factory, what: str) -> None:
        """Runs a database write off the caller's critical path, tracked so it is never lost.

        `factory` returns a fresh coroutine each time, so a failed write can be retried once.
        """

        async def run():
            if self._call_log_ready is not None:
                await asyncio.shield(self._call_log_ready)
            for attempt in (1, 2):
                try:
                    return await factory()
                except Exception:
                    logger.exception(f"[{self.call_id}] background write '{what}' failed (attempt {attempt})")
                    await asyncio.sleep(0.5)
            logger.error(f"[{self.call_id}] background write '{what}' LOST; reconcile manually")

        task = asyncio.ensure_future(run())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def _flag_urgent(self, reason: str) -> None:
        """Marks the call for an urgent staff callback, behind the reply (the caller never waits on this write)."""
        self._background(lambda: db.flag_urgent_callback(self.call_id, reason[:500]), "urgent callback")

    async def flush(self) -> None:
        """Waits for background writes. Called before reading them back and at call end."""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    def note_heard(self, text: str) -> None:
        self.heard.append(text)
        self.turn_searched = False  # a new caller turn: any availability claim now needs a fresh search
        if classify(text):  # an emergency or airway report: a patient in pain is never "rude", whatever they say
            self.emergency_seen = True
        elif not self.emergency_seen and is_abusive(text):
            self.abuse_strikes += 1

    def note_said(self, text: str) -> None:
        self.said.append(text)

    def _heard_words(self) -> list[str]:
        text = re.sub(r"[^\w']+", " ", " ".join(self.heard).lower())  # punctuation and hyphens split words
        words = text.split()
        # "s i o b h a n" (a name spelled aloud, with or without hyphens) also counts as the word it spells
        run: list[str] = []
        for w in words + [""]:
            if len(w) == 1:
                run.append(w)
                continue
            if len(run) >= 3:
                words.append("".join(run))
            run = []
        return [w.strip("'") for w in words]

    def _name_is_from_caller(self, name: str) -> bool:
        """The model must not book under a name the caller never gave (it did, in testing: it called
        book_appointment right after 'I knocked out my tooth', before asking who was calling)."""
        if (name or "").strip().lower() in _PLACEHOLDER_NAMES:
            return False
        if config.MULTILINGUAL_ENABLED:
            return True  # transcripts may be in another script, so a letter-by-letter check would reject real names
        heard_words = self._heard_words()
        tokens = [t.strip("'").lower() for t in re.split(r"[\s\-]+", name) if len(t.strip("'")) >= 2]
        if not tokens:
            return False
        return all(difflib.get_close_matches(tok, heard_words, n=1, cutoff=0.75) for tok in tokens)

    def _caller_confirmed(self) -> bool:
        """True when Mia's last line was a read-back question and the caller's last words were a yes.
        The model skipped the read-back in testing and booked straight after the caller picked a time."""
        return bool(self.said and self.heard) and is_readback(self.said[-1]) and is_affirmative(self.heard[-1])

    def _number_was_asked(self) -> bool:
        """Mia asked about the contact number at some point in the call (the caller may have volunteered one instead)."""
        return any(("number" in t.lower() or "phone" in t.lower()) and "?" in t for t in self.said) or any(
            len(_digits(h)) >= 10 for h in self.heard
        )

    def _snap_to_caller_id(self, phone: str) -> str:
        """The model reads the caller ID aloud and passes it back, and garbles digits doing it (it booked +5122... for a
        +1512... caller). A number one or two digits off the caller ID that the caller never said is the caller ID."""
        caller = _digits(self.caller_phone or "")[-10:]
        mine = _digits(phone)[-10:]
        if caller and mine != caller and mine not in _digits(" ".join(self.heard)):
            if difflib.SequenceMatcher(None, mine, caller).ratio() >= 0.8:
                return self.caller_phone
        return phone

    def _insist(self, kind: str, value: str) -> bool:
        """True once the model offers the same unverifiable value a second time. The guard exists to stop invented
        details, not to trap a caller whose real name the speech recognizer spelled differently: the read-back
        the caller already confirmed is the second check, so the second attempt goes through."""
        key = (kind, re.sub(r"\W+", "", value.lower()))
        self._rejected[key] = self._rejected.get(key, 0) + 1
        return self._rejected[key] >= 2

    def _phone_is_from_caller(self, phone: str) -> bool:
        number = _digits(phone)[-10:]
        if len(number) < 10:
            return False
        return phone_matches_caller_id(phone, self.caller_phone) or number in _digits(" ".join(self.heard))

    def _require_calendar(self) -> CalCom:
        if not self.calcom:
            raise CalComError("tenant has no Cal.com API key configured")
        return self.calcom

    # ------------------------------------------------------------------ tools

    async def check_availability(
        self, date: str, time_of_day: str = "any", is_emergency: bool = False, preferred_time: str | None = None
    ) -> dict:
        calcom = self._require_calendar()
        now = datetime.now(self.tz)
        self.turn_searched = True

        if is_emergency:
            return await self._emergency_slots(calcom, now)

        day, understood = resolve_day(date, now.date())
        if not preferred_time:
            # The model sometimes searches the whole day and then tells the caller their time is unavailable. A time the
            # caller just named is always searched for.
            preferred_time = time_from_caller(self.heard[-1] if self.heard else None)
        wanted = parse_time_pref(preferred_time)

        start = max(datetime.combine(day, time.min, self.tz), now + _MIN_LEAD_TIME)
        end = datetime.combine(day + timedelta(days=_SEARCH_DAYS), time.min, self.tz)
        slots = await self._get_slots(self.tenant.cal_com_event_type_id, start, end)
        slots = self._open_hours(s.astimezone(self.tz) for s in slots)
        part_of_day_missed = False
        if wanted is None:  # a named time beats a vague part of the day
            matching = [s for s in slots if _matches_time_of_day(s, time_of_day)]
            part_of_day_missed = bool(slots) and not matching and time_of_day not in ("any", None, "")
            slots = matching or slots  # never report "nothing open" while the calendar has openings, just not at that hour

        on_day = [s for s in slots if s.date() == day]
        pool = on_day
        if not pool and slots:  # nothing that day: the nearest day that has something
            pool = [s for s in slots if s.date() == slots[0].date()]

        if wanted is not None:  # closest to the requested time first
            target = wanted.hour * 60 + wanted.minute
            pool = sorted(pool, key=lambda s: (abs(s.hour * 60 + s.minute - target), s))
        result = self._offer(pool[:_MAX_OPTIONS], none_note=f"nothing_open_in_the_next_{_SEARCH_DAYS}_days")

        if not result["options"]:
            self._flag_urgent(f"Caller wanted an appointment ({date}); no openings found in the next {_SEARCH_DAYS} days")
            result["say"] = ("Nothing is open. Say so briefly and that a team member will call them back to find a time. "
                             "Do not call it urgent and do not call any tool.")
        result["searched_day"] = spoken_day(day)
        result["same_day"] = bool(on_day)
        shown = len(result["options"])
        if shown and len(on_day) > shown:
            # The options are a sample. Without this the model reads "I was shown three times" as "only three exist".
            result["more_open_that_day"] = (
                f"{len(on_day) - shown} more times are open that day. If the caller wants a different time or part of the day, "
                "call check_availability again with preferred_time or time_of_day. Never say a time is unavailable without searching for it."
            )
        if wanted is not None:
            exact = [s for s in on_day if s.hour == wanted.hour and s.minute == wanted.minute]
            result["exact_match"] = bool(exact)
            result["answer"] = (
                f"The caller's time IS open: {spoken_time(exact[0])}. Offer exactly that time first." if exact else
                "The caller's exact time is NOT open; offer the two closest options (listed first)."
            )
        if not understood:
            result["note"] = f"Could not read the requested day {date!r}; searched {spoken_day(day)}. Ask the caller to confirm the day."
        elif part_of_day_missed:
            result["note"] = f"Nothing open in the {time_of_day}; these are the closest openings. Say so, then offer them."
        elif pool and not on_day:
            result["note"] = (
                f"Nothing open on {spoken_day(day)}. These are on {spoken_day(pool[0].date())}, the nearest day with openings."
            )
        return result

    async def emergency_search(self) -> dict:
        """The same-day emergency search, callable without the LLM (used by app/emergency.py)."""
        return await self._emergency_slots(self._require_calendar(), datetime.now(self.tz))

    async def _emergency_slots(self, calcom: CalCom, now: datetime) -> dict:
        """Soonest slots for an emergency: within 2 hours, else later today, else the next two days.

        One calendar request covers all three. When nothing is open today, the urgent callback is
        flagged here, in code, so it never depends on the LLM remembering to do it.
        """
        self.emergency_seen = True
        event_type = self.tenant.emergency_event_type_id or self.tenant.cal_com_event_type_id
        slots = await self._get_slots(event_type, now, now + _EMERGENCY_SEARCH)
        slots = self._open_hours(s.astimezone(self.tz) for s in slots if s >= now)

        within_two_hours = [s for s in slots if s <= now + _EMERGENCY_WINDOW]
        later_today = [s for s in slots if s.date() == now.date()]
        if within_two_hours:
            result, window = self._offer(within_two_hours[:2]), "within_two_hours"
        elif later_today:
            result, window = self._offer(later_today[:2]), "later_today"
        elif slots:
            result, window = self._offer(slots[:2]), "next_available_not_today"
        else:
            result, window = {"options": []}, "none"

        result["window"] = window
        if window in ("next_available_not_today", "none"):
            self._flag_urgent("Emergency reported; no same-day slot available")
            result["urgent_callback"] = "flagged"
            result["say"] = (
                "No same-day opening. Tell them you marked this urgent and a team member from the practice "
                "will call them back very shortly to fit them in; also offer the earliest listed option."
                if slots else
                "No opening in the calendar. Tell them you marked this urgent and a team member from the "
                "practice will call them back very shortly to fit them in."
            )
        return result

    def _open_hours(self, slots) -> list[datetime]:
        return [s for s in slots if slot_in_hours(self.tenant.knowledge_base, s)]

    def _offer(self, slots: list[datetime], none_note: str = "no_slots") -> dict:
        slots = self._open_hours(slots)  # belt and braces: nothing outside opening hours is ever offered
        if not slots:
            return {"options": [], "note": none_note}
        options = []
        for s in slots:
            local = s.astimezone(self.tz)
            sid = _slot_id(local)
            self.offered_slots[sid] = local
            options.append({"slot_id": sid, "spoken": spoken_time(local)})
        return {"options": options}

    async def book_appointment(
        self,
        patient_name: str,
        phone_number: str,
        visit_reason: str,
        is_new_patient: bool,
        is_emergency: bool,
        slot_id: str,
    ) -> dict:
        calcom = self._require_calendar()
        slot = self.offered_slots.get(slot_id)
        if slot is None:
            return {"error": "unknown_slot", "say": "Call check_availability and offer real options first."}
        if not self._name_is_from_caller(patient_name) and not (
            (patient_name or "").strip().lower() not in _PLACEHOLDER_NAMES and self._insist("name", patient_name)
        ):
            return {"error": "name_not_confirmed", "say": "Ask the caller, once, to say their full name, then book with exactly what they say. Never guess or invent a name."}
        if not is_emergency and not self._caller_confirmed():
            return {"error": "confirmation_required", "say": (
                "The caller has not confirmed yet. In one sentence read back their name, the day and time, and the reason, "
                "then ask \"Is that correct?\". Call book_appointment only after they say yes.")}
        if not is_emergency and not self._number_was_asked():
            return {"error": "phone_not_confirmed", "say": (
                "You have not asked about their number yet. Ask once: \"Is the number you're calling from the best one to "
                "reach you?\" (do not read the digits aloud), and book with the number they give or the caller ID.")}
        phone = normalize_phone(phone_number) or self.caller_phone
        if not phone:
            return {"error": "missing_phone", "say": "Ask for the best contact number."}
        phone = self._snap_to_caller_id(phone)
        if not self._phone_is_from_caller(phone) and not self._insist("phone", phone):
            return {"error": "phone_not_confirmed", "say": "Offer the caller ID as their number, or ask for the best number. Never guess one."}

        event_type = (
            self.tenant.emergency_event_type_id if is_emergency and self.tenant.emergency_event_type_id
            else self.tenant.cal_com_event_type_id
        )

        # Shielded so a caller interruption can't cancel us between the Cal.com
        # write and handing our own record to the background writer.
        async def write():
            booking = await calcom.create_booking(
                event_type,
                slot.astimezone(timezone.utc),
                patient_name,
                phone,
                self.tenant.timezone,
                metadata={"visit_reason": visit_reason[:500], "source": "boltix-voice", "call_id": self.call_id},
            )
            row = {
                "client_id": self.tenant.id,
                "call_log_id": self.call_id,
                "cal_booking_uid": booking["uid"],
                "patient_name": patient_name,
                "patient_phone": phone,
                "visit_reason": visit_reason,
                "is_new_patient": is_new_patient,
                "is_emergency": is_emergency,
                "start_time": slot.astimezone(timezone.utc).isoformat(),
            }
            # The calendar is the source of truth and is already updated; our own record
            # is written behind the reply and flushed before anything reads it back.
            self._background(lambda: db.insert_appointment(row), f"appointment {booking['uid']}")
            # Email the practice. Scheduled, never awaited: the caller's reply does not wait for it, and nothing
            # in here may turn a booking that already exists in the calendar into an error for the caller.
            try:
                notify.notify_booking(
                    notify.BookingEmail(
                        practice=self.tenant.name,
                        patient_name=patient_name,
                        patient_phone=phone,
                        reason=visit_reason,
                        start=slot,
                        timezone_name=self.tenant.timezone,
                        is_emergency=is_emergency,
                        is_new_patient=is_new_patient,
                        booking_uid=booking["uid"],
                    ),
                    self.tenant.knowledge_base,
                )
            except Exception:
                logger.exception(f"[{self.call_id}] booking email could not be scheduled")

        self._drop_prefetch()  # the calendar is about to change: later searches must see it fresh
        try:
            await asyncio.shield(write())
        except CalComError as e:
            logger.warning(f"[{self.call_id}] booking rejected: {e}")
            self.offered_slots.pop(slot_id, None)
            return {"error": "slot_no_longer_available", "say": "Apologize, then check availability again."}

        self.appointment_booked = True
        self.writes += 1
        return {"status": "booked", "time": spoken_time(slot), "patient_name": patient_name}

    async def find_appointments(self, phone_number: str | None = None, patient_name: str | None = None) -> dict:
        await self.flush()
        phone = normalize_phone(phone_number) if phone_number else (None if patient_name else self.caller_phone)
        rows = await db.find_upcoming_appointments(
            self.tenant.id, phone, patient_name, datetime.now(timezone.utc).isoformat()
        )
        if not rows:
            return {"appointments": [], "note": "none_found"}
        appointments = []
        for r in rows:
            self.found_appointments.add(r["id"])
            self._appt_cache[r["id"]] = {**r, "status": "booked"}  # find_appointments never returns cancelled rows
            appointments.append(
                {
                    "appointment_id": r["id"],
                    "patient_name": r["patient_name"],
                    "time": spoken_time(datetime.fromisoformat(r["start_time"]).astimezone(self.tz)),
                    "visit_reason": r["visit_reason"],
                }
            )
        return {"appointments": appointments}

    async def _load_verified(self, appointment_id: str, phone_last4: str | None) -> tuple[dict | None, dict | None]:
        if appointment_id not in self.found_appointments:
            return None, {"error": "unknown_appointment", "say": "Look the appointment up with find_appointments first."}
        appt = self._appt_cache.get(appointment_id)
        if appt is None:  # not looked up through the cache (should not happen): fall back to the database
            await self.flush()
            appt = await db.get_appointment(self.tenant.id, appointment_id)
        if not appt or appt["status"] == "cancelled":
            return None, {"error": "appointment_not_active"}
        if appt["patient_phone"] != self.caller_phone:
            given = re.sub(r"\D", "", phone_last4 or "")
            if len(given) < 4 or not appt["patient_phone"].endswith(given[-4:]):
                return None, {
                    "error": "verification_required",
                    "say": "Ask for the last four digits of the phone number on the appointment.",
                }
        return appt, None

    async def reschedule_appointment(self, appointment_id: str, slot_id: str, phone_last4: str | None = None) -> dict:
        calcom = self._require_calendar()
        appt, error = await self._load_verified(appointment_id, phone_last4)
        if error:
            return error
        slot = self.offered_slots.get(slot_id)
        if slot is None:
            return {"error": "unknown_slot", "say": "Call check_availability and offer real options first."}

        async def write():
            booking = await calcom.reschedule_booking(appt["cal_booking_uid"], slot.astimezone(timezone.utc))
            changes = {
                "cal_booking_uid": booking.get("uid", appt["cal_booking_uid"]),
                "start_time": slot.astimezone(timezone.utc).isoformat(),
                "status": "rescheduled",
            }
            self._appt_cache.setdefault(appointment_id, appt).update(changes)  # a later change in this call sees it
            # The calendar is already updated; our own record follows behind the reply.
            self._background(lambda: db.update_appointment(self.tenant.id, appointment_id, changes), f"reschedule {appointment_id}")

        self._drop_prefetch()
        try:
            await asyncio.shield(write())
        except CalComError as e:
            logger.warning(f"[{self.call_id}] reschedule rejected: {e}")
            self.offered_slots.pop(slot_id, None)
            return {"error": "slot_no_longer_available", "say": "Apologize, then check availability again."}
        self.writes += 1
        return {"status": "rescheduled", "time": spoken_time(slot)}

    async def cancel_appointment(self, appointment_id: str, reason: str, phone_last4: str | None = None) -> dict:
        calcom = self._require_calendar()
        appt, error = await self._load_verified(appointment_id, phone_last4)
        if error:
            return error

        async def write():
            await calcom.cancel_booking(appt["cal_booking_uid"], reason[:500] or "Cancelled by patient")
            self._appt_cache.setdefault(appointment_id, appt)["status"] = "cancelled"
            self._background(
                lambda: db.update_appointment(self.tenant.id, appointment_id, {"status": "cancelled"}), f"cancel {appointment_id}"
            )

        self._drop_prefetch()
        await asyncio.shield(write())
        self.writes += 1
        return {"status": "cancelled"}

    async def set_language(self, language: str) -> dict:
        code = (language or "en").lower()
        if code not in language_module.SUPPORTED:
            return {"error": "unsupported_language", "say": "Apologize briefly and continue in English."}
        if self.apply_language and code != self.language_state.current:
            await self.apply_language(code)
        else:
            self.language_state.current = code
        return {"status": "ok", "language": code, "say": "Reply in this language from now on."}

    async def request_boltix_callback(
        self,
        name: str,
        practice_name: str,
        phone_number: str,
        best_time: str | None = None,
        notes: str | None = None,
    ) -> dict:
        """Demo line: a practice manager asks the Boltix AI team to call them. Same guardrails as a booking: the name must
        be one the caller said, the number the caller ID or one they said, and the caller must have said yes to the read-back."""
        if not self.tenant.demo_mode:
            return {"error": "not_available"}
        if self.callback is not None:
            return {"status": "saved", "say": "Already saved. Tell them: The Boltix AI team will give you a call back. Never say shortly or soon."}
        name, practice = (name or "").strip(), (practice_name or "").strip()
        if not practice or practice.lower() in _PLACEHOLDER_NAMES:
            return {"error": "practice_not_given", "say": "Ask the caller, once, for the name of their practice."}
        if not self._name_is_from_caller(name) and not (
            name.lower() not in _PLACEHOLDER_NAMES and self._insist("name", name)
        ):
            return {"error": "name_not_confirmed", "say": "Ask the caller, once, to say their name, then use exactly what they say. Never guess or invent a name."}
        if not self._caller_confirmed():
            return {"error": "confirmation_required", "say": (
                "The caller has not confirmed yet. In one sentence read back their name, practice, and number, then ask "
                "\"Is that correct?\". Call this tool only after they say yes.")}
        if not self._number_was_asked():
            return {"error": "phone_not_confirmed", "say": (
                "You have not asked about their number yet. Ask once: \"Is the number you're calling from the best one to "
                "reach you?\" (do not read the digits aloud), and use the number they give or the caller ID.")}
        phone = normalize_phone(phone_number) or self.caller_phone
        if not phone:
            return {"error": "missing_phone", "say": "Ask for the best number to reach them."}
        phone = self._snap_to_caller_id(phone)
        if not self._phone_is_from_caller(phone) and not self._insist("phone", phone):
            return {"error": "phone_not_confirmed", "say": "Offer the caller ID as their number, or ask for the best number. Never guess one."}

        lead = {
            "call_log_id": self.call_id,
            "name": name[:200],
            "practice_name": practice[:200],
            "phone": phone,
            "best_time": (best_time or "").strip()[:200] or None,
            "notes": (notes or "").strip()[:500] or None,
        }
        self.callback = lead
        self.writes += 1  # counts as a saved write, so "the team will call you" is never flagged as an unsaved claim
        self._background(lambda: db.insert_demo_lead(lead), "demo lead")
        try:  # an email to Boltix AI: scheduled, never awaited, and never able to fail the tool
            notify.notify_demo_lead(
                notify.DemoLeadEmail(
                    name=lead["name"], practice_name=lead["practice_name"], phone=phone,
                    best_time=lead["best_time"] or "", notes=lead["notes"] or "", call_id=self.call_id,
                )
            )
        except Exception:
            logger.exception(f"[{self.call_id}] callback email could not be scheduled")
        return {"status": "saved", "say": "Tell them: The Boltix AI team will give you a call back. Never say shortly or soon, and give no time."}

    async def flag_urgent_callback(self, reason: str) -> dict:
        self._flag_urgent(reason)
        return {"status": "flagged", "say": "Tell them a team member will call back as soon as possible."}

    async def end_call(self, params: FunctionCallParams) -> None:
        farewell = str(params.arguments.get("farewell") or "Thanks for calling. Have a great day!")
        await params.result_callback({"status": "ending"}, properties=FunctionCallResultProperties(run_llm=False))
        # EndWorkerFrame queued behind the farewell, so the goodbye plays fully
        # before the pipeline ends and Telnyx hangs up.
        await params.llm.push_frame(TTSSpeakFrame(farewell))
        await params.llm.push_frame(EndWorkerFrame(reason="caller_finished"))

    # ---------------------------------------------------------------- helpers

    def transcript_entry(self, role: str, content: str) -> dict[str, Any]:
        return {"role": role, "content": content, "at": datetime.now(timezone.utc).isoformat()}
