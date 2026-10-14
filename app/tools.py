"""LLM tools for one call. State lives on CallSession so handlers can enforce
guardrails in code: only slots offered this call can be booked, and only
appointments looked up this call can be changed."""

import asyncio
import re
import time as time_module
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndWorkerFrame, FunctionCallResultProperties, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams, LLMService

from app import db
from app.calcom import CalCom, CalComError
from app.db import Tenant

_MAX_OPTIONS = 4  # returned to the LLM; the prompt tells it to offer two
_SEARCH_DAYS = 7
_EMERGENCY_WINDOW = timedelta(hours=2)
_EMERGENCY_SEARCH = timedelta(hours=48)
_MIN_LEAD_TIME = timedelta(minutes=30)


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
    hour = dt.hour % 12 or 12
    minutes = "" if dt.minute == 0 else f":{dt.minute:02d}"
    return f"{dt:%A, %B} {dt.day} at {hour}{minutes} {'AM' if dt.hour < 12 else 'PM'}"


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
        # The call_logs insert runs in the background so the greeting isn't delayed;
        # anything that references it (appointments FK, billing) awaits this first.
        self._call_log_ready = call_log_ready
        self._pending: set[asyncio.Task] = set()
        self.caller_phone = normalize_phone(caller_phone)
        self.tz = ZoneInfo(tenant.timezone)
        self.calcom = CalCom(tenant.cal_com_api_key) if tenant.cal_com_api_key else None
        self.offered_slots: dict[str, datetime] = {}
        self.found_appointments: set[str] = set()
        self.appointment_booked = False

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
                        "date": {"type": "string", "description": "YYYY-MM-DD; today if no preference."},
                        "time_of_day": {"type": "string", "enum": ["morning", "afternoon", "evening", "any"]},
                        "is_emergency": {"type": "boolean", "description": "True: search only the next two hours."},
                    },
                    required=["date", "time_of_day", "is_emergency"],
                ),
                FunctionSchema(
                    name="book_appointment",
                    description="Book a slot after the caller confirmed the read-back.",
                    properties={
                        "patient_name": {"type": "string"},
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
                FunctionSchema(
                    name="end_call",
                    description="Say goodbye and hang up.",
                    properties={"farewell": {"type": "string", "description": "One short goodbye."}},
                    required=["farewell"],
                ),
            ]
        )

    def register(self, llm: LLMService) -> None:
        handlers = {
            "check_availability": self.check_availability,
            "book_appointment": self.book_appointment,
            "find_appointments": self.find_appointments,
            "reschedule_appointment": self.reschedule_appointment,
            "cancel_appointment": self.cancel_appointment,
            "flag_urgent_callback": self.flag_urgent_callback,
        }
        for name, handler in handlers.items():
            llm.register_function(name, self._wrap(name, handler), timeout_secs=12)
        llm.register_function("end_call", self.end_call)

    def _wrap(self, name: str, handler):
        # Errors go back to the LLM as data so Mia can recover in conversation
        # instead of the pipeline logging a stack trace and going silent.
        async def run(params: FunctionCallParams):
            started = time_module.monotonic()
            try:
                result = await handler(**dict(params.arguments))
            except CalComError as e:
                logger.warning(f"[{self.call_id}] {name} Cal.com failure: {e}")
                result = {"error": "calendar_unavailable", "say": "Apologize and offer to have the team call back."}
            except Exception:
                logger.exception(f"[{self.call_id}] {name} failed")
                result = {"error": "internal_error", "say": "Apologize and offer to have the team call back."}
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

    async def flush(self) -> None:
        """Waits for background writes. Called before reading them back and at call end."""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    def _require_calendar(self) -> CalCom:
        if not self.calcom:
            raise CalComError("tenant has no Cal.com API key configured")
        return self.calcom

    # ------------------------------------------------------------------ tools

    async def check_availability(self, date: str, time_of_day: str = "any", is_emergency: bool = False) -> dict:
        calcom = self._require_calendar()
        now = datetime.now(self.tz)

        if is_emergency:
            return await self._emergency_slots(calcom, now)

        try:
            day = datetime.strptime(date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            day = now.date()
        day = max(day, now.date())

        start = max(datetime.combine(day, time.min, self.tz), now + _MIN_LEAD_TIME)
        end = datetime.combine(day + timedelta(days=_SEARCH_DAYS), time.min, self.tz)
        slots = await calcom.get_slots(self.tenant.cal_com_event_type_id, start, end, self.tenant.timezone)
        slots = [s.astimezone(self.tz) for s in slots if _matches_time_of_day(s.astimezone(self.tz), time_of_day)]

        on_requested_day = [s for s in slots if s.date() == day]
        if on_requested_day:
            return self._offer(on_requested_day[:_MAX_OPTIONS])
        result = self._offer(slots[:_MAX_OPTIONS], none_note=f"nothing_open_in_the_next_{_SEARCH_DAYS}_days")
        if slots:
            result["note"] = "Nothing open on the requested day; these are the nearest alternatives."
        return result

    async def _emergency_slots(self, calcom: CalCom, now: datetime) -> dict:
        """Soonest slots for an emergency: within 2 hours, else later today, else the next two days.

        One calendar request covers all three. When nothing is open today, the urgent callback is
        flagged here, in code, so it never depends on the LLM remembering to do it.
        """
        event_type = self.tenant.emergency_event_type_id or self.tenant.cal_com_event_type_id
        slots = await calcom.get_slots(event_type, now, now + _EMERGENCY_SEARCH, self.tenant.timezone)
        slots = [s.astimezone(self.tz) for s in slots if s >= now]

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
            await db.flag_urgent_callback(self.call_id, "Emergency reported; no same-day slot available")
            result["urgent_callback"] = "flagged"
            result["say"] = (
                "No same-day opening. Tell them you marked this urgent and a team member from the practice "
                "will call them back very shortly to fit them in; also offer the earliest listed option."
                if slots else
                "No opening in the calendar. Tell them you marked this urgent and a team member from the "
                "practice will call them back very shortly to fit them in."
            )
        return result

    def _offer(self, slots: list[datetime], none_note: str = "no_slots") -> dict:
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
        phone = normalize_phone(phone_number) or self.caller_phone
        if not phone:
            return {"error": "missing_phone", "say": "Ask for the best contact number."}

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
                metadata={"visit_reason": visit_reason[:500], "source": "boltex-voice", "call_id": self.call_id},
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

        try:
            await asyncio.shield(write())
        except CalComError as e:
            logger.warning(f"[{self.call_id}] booking rejected: {e}")
            self.offered_slots.pop(slot_id, None)
            return {"error": "slot_no_longer_available", "say": "Apologize, then check availability again."}

        self.appointment_booked = True
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
        await self.flush()
        if appointment_id not in self.found_appointments:
            return None, {"error": "unknown_appointment", "say": "Look the appointment up with find_appointments first."}
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
            await db.update_appointment(
                self.tenant.id,
                appointment_id,
                {
                    "cal_booking_uid": booking.get("uid", appt["cal_booking_uid"]),
                    "start_time": slot.astimezone(timezone.utc).isoformat(),
                    "status": "rescheduled",
                },
            )

        try:
            await asyncio.shield(write())
        except CalComError as e:
            logger.warning(f"[{self.call_id}] reschedule rejected: {e}")
            self.offered_slots.pop(slot_id, None)
            return {"error": "slot_no_longer_available", "say": "Apologize, then check availability again."}
        return {"status": "rescheduled", "time": spoken_time(slot)}

    async def cancel_appointment(self, appointment_id: str, reason: str, phone_last4: str | None = None) -> dict:
        calcom = self._require_calendar()
        appt, error = await self._load_verified(appointment_id, phone_last4)
        if error:
            return error

        async def write():
            await calcom.cancel_booking(appt["cal_booking_uid"], reason[:500] or "Cancelled by patient")
            await db.update_appointment(self.tenant.id, appointment_id, {"status": "cancelled"})

        await asyncio.shield(write())
        return {"status": "cancelled"}

    async def flag_urgent_callback(self, reason: str) -> dict:
        await db.flag_urgent_callback(self.call_id, reason[:500])
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
