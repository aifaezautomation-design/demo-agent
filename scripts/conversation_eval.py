"""Text-only conversation eval: simulated patients talk to Mia's REAL prompt, tools and guards.

No audio, no real Cal.com or database (an in-memory calendar and appointment table stand in), so a run costs a few
cents of OpenAI and finishes in a minute. It exists to measure how often a conversation goes wrong, e.g. Mia promising
"let me check" and then never checking, before a caller finds out on the phone.

    python scripts/conversation_eval.py            # every scenario once
    python scripts/conversation_eval.py 4          # every scenario four times
    python scripts/conversation_eval.py 3 book_    # only scenarios whose name starts with book_
    python scripts/conversation_eval.py 1 -v       # also print every transcript, not just the failures

Each scenario states what the caller wants and what the finished call must look like (which slot got booked, for whom,
or that nothing may be booked). A run fails when the outcome is wrong, when a turn ended in silence, or when Mia broke
a promise that even the production re-prompt could not repair.
"""
import asyncio
import contextvars
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RESEND_API_KEY", "")
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ["RESEND_API_KEY"] = ""
from openai import AsyncOpenAI

from app import config, tools
from app.db import Tenant
from app.guards import NOTES, PromiseGuard
from app.prompts import build_system_prompt
from app.tools import CallSession

TZ = ZoneInfo("America/Chicago")
CALLER_ID = "+15128802093"
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
GREETING = "Thanks for calling Apex Dental Care, this is Mia. How can I help you today?"


class Calendar:
    """Mon-Sat 8 AM-6 PM Central, hourly, for the next three weeks (Cal.com's own hours are Mon-Sat 9-8 Eastern)."""

    def __init__(self):
        now = datetime.now(TZ)
        self.slots = []
        for d in range(0, 21):
            day = (now + timedelta(days=d)).replace(minute=0, second=0, microsecond=0)
            if day.weekday() == 6:
                continue
            self.slots += [day.replace(hour=h) for h in range(8, 19) if day.replace(hour=h) > now + timedelta(hours=2)]
        self.bookings = []
        self.cancelled = []

    async def get_slots(self, event_type_id, start, end, time_zone):
        taken = {b[0] for b in self.bookings}
        return [s for s in self.slots if start <= s < end and s not in taken]

    async def create_booking(self, event_type_id, start_utc, name, phone, time_zone, metadata):
        self.bookings.append((start_utc.astimezone(TZ), name, phone))
        return {"uid": f"uid-{len(self.bookings)}"}

    async def reschedule_booking(self, uid, new_start_utc):
        return {"uid": uid + "-r"}

    async def cancel_booking(self, uid, reason):
        self.cancelled.append(uid)


class Table:
    def __init__(self):
        self.rows = {}

    async def insert_appointment(self, row):
        row = {**row, "id": f"appt-{len(self.rows) + 1}", "status": "booked"}
        self.rows[row["id"]] = row
        return row

    async def find_upcoming_appointments(self, client_id, phone, name, now_iso):
        return [a for a in self.rows.values() if a["status"] == "booked"
                and ((phone and a["patient_phone"] == phone) or (name and name.lower() in a["patient_name"].lower()))]

    async def get_appointment(self, client_id, appointment_id):
        return self.rows.get(appointment_id)

    async def update_appointment(self, client_id, appointment_id, fields):
        self.rows[appointment_id].update(fields)

    async def flag_urgent_callback(self, call_id, reason):
        pass


_table: contextvars.ContextVar = contextvars.ContextVar("table")


class _DbProxy:
    """tools.db is module-global; each concurrent conversation reaches its own table through its own task context."""

    def __getattr__(self, name):
        return getattr(_table.get(), name)


def _next(weekday: int, weeks_ahead: int = 0):
    today = datetime.now(TZ).date()
    d = today + timedelta(days=(weekday - today.weekday()) % 7)
    return d if weeks_ahead == 0 else d + timedelta(days=7 * weeks_ahead)


def tomorrow():
    d = datetime.now(TZ).date() + timedelta(days=1)
    return d if d.weekday() != 6 else d + timedelta(days=1)


# name, what the patient wants (told to the simulator), expected outcome check
def scenarios():
    tm = tomorrow()
    fri = _next(4)
    return {
        "book_all_upfront": dict(
            want=f"You are a NEW patient who wants a routine cleaning tomorrow ({tm:%A}) at 2 PM. Your name is Maria Lopez; the number you are calling from is fine. Give the day, time and reason in your first sentence.",
            expect=dict(day=tm, hour=14, name="Maria")),
        "book_step_by_step": dict(
            want=f"You are a NEW patient. First say only that you would like to book an appointment. Answer each question briefly and only what is asked. You want a checkup this Friday ({fri:%B %d}) in the morning, and you take the first time offered. Name: Joy Smith. The calling number is fine.",
            expect=dict(day=fri, name="Joy", morning=True)),
        "book_day_first_vague_time": dict(
            want="You are a NEW patient with a sensitive back tooth who wants a checkup. In your first sentence ask if there are any open spots this Friday. You ONLY want an afternoon time: if Mia offers morning times, say you need the afternoon and wait for afternoon times, then take the earliest one. Name: Ahmed Khan. The calling number is fine.",
            expect=dict(day=fri, name="Ahmed", afternoon=True)),
        "book_returning_next_week": dict(
            want="You are a RETURNING patient who needs a cleaning sometime next Tuesday. You have no time preference and take the first time offered. Name: Priya Patel. The calling number is fine.",
            expect=dict(day=_next(1) if _next(1) > datetime.now(TZ).date() else _next(1, 1), name="Priya", next_week=True)),
        "book_change_mind": dict(
            want=f"You are a NEW patient who wants a filling tomorrow ({tm:%A}). When Mia offers times, pick the first one. Then, before confirming, say you changed your mind and want 3 PM instead. Name: Tom Baker. The calling number is fine. Confirm the final time.",
            expect=dict(day=tm, hour=15, name="Tom")),
        "book_different_phone": dict(
            want=f"You are a NEW patient who wants a cleaning tomorrow ({tm:%A}) at 10 AM. Name: Sara Nguyen. When asked about the number, say no, use 5 1 2, 5 5 5, 0 1 4 7 instead.",
            expect=dict(day=tm, hour=10, name="Sara", phone="+15125550147")),
        "faq_then_book": dict(
            want=f"First ask what your insurance options are (you have Delta Dental) and where the office is. Then say you are a NEW patient and want a cleaning tomorrow ({tm:%A}) at 11 AM. Name: Linda Cruz. The calling number is fine.",
            expect=dict(day=tm, hour=11, name="Linda")),
        "book_closed_day": dict(
            want="You are a NEW patient who wants a cleaning on Sunday. When told the office is closed, ask for the next Monday morning and take the first time offered. Name: Greg Hall. The calling number is fine.",
            expect=dict(day=_next(0) if _next(0) > datetime.now(TZ).date() else _next(0, 1), name="Greg", morning=True)),
        "book_name_correction": dict(
            want=f"You are a NEW patient who wants a cleaning tomorrow ({tm:%A}) at 9 AM. Your name is Jon Smith. At the read-back, if the name Mia says is not exactly Jon Smith, correct her. Say yes only when it is right. The calling number is fine.",
            expect=dict(day=tm, hour=9, name="Jon")),
        "book_question_midway": dict(
            want=f"You are a NEW patient who wants a cleaning tomorrow ({tm:%A}) at 1 PM. After Mia offers the time and before you give your name, ask whether they take Cigna insurance, then continue. Name: Dana Wu. The calling number is fine.",
            expect=dict(day=tm, hour=13, name="Dana")),
        "book_earliest": dict(
            want=f"You are a NEW patient who wants the very earliest opening tomorrow ({tm:%A}) for a checkup. Take whatever the earliest time is. Name: Omar Ali. The calling number is fine.",
            expect=dict(day=tm, hour=8, name="Omar")),
        "book_switch_day": dict(
            want=f"You are a NEW patient who first asks about tomorrow ({tm:%A}) at 9 AM for a cleaning. After Mia offers it, say actually you need Saturday instead, any morning time offered first. Name: Eve Rossi. The calling number is fine.",
            expect=dict(day=_next(5) if _next(5) > tm else _next(5, 1), name="Eve", morning=True)),
        "browse_no_book": dict(
            want=f"You only want to know what times are free tomorrow ({tm:%A}) morning and then say you will call back later. Do NOT book anything. Say goodbye.",
            expect=dict(nothing_booked=True)),
        "faq_only": dict(
            want="You only want to know the Saturday hours, whether they do whitening, and the price of a root canal. Then say thanks, goodbye.",
            expect=dict(nothing_booked=True)),
        "cancel": dict(
            want="You are Maria Lopez with an existing appointment. You want to cancel it. The number you are calling from is the one on the appointment. If asked for the last four digits of the phone, say 2093. Confirm the cancellation.",
            seed=True, expect=dict(cancelled=True)),
        "reschedule": dict(
            want="You are Maria Lopez with an existing appointment. You want to move it to next Monday in the morning, first time offered. The calling number is the one on the appointment; if asked for the last four digits say 2093. Confirm the change.",
            seed=True, expect=dict(rescheduled=True)),
    }


PATIENT_RULES = """You are role-playing a caller on a phone call with a dental receptionist. Speak like a real person on the phone:
short, casual, sometimes a bit rambling, 1-2 short sentences, no lists. Answer only what is asked unless your goal says otherwise.
Never invent facts beyond your goal. Do not narrate. When your goal is complete (or the receptionist says goodbye), reply exactly: [HANGUP]
Your goal: """


async def patient_says(goal: str, transcript: list[tuple[str, str]]) -> str:
    msgs = [{"role": "system", "content": PATIENT_RULES + goal}]
    for who, text in transcript:
        msgs.append({"role": "user" if who == "mia" else "assistant", "content": text})
    r = await client.chat.completions.create(model="gpt-4o-mini", temperature=0.7, max_tokens=60, messages=msgs)
    return (r.choices[0].message.content or "").strip()


async def run_one(name: str, sc: dict) -> dict:
    cal, table = Calendar(), Table()
    _table.set(table)
    tenant = Tenant(
        id="apex-dental", name="Apex Dental Care", phone_number="+15126411185", timezone="America/Chicago",
        cal_com_event_type_id=1, emergency_event_type_id=None, cal_com_api_key="x",
        knowledge_base=json.loads((ROOT / "data" / "apex_dental_care.json").read_text(encoding="utf-8")), demo_mode=False,
    )
    session = CallSession(tenant, "eval", CALLER_ID)
    session.calcom = cal
    if sc.get("seed"):
        d = tomorrow()
        slot = datetime(d.year, d.month, d.day, 13, tzinfo=TZ)
        cal.bookings.append((slot, "Maria Lopez", CALLER_ID))
        await table.insert_appointment(dict(client_id="apex-dental", patient_name="Maria Lopez", patient_phone=CALLER_ID,
                                            start_time=slot.astimezone(ZoneInfo("UTC")).isoformat(), cal_booking_uid="uid-0",
                                            visit_reason="cleaning", is_emergency=False))
    handlers = {"check_availability": session.check_availability, "book_appointment": session.book_appointment,
                "find_appointments": session.find_appointments, "reschedule_appointment": session.reschedule_appointment,
                "cancel_appointment": session.cancel_appointment, "flag_urgent_callback": session.flag_urgent_callback}
    tool_schema = [{"type": "function", "function": f.to_default_dict()} for f in session.tools_schema().standard_tools]
    messages = [{"role": "system", "content": build_system_prompt(tenant, CALLER_ID)}, {"role": "assistant", "content": GREETING}]
    heard: list[tuple[str, str]] = [("mia", GREETING)]
    log: list[str] = [f"MIA: {GREETING}"]
    problems, raw_broken, repaired = [], 0, 0
    guard = PromiseGuard()
    ended = False

    async def llm_turn(extra_system: str | None = None):
        msgs = messages + ([{"role": "system", "content": extra_system}] if extra_system else [])
        r = await client.chat.completions.create(model=config.OPENAI_MODEL, temperature=config.OPENAI_TEMPERATURE,
                                                 max_tokens=config.OPENAI_MAX_TOKENS, tools=tool_schema, messages=msgs)
        return r.choices[0].message

    for _ in range(18):  # caller turns
        said = await patient_says(sc["want"], heard)
        if "[HANGUP]" in said:
            break
        log.append(f"CALLER: {said}")
        heard.append(("patient", said))
        messages.append({"role": "user", "content": said})
        session.note_heard(said)

        nudged = nudged_stall = False
        spoken_this_turn = []
        for _hop in range(8):  # a turn may chain tool calls
            m = await llm_turn()
            text, calls = (m.content or "").strip(), m.tool_calls or []
            if text:
                session.note_said(text)  # production records every spoken reply the moment it ends
            kind = guard.check(text, bool(calls), session.writes, session.turn_searched)
            if calls:
                messages.append({"role": "assistant", "content": m.content, "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
                if text:
                    spoken_this_turn.append(text)
                    log.append(f"MIA: {text}")
                for c in calls:
                    args = json.loads(c.function.arguments or "{}")
                    log.append(f"  [tool {c.function.name} {json.dumps(args)[:110]}]")
                    if c.function.name == "end_call":
                        ended = True
                        farewell = args.get("farewell", "Goodbye!")
                        spoken_this_turn.append(farewell)
                        log.append(f"MIA: {farewell}")
                        messages.append({"role": "tool", "tool_call_id": c.id, "content": '{"status":"ending"}'})
                        continue
                    try:
                        res = await handlers[c.function.name](**args)
                    except Exception as e:  # tool errors reach the model the way the production wrapper reports them
                        res = {"error": "internal_error", "say": "Apologize and offer to have the team call back."}
                        log.append(f"  [tool error {type(e).__name__}: {e}]")
                    if isinstance(res, dict) and res.get("error"):
                        log.append(f"  [tool said {res['error']}]")
                    messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res)})
                if ended:
                    break
                continue
            # a spoken reply
            if not text:
                if not nudged_stall:
                    nudged_stall = True
                    log.append("  [SILENT reply -> production re-prompts]")
                    messages.append({"role": "system", "content": NOTES["stall"]})
                    continue
                problems.append("a turn ended in silence even after the re-prompt")
                break
            spoken_this_turn.append(text)
            log.append(f"MIA: {text}")
            messages.append({"role": "assistant", "content": text})
            if kind:
                raw_broken += 1
                log.append(f"  [GUARD {kind}: {text!r}]")
                if not nudged:
                    nudged = True
                    messages.append({"role": "system", "content": NOTES[kind]})
                    continue  # production re-prompts the model once per caller turn
                problems.append(f"{kind} not repaired: {text!r}")
            break
        nudged_promise = nudged
        if nudged_promise and not problems:
            repaired += 1
        for t in spoken_this_turn:
            heard.append(("mia", t))
        if ended:
            break
    else:
        problems.append("call did not finish in 18 turns")

    exp = sc["expect"]
    b = cal.bookings
    if exp.get("nothing_booked") and b:
        problems.append(f"booked something it should not have: {b}")
    if "day" in exp:
        new = [x for x in b if not sc.get("seed")]
        if len(new) != 1:
            problems.append(f"expected exactly 1 booking, got {len(new)}: {new}")
        else:
            start, who, phone = new[0]
            if start.date() != exp["day"]:
                problems.append(f"wrong day: {start:%a %m-%d %H:%M} wanted {exp['day']}")
            if "hour" in exp and start.hour != exp["hour"]:
                problems.append(f"wrong hour: {start.hour} wanted {exp['hour']}")
            if exp.get("morning") and start.hour >= 12:
                problems.append(f"wanted morning, got {start:%H:%M}")
            if exp.get("afternoon") and start.hour < 12:
                problems.append(f"wanted afternoon, got {start:%H:%M}")
            if exp["name"].lower() not in who.lower():
                problems.append(f"wrong name {who!r}")
            if phone != exp.get("phone", CALLER_ID):
                problems.append(f"wrong phone {phone}")
    if exp.get("cancelled") and not cal.cancelled:
        problems.append("appointment was not cancelled")
    if exp.get("rescheduled") and session.writes == 0:
        problems.append("appointment was not rescheduled")
    return dict(name=name, problems=problems, raw_broken=raw_broken, repaired=repaired, log=log)


async def main():
    tools.db = _DbProxy()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    reps = int(args[0]) if args else 1
    prefix = args[1] if len(args) > 1 else ""
    todo = [(n, s) for n, s in scenarios().items() if n.startswith(prefix)] * reps
    results = await asyncio.gather(*[run_one(n, s) for n, s in todo])
    bad = 0
    by = {}
    for r in results:
        by.setdefault(r["name"], []).append(r)
    for name, rs in by.items():
        fails = [r for r in rs if r["problems"]]
        bad += len(fails)
        broke = sum(r["raw_broken"] for r in rs)
        print(f"{'PASS' if not fails else 'FAIL'}  {name:28} {len(rs) - len(fails)}/{len(rs)} ok   guard firings: {broke}")
    for r in results:
        if r["problems"] or verbose:
            print(f"\n--- {r['name']} {'FAILED: ' + '; '.join(r['problems']) if r['problems'] else 'ok'}")
            print("\n".join(r["log"]))
    print(f"\n{len(results) - bad}/{len(results)} conversations ended correctly; "
          f"{sum(r['raw_broken'] for r in results)} guard firings, {sum(r['repaired'] for r in results)} repaired by the re-prompt")


asyncio.run(main())
