"""Local end-to-end check against your real Supabase project and Cal.com account.
No phone call, Telnyx, Deepgram, or Cartesia needed.

It runs the same tool code Mia runs on a live call:

  1. Preflight      schema applied, tenant row and Cal.com key readable
  2. Availability   real Cal.com slots for the tenant's event type
  3. Booking        creates a Cal.com booking plus an `appointments` row
  4. Lookup         finds it again by caller ID
  5. Reschedule     moves it to another offered slot
  6. Cancel         cancels it in Cal.com and marks the row cancelled
  7. Emergency      searches the 2-hour emergency window, flags an urgent callback
  8. Billing        finalize_call deducts minutes exactly once
  9. LLM triage     gpt-4o-mini with Mia's real prompt routes emergencies correctly

By default the test call log is deleted and its minutes refunded at the end.
Pass --keep to leave them in Supabase for inspection.

Usage (from the project root):
  .venv\\Scripts\\python scripts\\simulate_call.py --number +15125550100
  .venv\\Scripts\\python scripts\\simulate_call.py --number +15125550100 --chat
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

# app.config requires the voice/telephony keys at import, but this script never
# uses them. Fill blanks with placeholders so you can test the data layer first.
_PLACEHOLDER = "unused-by-simulator"
for _name in ("PUBLIC_HOST", "WS_AUTH_TOKEN", "TELNYX_API_KEY", "DEEPGRAM_API_KEY",
              "CARTESIA_API_KEY", "CARTESIA_VOICE_ID", "OPENAI_API_KEY"):
    if not os.getenv(_name):
        os.environ[_name] = _PLACEHOLDER
HAS_OPENAI = os.environ["OPENAI_API_KEY"] != _PLACEHOLDER


def _check_supabase_env() -> str | None:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or "xxxx" in url:
        return "SUPABASE_URL is not set in .env"
    if not key:
        return "SUPABASE_SERVICE_ROLE_KEY is not set in .env"
    if key.startswith("sb_publishable_"):
        return "SUPABASE_SERVICE_ROLE_KEY is the publishable key; use the secret (service_role) key"
    if key.count(".") == 2:  # legacy JWT key: check its role claim
        try:
            payload = key.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            if claims.get("role") != "service_role":
                return f"SUPABASE_SERVICE_ROLE_KEY has role '{claims.get('role')}'; use the service_role key"
        except ValueError:
            pass
    return None


# Checked in main(), after argparse, so --help works before .env exists.
_ENV_PROBLEM = _check_supabase_env()
for _name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
    if not os.getenv(_name):
        os.environ[_name] = _PLACEHOLDER

from app import db  # noqa: E402
from app.calcom import CalComError  # noqa: E402
from app.prompts import build_system_prompt  # noqa: E402
from app.tools import CallSession, normalize_phone  # noqa: E402


class Report:
    def __init__(self):
        self.failures = 0

    def ok(self, msg):
        print(f"  [PASS] {msg}")

    def fail(self, msg):
        self.failures += 1
        print(f"  [FAIL] {msg}")

    def skip(self, msg):
        print(f"  [SKIP] {msg}")

    def info(self, msg):
        print(f"         {msg}")

    def check(self, cond, msg, detail=""):
        if cond:
            self.ok(msg)
        else:
            self.fail(f"{msg}{' (' + detail + ')' if detail else ''}")
        return cond


def heading(n, title):
    print(f"\n{n}. {title}")


# --------------------------------------------------------------------- helpers


async def fetch_one(table: str, column: str, value) -> dict | None:
    sb = await db.get_db()
    res = await sb.table(table).select("*").eq(column, value).execute()
    return res.data[0] if res.data else None


async def used_minutes(client_id: str) -> float:
    return float((await fetch_one("clients", "id", client_id))["used_minutes"])


def openai_tools(session: CallSession) -> list[dict]:
    return [{"type": "function", "function": f.to_default_dict()}
            for f in session.tools_schema().standard_tools]


def greeting_for(tenant) -> str:
    return f"Thanks for calling {tenant.name}, this is Mia. How can I help you today?"


async def load_tenant(number: str, r: Report):
    try:
        tenant = await db.get_tenant_by_number(number)
    except Exception as e:
        text = str(e)
        if "does not exist" in text or "PGRST205" in text or "schema cache" in text:
            r.fail("schema.sql has not been applied (tables not found)")
        else:
            r.fail(f"Supabase query failed: {text[:300]}")
        return None
    if tenant is None:
        r.fail(f"no clients row with phone_number = {number}")
        r.info("Check the number is E.164 (e.g. +15125550100) and matches the clients table.")
        return None
    r.ok(f"tenant '{tenant.id}' ({tenant.name}) loaded")
    return tenant


# ---------------------------------------------------------------- scripted run


async def scripted_run(args) -> int:
    r = Report()
    sb = await db.get_db()

    heading(1, "Preflight")
    tenant = await load_tenant(args.number, r)
    if tenant is None:
        return 1
    if not tenant.demo_mode and not args.allow_production:
        r.fail("tenant has demo_mode = false. This run creates real bookings, so it refuses "
               "to touch a live practice. Use a demo tenant or pass --allow-production.")
        return 1
    r.info(f"minutes: {tenant.used_minutes:.2f} used of {tenant.total_minute_limit}")
    calendar_ready = r.check(bool(tenant.cal_com_api_key), "Cal.com API key found in client_secrets")
    calendar_ready &= r.check(bool(tenant.cal_com_event_type_id), "cal_com_event_type_id is set")
    try:
        buckets = [b.id for b in await sb.storage.list_buckets()]
        r.check("recordings" in buckets, "storage bucket 'recordings' exists")
    except Exception as e:
        r.fail(f"could not list storage buckets: {e}")

    caller = normalize_phone(args.caller)
    call_id = await db.create_call_log(tenant.id, caller, f"sim-{uuid.uuid4()}", None)
    r.ok(f"call_logs row created ({call_id})")
    session = CallSession(tenant, call_id, caller)
    tz = session.tz
    appt_id = None
    billed = 0.0

    try:
        heading(2, "Availability")
        options = []
        if not calendar_ready:
            r.skip("calendar not configured; skipping steps 2-6")
        else:
            day = (datetime.now(tz) + timedelta(days=1)).date().isoformat()
            try:
                result = await session.check_availability(day, "any")
                options = result.get("options", [])
                if r.check(options, "Cal.com returned open slots", result.get("note", "")):
                    for o in options:
                        r.info(f"{o['spoken']}  ({o['slot_id']})")
                    if "note" in result:
                        r.info(result["note"])
            except CalComError as e:
                r.fail(f"Cal.com slots request failed: {e}")

        heading(3, "Booking")
        if not options:
            r.skip("no slots to book")
        else:
            result = await session.book_appointment(
                patient_name="Simulation Test Patient",
                phone_number=caller,
                visit_reason="Routine Cleaning",
                is_new_patient=True,
                is_emergency=False,
                slot_id=options[0]["slot_id"],
            )
            if r.check(result.get("status") == "booked", f"booked {options[0]['spoken']}", str(result)):
                row = (await sb.table("appointments").select("*").eq("call_log_id", call_id).execute()).data
                if r.check(len(row) == 1, "appointments row written"):
                    appt_id = row[0]["id"]
                    r.info(f"Cal.com booking uid: {row[0]['cal_booking_uid']}")
            else:
                r.info("See the Cal.com error logged above. A common cause is the event type still "
                       "requiring an email; make Email optional in its booking questions.")
            r.check(session.appointment_booked == bool(appt_id), "session marks the call as booked")

        heading(4, "Lookup by caller ID")
        if not appt_id:
            r.skip("nothing booked")
        else:
            found = (await session.find_appointments())["appointments"]
            r.check(any(a["appointment_id"] == appt_id for a in found),
                    f"find_appointments returned the booking ({len(found)} upcoming for this caller)")

        heading(5, "Reschedule")
        if not appt_id or len(options) < 2:
            r.skip("needs a booking and a second offered slot")
        else:
            result = await session.reschedule_appointment(appt_id, options[1]["slot_id"])
            if r.check(result.get("status") == "rescheduled", f"moved to {options[1]['spoken']}", str(result)):
                row = await fetch_one("appointments", "id", appt_id)
                moved = datetime.fromisoformat(row["start_time"]) == session.offered_slots[options[1]["slot_id"]]
                r.check(moved and row["status"] == "rescheduled", "appointments row updated")

        heading(6, "Cancel")
        if not appt_id:
            r.skip("nothing booked")
        else:
            try:
                result = await session.cancel_appointment(appt_id, "Simulation test cleanup")
                r.check(result.get("status") == "cancelled", "cancelled in Cal.com")
                row = await fetch_one("appointments", "id", appt_id)
                r.check(row["status"] == "cancelled", "appointments row marked cancelled")
            except CalComError as e:
                r.fail(f"Cal.com cancel failed: {e}")

        heading(7, "Emergency triage")
        if not calendar_ready:
            r.skip("calendar not configured; skipping the 2-hour slot search")
        else:
            try:
                result = await session.check_availability(datetime.now(tz).date().isoformat(), "any", is_emergency=True)
                slots = result.get("options", [])
                if slots:
                    r.ok(f"emergency slot within 2 hours: {slots[0]['spoken']}")
                    if args.book_emergency:
                        booked = await session.book_appointment(
                            "Simulation Emergency Patient", caller, "Emergency: knocked-out tooth",
                            False, True, slots[0]["slot_id"])
                        if r.check(booked.get("status") == "booked", "emergency slot booked"):
                            row = (await sb.table("appointments").select("*")
                                   .eq("call_log_id", call_id).eq("is_emergency", True).execute()).data[0]
                            r.check(row["is_emergency"], "appointments row has is_emergency = true")
                            await session.find_appointments()  # tools only change looked-up appointments
                            cancelled = await session.cancel_appointment(row["id"], "Simulation test cleanup")
                            r.check(cancelled.get("status") == "cancelled", "emergency booking cancelled again")
                else:
                    r.info("No slot in the next 2 hours, so Mia would flag an urgent callback. If you expected "
                           "one, lower the event type's minimum notice (Cal.com defaults to 2 hours).")
            except CalComError as e:
                r.fail(f"Cal.com emergency slot search failed: {e}")
        result = await session.flag_urgent_callback("Simulation: knocked-out tooth, no slot in 2 hours")
        row = await fetch_one("call_logs", "id", call_id)
        r.check(result["status"] == "flagged" and row["urgent_callback"], "urgent callback flagged on the call log")

        heading(8, "Billing")
        before = await used_minutes(tenant.id)
        expected = round(args.duration / 60, 2)
        transcript = [
            session.transcript_entry("mia", greeting_for(tenant)),
            session.transcript_entry("patient", "Simulated call from scripts/simulate_call.py"),
        ]
        billed = await db.finalize_call(call_id, args.duration, transcript, "Simulated test call.",
                                        session.appointment_booked)
        after = await used_minutes(tenant.id)
        r.check(billed == expected, f"{args.duration}s billed as {expected} minutes", f"got {billed}")
        r.check(round(after - before, 2) == expected, f"clients.used_minutes {before:.2f} -> {after:.2f}")
        again = await db.finalize_call(call_id, args.duration, transcript, None, False)
        r.check(again == 0 and await used_minutes(tenant.id) == after, "second finalize_call is a no-op")
        row = await fetch_one("call_logs", "id", call_id)
        r.check(row["status"] == "completed" and float(row["billed_minutes"]) == expected,
                "call_logs row completed with billed_minutes")
        r.info(f"remaining: {tenant.total_minute_limit - after:.2f} minutes")

        heading(9, "LLM triage (Mia's real prompt and tools, first reply only)")
        if not HAS_OPENAI:
            r.skip("OPENAI_API_KEY not set")
        elif args.skip_llm:
            r.skip("--skip-llm")
        else:
            await llm_triage(session, tenant, caller, r)

    finally:
        print("\nCleanup")
        # Any booking a failed step left behind is a real calendar entry; cancel it.
        leftovers = (await sb.table("appointments").select("cal_booking_uid")
                     .eq("call_log_id", call_id).neq("status", "cancelled").execute()).data
        for row in leftovers:
            uid = row["cal_booking_uid"]
            try:
                await session.calcom.cancel_booking(uid, "Simulation test cleanup")
                print(f"  cancelled leftover Cal.com booking {uid}")
            except CalComError as e:
                print(f"  could not cancel Cal.com booking {uid}: {e}  <- cancel it by hand")
        if args.keep:
            print(f"  --keep: left call_logs row {call_id} and {billed} billed minutes in Supabase")
        else:
            await sb.table("appointments").delete().eq("call_log_id", call_id).execute()
            await sb.table("call_logs").delete().eq("id", call_id).execute()
            if billed:
                current = await used_minutes(tenant.id)
                await sb.table("clients").update({"used_minutes": max(round(current - billed, 2), 0)}) \
                    .eq("id", tenant.id).execute()
            print(f"  deleted test rows and refunded {billed} minutes")

    print(f"\n{'All checks passed.' if r.failures == 0 else f'{r.failures} check(s) failed.'}")
    return 0 if r.failures == 0 else 1


TRIAGE_CASES = [
    # (label, patient line, kind)
    ("knocked-out tooth", "I just knocked my front tooth out playing basketball. What do I do?", "emergency"),
    ("airway swelling", "My jaw is really swollen and it's getting hard to swallow and breathe.", "airway"),
    ("severe pain", "I have a terrible toothache, like nine out of ten. I can't sleep.", "emergency"),
    ("routine cleaning", "Hi, I'd like to book a cleaning sometime next week.", "routine"),
]


async def llm_triage(session: CallSession, tenant, caller: str, r: Report):
    from openai import AsyncOpenAI

    from app import config

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    for label, line, kind in TRIAGE_CASES:
        resp = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=200,
            tools=openai_tools(session),
            messages=[
                {"role": "system", "content": build_system_prompt(tenant, caller)},
                {"role": "assistant", "content": greeting_for(tenant)},
                {"role": "user", "content": line},
            ],
        )
        msg = resp.choices[0].message
        text = (msg.content or "").lower()
        calls = [(tc.function.name, json.loads(tc.function.arguments or "{}")) for tc in msg.tool_calls or []]
        emergency_lookup = any(n == "check_availability" and a.get("is_emergency") for n, a in calls)
        routine_lookup = any(n == "check_availability" and not a.get("is_emergency") for n, a in calls)

        if kind == "airway":
            r.check("911" in text or "emergency room" in text, f"{label}: sends caller to 911 / ER first")
        elif kind == "emergency":
            r.check(not routine_lookup, f"{label}: not handled as a routine booking")
            if label == "knocked-out tooth":
                r.check("milk" in text or emergency_lookup, f"{label}: gives milk advice or looks up emergency slots")
        else:
            r.check(not emergency_lookup, f"{label}: not treated as an emergency")
        r.info(f"Mia: {msg.content or '(no text)'}")
        for name, a in calls:
            r.info(f"tool: {name}({json.dumps(a)})")
    r.info("LLM replies vary between runs; rerun a failed case before changing the prompt.")


# ------------------------------------------------------------------ chat mode


async def chat(args) -> int:
    """Type as the patient. Real prompt, real tools, real Cal.com and Supabase."""
    if not HAS_OPENAI:
        sys.exit("[FAIL] --chat needs OPENAI_API_KEY in .env")
    from openai import AsyncOpenAI

    from app import config
    from app.bot import _summarize

    r = Report()
    tenant = await load_tenant(args.number, r)
    if tenant is None:
        return 1
    caller = normalize_phone(args.caller)
    call_id = await db.create_call_log(tenant.id, caller, f"chat-{uuid.uuid4()}", None)
    session = CallSession(tenant, call_id, caller)
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    tools = openai_tools(session)

    # Same error handling as a live call: tool failures go back to the LLM as data.
    handlers = {
        name: session._wrap(name, getattr(session, name))
        for name in ("check_availability", "book_appointment", "find_appointments",
                     "reschedule_appointment", "cancel_appointment", "flag_urgent_callback")
    }

    greeting = greeting_for(tenant)
    messages = [{"role": "system", "content": build_system_prompt(tenant, caller)},
                {"role": "assistant", "content": greeting}]
    transcript = [session.transcript_entry("mia", greeting)]
    started = time.monotonic()
    print(f"\nCall {call_id} | caller ID {caller} | type 'quit' to hang up\n")
    print(f"Mia: {greeting}")

    ended = False
    while not ended:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("", "quit", "exit"):
            break
        messages.append({"role": "user", "content": line})
        transcript.append(session.transcript_entry("patient", line))

        while True:
            t0 = time.monotonic()
            resp = await client.chat.completions.create(
                model=config.OPENAI_MODEL, temperature=config.OPENAI_TEMPERATURE,
                max_tokens=200, tools=tools, messages=messages,
            )
            msg = resp.choices[0].message
            entry = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            messages.append(entry)
            if msg.content:
                print(f"Mia: {msg.content}   ({(time.monotonic() - t0) * 1000:.0f} ms)")
                transcript.append(session.transcript_entry("mia", msg.content))
            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                name, call_args = tc.function.name, json.loads(tc.function.arguments or "{}")
                print(f"     [tool] {name}({json.dumps(call_args)})")
                if name == "end_call":
                    farewell = call_args.get("farewell", "Goodbye!")
                    print(f"Mia: {farewell}")
                    transcript.append(session.transcript_entry("mia", farewell))
                    result, ended = {"status": "ending"}, True
                else:
                    box = {}

                    async def capture(res, properties=None, _box=box):
                        _box["result"] = res

                    await handlers[name](SimpleNamespace(arguments=call_args, result_callback=capture))
                    result = box.get("result")
                print(f"     [result] {json.dumps(result)}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            if ended:
                break

    duration = int(time.monotonic() - started)
    summary = await _summarize(transcript)
    billed = await db.finalize_call(call_id, duration, transcript, summary, session.appointment_booked)
    print(f"\nCall logged: {call_id}")
    print(f"  duration {duration}s, billed {billed} min, booked={session.appointment_booked}")
    print(f"  summary: {summary}")
    print("  Bookings made in chat are real. Cancel them in Cal.com, or call back and ask Mia to.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--number", default="+15125550100", help="the clinic's number as stored in clients.phone_number")
    parser.add_argument("--caller", default="+15125550199", help="simulated caller ID")
    parser.add_argument("--duration", type=int, default=95, help="simulated call length in seconds for billing")
    parser.add_argument("--chat", action="store_true", help="type as the patient instead of running the checks")
    parser.add_argument("--keep", action="store_true", help="keep the test call log and billed minutes")
    parser.add_argument("--book-emergency", action="store_true", help="also book (then cancel) an emergency slot")
    parser.add_argument("--skip-llm", action="store_true", help="skip the OpenAI triage checks")
    parser.add_argument("--allow-production", action="store_true", help="allow running against a demo_mode=false tenant")
    args = parser.parse_args()
    if _ENV_PROBLEM:
        sys.exit(f"[FAIL] {_ENV_PROBLEM}")
    args.number = normalize_phone(args.number)
    sys.exit(asyncio.run(chat(args) if args.chat else scripted_run(args)))


if __name__ == "__main__":
    main()
