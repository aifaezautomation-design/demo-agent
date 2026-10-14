import asyncio
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from app import config, notify

TZ = ZoneInfo("America/Chicago")


def booking(**kw) -> notify.BookingEmail:
    fields = dict(
        practice="Apex Dental Care", patient_name="Jonathan Whitfield", patient_phone="+15125550199",
        reason="Routine cleaning", start=datetime(2030, 1, 10, 14, 30, tzinfo=TZ), timezone_name="America/Chicago",
        is_emergency=False, is_new_patient=True, booking_uid="uid-1",
    )
    fields.update(kw)
    return notify.BookingEmail(**fields)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(config, "RESEND_API_KEY", "re_test")
    monkeypatch.setattr(config, "NOTIFY_EMAIL_TO", "owner@example.com")
    monkeypatch.setattr(config, "NOTIFY_EMAIL_FROM", "Mia <onboarding@resend.dev>")


def test_email_contains_every_requested_detail():
    subject, html_body, text = notify.render(booking())
    for fact in ["Jonathan Whitfield", "(512) 555-0199", "Routine cleaning", "Thursday, January 10, 2030 at 2:30 PM",
                 "CST (America/Chicago)", "New patient"]:
        assert fact in html_body, fact
        assert fact in text, fact
    assert subject == "New appointment: Jonathan Whitfield, Thu Jan 10, 2:30 PM"


def test_emergency_is_marked_urgent():
    subject, html_body, text = notify.render(booking(is_emergency=True))
    assert subject.startswith("URGENT:") and "Emergency appointment" in html_body and text.startswith("EMERGENCY")


def test_patient_text_cannot_inject_html():
    evil = '<script>alert(1)</script><img src=x onerror=alert(2)>'
    _, html_body, _ = notify.render(booking(patient_name=evil, reason=evil))
    assert "<script>" not in html_body and "<img" not in html_body and "&lt;script&gt;" in html_body


def test_recipient_prefers_the_tenants_own_address(on):
    assert notify.recipient_for({}) == "owner@example.com"
    assert notify.recipient_for({"notify_email": "front-desk@apex.example"}) == "front-desk@apex.example"


def test_off_without_a_key_or_recipient(monkeypatch):
    monkeypatch.setattr(config, "RESEND_API_KEY", "")
    monkeypatch.setattr(config, "NOTIFY_EMAIL_TO", "owner@example.com")
    assert notify.notify_booking(booking()) is False

    monkeypatch.setattr(config, "RESEND_API_KEY", "re_test")
    monkeypatch.setattr(config, "NOTIFY_EMAIL_TO", "")
    assert notify.notify_booking(booking(), {}) is False


def test_send_posts_the_right_request(on, monkeypatch):
    seen = {}

    async def fake_post(payload, key):
        seen.update(payload=payload, key=key)
        return httpx.Response(200, json={"id": "abc"})

    monkeypatch.setattr(notify, "_post", fake_post)
    assert asyncio.run(notify.send(booking(), "owner@example.com")) is True
    p = seen["payload"]
    assert p["to"] == ["owner@example.com"] and p["from"] == "Mia <onboarding@resend.dev>"
    assert "Jonathan Whitfield" in p["html"] and p["text"] and seen["key"] == "booking-uid-1"


@pytest.mark.parametrize("failure", [
    RuntimeError("boom"), httpx.ConnectTimeout("slow"), httpx.ConnectError("down"),
])
def test_a_failing_send_is_swallowed_and_retried_once(on, monkeypatch, failure):
    calls = []

    async def failing(payload, key):
        calls.append(1)
        raise failure

    async def no_sleep(_):
        return None

    monkeypatch.setattr(notify, "_post", failing)
    monkeypatch.setattr(notify.asyncio, "sleep", no_sleep)
    assert asyncio.run(notify.send(booking(), "owner@example.com")) is False  # returns, never raises
    assert len(calls) == 2


@pytest.mark.parametrize("status,expected_calls", [(401, 1), (403, 1), (422, 1), (500, 2), (429, 2)])
def test_client_errors_are_not_retried_but_server_errors_are(on, monkeypatch, status, expected_calls):
    calls = []

    async def respond(payload, key):
        calls.append(1)
        return httpx.Response(status, json={"message": "x"})

    async def no_sleep(_):
        return None

    monkeypatch.setattr(notify, "_post", respond)
    monkeypatch.setattr(notify.asyncio, "sleep", no_sleep)
    assert asyncio.run(notify.send(booking(), "owner@example.com")) is False
    assert len(calls) == expected_calls


def test_scheduling_returns_immediately_even_when_resend_hangs(on, monkeypatch):
    """The 0 ms guarantee: with a send that would take 30 seconds, the call path still returns in microseconds."""
    started = {}

    async def hangs(payload, key):
        started["yes"] = True
        await asyncio.sleep(30)
        return httpx.Response(200, json={})

    monkeypatch.setattr(notify, "_post", hangs)

    async def go():
        t0 = time.perf_counter()
        scheduled = notify.notify_booking(booking())
        elapsed = time.perf_counter() - t0
        await asyncio.sleep(0)  # let the task start, to prove it really is running in the background
        pending = list(notify._tasks)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return scheduled, elapsed, started.get("yes")

    scheduled, elapsed, running = asyncio.run(go())
    assert scheduled is True and running is True
    assert elapsed < 0.005, f"scheduling took {elapsed * 1000:.2f} ms"


# --- the booking path -----------------------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_a_successful_booking_triggers_one_email_and_does_not_wait_for_it(session, on, monkeypatch):
    sent = []

    async def slow_send(b, to):
        sent.append((b, to))
        await asyncio.sleep(30)

    monkeypatch.setattr(notify, "send", slow_send)
    session.tenant.knowledge_base = {}

    async def go():
        option = (await session.check_availability(session.tomorrow.date().isoformat(), "morning"))["options"][0]
        session.note_heard("My name is Jane Wilson")
        t0 = time.perf_counter()
        result = await session.book_appointment("Jane Wilson", "512-555-0199", "Routine cleaning", True, False, option["slot_id"])
        elapsed = time.perf_counter() - t0
        await asyncio.sleep(0)
        for t in list(notify._tasks):
            t.cancel()
        await asyncio.gather(*list(notify._tasks), return_exceptions=True)
        return result, elapsed

    result, elapsed = _run(go())
    assert result["status"] == "booked" and elapsed < 1.0  # booking reply is not held up by the 30 s email
    assert len(sent) == 1
    b, to = sent[0]
    assert (b.patient_name, b.patient_phone, b.reason, b.is_new_patient) == ("Jane Wilson", "+15125550199", "Routine cleaning", True)
    assert b.timezone_name == "America/Chicago" and b.booking_uid == "uid-1" and to == "owner@example.com"


def test_a_rejected_booking_sends_no_email(session, on, monkeypatch):
    sent = []

    async def record(b, to):
        sent.append(b)

    monkeypatch.setattr(notify, "send", record)

    async def go():
        option = (await session.check_availability(session.tomorrow.date().isoformat(), "any"))["options"][0]
        session.note_heard("Jane Wilson")
        session.calcom.fail_booking = True
        return await session.book_appointment("Jane Wilson", "5125550199", "Cleaning", False, False, option["slot_id"])

    assert _run(go())["error"] == "slot_no_longer_available"
    assert sent == []


def test_a_booking_still_succeeds_when_the_email_code_blows_up(session, on, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("email system on fire")

    monkeypatch.setattr(notify, "BookingEmail", explode)

    async def go():
        option = (await session.check_availability(session.tomorrow.date().isoformat(), "any"))["options"][0]
        session.note_heard("Jane Wilson")
        return await session.book_appointment("Jane Wilson", "5125550199", "Cleaning", False, False, option["slot_id"])

    # the failure must not turn a real calendar booking into an error for the caller
    assert _run(go())["status"] == "booked"
