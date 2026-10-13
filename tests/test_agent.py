import asyncio
import json
import base64
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from app import config, db, server, tools
from app.calcom import CalComError
from app.db import Tenant
from app.prompts import build_system_prompt
from app.tools import CallSession, normalize_phone, spoken_time

TZ = ZoneInfo("America/Chicago")


def make_tenant(**overrides) -> Tenant:
    fields = dict(
        id="apex-dental", name="Apex Dental Center", phone_number="+15125550100",
        timezone="America/Chicago", cal_com_event_type_id=111, emergency_event_type_id=None,
        cal_com_api_key="cal_test", knowledge_base={"address": "742 Evergreen Terrace", "services": [
            {"name": "Fillings", "price": "$180–$350", "minutes": 45}]},
    )
    fields.update(overrides)
    return Tenant(**fields)


class FakeCalCom:
    def __init__(self, slots):
        self.slots = slots
        self.bookings = []
        self.cancelled = []
        self.fail_booking = False

    async def get_slots(self, event_type_id, start, end, time_zone):
        return [s for s in self.slots if start <= s < end]

    async def create_booking(self, event_type_id, start_utc, name, phone, time_zone, metadata):
        if self.fail_booking:
            raise CalComError("409 slot taken")
        self.bookings.append((start_utc, name, phone))
        return {"uid": f"uid-{len(self.bookings)}"}

    async def reschedule_booking(self, uid, new_start_utc):
        return {"uid": uid + "-r"}

    async def cancel_booking(self, uid, reason):
        self.cancelled.append(uid)


class FakeDB:
    def __init__(self):
        self.appointments = {}

    async def insert_appointment(self, row):
        row = {**row, "id": f"appt-{len(self.appointments) + 1}", "status": "booked"}
        self.appointments[row["id"]] = row
        return row

    async def find_upcoming_appointments(self, client_id, phone, name, now_iso):
        return [a for a in self.appointments.values()
                if (phone and a["patient_phone"] == phone) or (name and name.lower() in a["patient_name"].lower())]

    async def get_appointment(self, client_id, appointment_id):
        return self.appointments.get(appointment_id)

    async def update_appointment(self, client_id, appointment_id, fields):
        self.appointments[appointment_id].update(fields)

    async def flag_urgent_callback(self, call_id, reason):
        self.urgent = reason


@pytest.fixture
def session(monkeypatch):
    fake_db = FakeDB()
    for name in ["insert_appointment", "find_upcoming_appointments", "get_appointment",
                 "update_appointment", "flag_urgent_callback"]:
        monkeypatch.setattr(tools.db, name, getattr(fake_db, name))
    tomorrow = (datetime.now(TZ) + timedelta(days=1)).replace(minute=0, second=0, microsecond=0)
    slots = [tomorrow.replace(hour=h) for h in (9, 10, 14, 15)]
    s = CallSession(make_tenant(), "call-1", "(512) 555-0199")
    s.calcom = FakeCalCom(slots)
    s.fake_db = fake_db
    s.tomorrow = tomorrow
    return s


def run(coro):
    return asyncio.run(coro)


def test_normalize_phone():
    assert normalize_phone("(512) 555-0199") == "+15125550199"
    assert normalize_phone("1-512-555-0199") == "+15125550199"
    assert normalize_phone("+44 20 7946 0958") == "+442079460958"
    assert normalize_phone("") is None


def test_spoken_time():
    assert spoken_time(datetime(2026, 10, 15, 9, 0, tzinfo=TZ)) == "Thursday, October 15 at 9 AM"
    assert spoken_time(datetime(2026, 10, 15, 14, 30, tzinfo=TZ)) == "Thursday, October 15 at 2:30 PM"


def test_prompt_renders_and_respects_demo_mode():
    demo = build_system_prompt(make_tenant(demo_mode=True), "+15125550199")
    prod = build_system_prompt(make_tenant(demo_mode=False), "+15125550199")
    assert "Boltex AI" in demo and "Fillings: $180–$350" in demo
    assert "Boltex" not in prod


def test_availability_filters_by_time_of_day(session):
    result = run(session.check_availability(session.tomorrow.date().isoformat(), "afternoon"))
    assert [o["spoken"].split(" at ")[1] for o in result["options"]] == ["2 PM", "3 PM"]


def test_booking_rejects_slot_that_was_never_offered(session):
    result = run(session.book_appointment("Jane Doe", "5125550199", "Cleaning", True, False, "2030-01-01T15:00:00Z"))
    assert result["error"] == "unknown_slot"
    assert session.calcom.bookings == []


def test_booking_offered_slot_records_appointment(session):
    option = run(session.check_availability(session.tomorrow.date().isoformat(), "morning"))["options"][0]
    result = run(session.book_appointment("Jane Doe", "512-555-0199", "Cleaning", True, False, option["slot_id"]))
    assert result["status"] == "booked"
    assert session.appointment_booked
    appt = session.fake_db.appointments["appt-1"]
    assert appt["patient_phone"] == "+15125550199" and appt["cal_booking_uid"] == "uid-1"


def test_taken_slot_is_reported_and_forgotten(session):
    option = run(session.check_availability(session.tomorrow.date().isoformat(), "any"))["options"][0]
    session.calcom.fail_booking = True
    result = run(session.book_appointment("Jane", "5125550199", "Cleaning", False, False, option["slot_id"]))
    assert result["error"] == "slot_no_longer_available"
    assert option["slot_id"] not in session.offered_slots


def test_cancel_requires_lookup_and_phone_verification(session):
    run(session.fake_db.insert_appointment({"patient_name": "Bob Smith", "patient_phone": "+15125550777",
                                            "cal_booking_uid": "uid-x", "start_time": session.tomorrow.isoformat(),
                                            "visit_reason": "Crown"}))
    assert run(session.cancel_appointment("appt-1", "conflict"))["error"] == "unknown_appointment"

    found = run(session.find_appointments(patient_name="bob smith"))["appointments"]
    assert found[0]["appointment_id"] == "appt-1"
    # Caller ID differs from the number on file, so last four digits are required.
    assert run(session.cancel_appointment("appt-1", "conflict"))["error"] == "verification_required"
    assert run(session.cancel_appointment("appt-1", "conflict", phone_last4="0000"))["error"] == "verification_required"
    assert run(session.cancel_appointment("appt-1", "conflict", phone_last4="0777"))["status"] == "cancelled"
    assert session.calcom.cancelled == ["uid-x"]


def test_services_construct_with_configured_settings():
    from app.bot import build_services

    stt, llm, tts = build_services()
    assert llm._settings.model == "gpt-4o-mini" and llm._settings.temperature == 0.2
    assert stt._settings.model == "nova-3-general" and stt._settings.smart_format is True
    assert tts._settings.voice == config.CARTESIA_VOICE_ID


@pytest.fixture
def client(monkeypatch):
    tenants = {
        "+15125550100": make_tenant(),
        "+15125550101": make_tenant(id="broke", name="Broke & Co Dental", used_minutes=2000.0),
        "+15125550102": make_tenant(id="rec", record_calls=True),
    }

    async def by_number(number):
        return tenants.get(number)

    monkeypatch.setattr(server.db, "get_tenant_by_number", by_number)
    return TestClient(server.app)


def test_texml_routes_by_dialed_number(client):
    assert "<Reject/>" in client.post("/telnyx/texml", data={"To": "+19999999999"}).text

    broke = client.post("/telnyx/texml", data={"To": "+15125550101"}).text
    assert "<Hangup/>" in broke and "Broke &amp; Co Dental" in broke

    ok = client.post("/telnyx/texml", data={"To": "+1 (512) 555-0100", "CallSid": "v3:CA1"}).text
    assert 'bidirectionalMode="rtp" enableReconnect="false"' in ok
    assert f'url="wss://voice.test/ws/{config.WS_AUTH_TOKEN}/%2B15125550100/v3%3ACA1"' in ok
    assert "<Recording" not in ok and "<Pause" not in ok

    assert 'recordingStatusCallback="https://voice.test/telnyx/recording?call_sid=CA1"' in client.post(
        "/telnyx/texml", data={"To": "+15125550102", "CallSid": "CA1"}).text


def test_websocket_rejects_bad_token(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/wrong/%2B15125550100/CA1") as ws:
            ws.receive_text()


def test_websocket_accepts_telnyx_handshake(client, monkeypatch):
    # The two messages Telnyx sends when a TeXML <Stream> connects.
    seen = {}

    async def fake_run_bot(websocket, call_data, tenant, call_sid):
        seen.update(call_data=call_data, tenant=tenant, call_sid=call_sid)

    monkeypatch.setattr(server, "run_bot", fake_run_bot)
    url = f"/ws/{config.WS_AUTH_TOKEN}/%2B15125550100/v3%3ACA1"
    with client.websocket_connect(url) as ws:
        ws.send_text(json.dumps({"event": "connected", "version": "1.0.0"}))
        ws.send_text(json.dumps({
            "event": "start", "sequence_number": "1", "stream_id": "stream-123",
            "start": {"call_control_id": "v3:ccid-1", "from": "+15125550199", "to": "+15125550100",
                      "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1}},
        }))
    assert seen["tenant"].id == "apex-dental"
    assert seen["call_sid"] == "v3:CA1"
    data = seen["call_data"]
    assert (data.stream_id, data.call_id, data.from_number, data.outbound_encoding) == (
        "stream-123", "v3:ccid-1", "+15125550199", "PCMU")


def test_webhook_signature_enforced(client, monkeypatch):
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes_raw()
    monkeypatch.setattr(config, "TELNYX_PUBLIC_KEY", base64.b64encode(public).decode())

    body = b"To=%2B15125550100&CallSid=CA1"
    ts = str(int(time.time()))
    good = base64.b64encode(key.sign(f"{ts}|".encode() + body)).decode()
    headers = {"content-type": "application/x-www-form-urlencoded", "telnyx-timestamp": ts}

    assert client.post("/telnyx/texml", content=body, headers=headers).status_code == 401
    assert client.post("/telnyx/texml", content=body,
                       headers={**headers, "telnyx-signature-ed25519": good}).status_code == 200
    tampered = body.replace(b"0100", b"0101")
    assert client.post("/telnyx/texml", content=tampered,
                       headers={**headers, "telnyx-signature-ed25519": good}).status_code == 401
