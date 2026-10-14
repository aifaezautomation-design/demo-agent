import asyncio
import json
import base64
import time
from datetime import date, datetime, timedelta
from datetime import time as dtime
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
    async def go():
        option = (await session.check_availability(session.tomorrow.date().isoformat(), "morning"))["options"][0]
        result = await session.book_appointment("Jane Doe", "512-555-0199", "Cleaning", True, False, option["slot_id"])
        # Our own record is written behind the reply; flush() is what lookups and call-end wait on.
        await session.flush()
        return result

    result = run(go())
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
    assert tts._settings.generation_config.volume == config.CARTESIA_VOLUME == 1.4


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


def test_prompt_prefix_is_identical_across_calls():
    # OpenAI caches the longest identical prompt prefix; per-call details must stay at the end.
    a = build_system_prompt(make_tenant(), "+15125550101")
    b = build_system_prompt(make_tenant(), "+19998887777")
    first_diff = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    assert first_diff > a.index("# This call")


# --- emergency search: same-day first, urgent callback flagged in code -----------------------

def _emergency_session(session, slot_hours_from_now):
    now = datetime(2030, 1, 7, 9, 0, tzinfo=TZ)  # a Monday morning
    session.calcom.slots = [now + timedelta(hours=h) for h in slot_hours_from_now]
    flagged = []

    async def flag(call_id, reason):
        flagged.append(reason)

    return now, flagged, flag


def test_emergency_prefers_slot_within_two_hours(session, monkeypatch):
    now, flagged, flag = _emergency_session(session, [0.5, 6, 27])
    monkeypatch.setattr(tools.db, "flag_urgent_callback", flag)
    result = run(session._emergency_slots(session.calcom, now))
    assert result["window"] == "within_two_hours" and len(result["options"]) == 1
    assert "urgent_callback" not in result and not flagged


def test_emergency_falls_back_to_later_today(session, monkeypatch):
    now, flagged, flag = _emergency_session(session, [6, 7, 27])
    monkeypatch.setattr(tools.db, "flag_urgent_callback", flag)
    result = run(session._emergency_slots(session.calcom, now))
    assert result["window"] == "later_today" and len(result["options"]) == 2
    assert not flagged


def test_emergency_with_no_same_day_slot_flags_callback_in_code(session, monkeypatch):
    now, flagged, flag = _emergency_session(session, [27])  # tomorrow only
    monkeypatch.setattr(tools.db, "flag_urgent_callback", flag)
    result = run(session._emergency_slots(session.calcom, now))
    assert result["window"] == "next_available_not_today" and result["options"]
    assert result["urgent_callback"] == "flagged" and len(flagged) == 1
    assert "another" not in result["say"].lower() and "emergency room" not in result["say"].lower()


def test_emergency_with_empty_calendar_still_flags_callback(session, monkeypatch):
    now, flagged, flag = _emergency_session(session, [])
    monkeypatch.setattr(tools.db, "flag_urgent_callback", flag)
    result = run(session._emergency_slots(session.calcom, now))
    assert result["options"] == [] and result["window"] == "none" and len(flagged) == 1


def test_prompt_emergency_rules():
    prompt = build_system_prompt(make_tenant(), "+15125550101")
    section = prompt[prompt.index("# Emergencies"): prompt.index("# Ending")]
    assert "never send the caller to another dentist" in section
    assert "911" in section  # life-threatening only: the single exception
    assert "nearest emergency room" not in prompt
    assert "check_availability call with is_emergency true" in section


# --- date grounding ----------------------------------------------------------------------------

TODAY = date(2026, 10, 14)  # a Wednesday


@pytest.mark.parametrize("text,expected", [
    ("today", date(2026, 10, 14)), ("tomorrow", date(2026, 10, 15)),
    ("friday", date(2026, 10, 16)), ("this friday", date(2026, 10, 16)),
    ("next friday", date(2026, 10, 23)), ("next thursday", date(2026, 10, 22)),
    ("thursday next week", date(2026, 10, 22)), ("next week", date(2026, 10, 19)),
    ("monday", date(2026, 10, 19)), ("wednesday", date(2026, 10, 14)),
    ("next wednesday", date(2026, 10, 21)), ("2026-10-28", date(2026, 10, 28)),
    ("2020-01-01", date(2026, 10, 14)),  # the past is clamped to today
    ("the 26th", date(2026, 10, 26)), ("26th", date(2026, 10, 26)), ("october 26", date(2026, 10, 26)),
    ("oct 26th", date(2026, 10, 26)), ("the 3rd", date(2026, 11, 3)),   # already past this month -> next month
    ("january 5", date(2027, 1, 5)), ("in two weeks", date(2026, 10, 28)), ("in 5 days", date(2026, 10, 19)),
    ("the week after next", date(2026, 10, 26)), ("wednesday of the week after next", date(2026, 10, 28)),
])
def test_resolve_day(text, expected):
    assert tools.resolve_day(text, TODAY) == (expected, True)


def test_resolve_day_reports_when_it_cannot_understand():
    assert tools.resolve_day("banana", TODAY) == (TODAY, False)


@pytest.mark.parametrize("text,expected", [
    ("14:00", dtime(14, 0)), ("2pm", dtime(14, 0)), ("2:30 PM", dtime(14, 30)),
    ("12am", dtime(0, 0)), ("12pm", dtime(12, 0)), ("25:00", None), ("later", None), (None, None),
])
def test_parse_time_pref(text, expected):
    assert tools.parse_time_pref(text) == expected


def test_prompt_contains_todays_date_and_tells_the_model_not_to_compute_dates():
    prompt = build_system_prompt(make_tenant(), "+15125550101")
    now = datetime.now(TZ)
    assert f"{now:%A, %B} {now.day}, {now.year}" in prompt
    assert "Never work out dates yourself" in prompt and "next thursday" in prompt


def test_availability_ranks_by_preferred_time_and_flags_exact_match(session):
    day = session.tomorrow.date()
    session.calcom.slots = [session.tomorrow.replace(hour=h) for h in (9, 11, 14, 16)]
    result = run(session.check_availability(day.isoformat(), "any", False, "2pm"))
    assert result["exact_match"] is True and result["same_day"] is True
    assert result["options"][0]["spoken"].endswith("2 PM")  # closest first
    assert result["searched_day"] == tools.spoken_day(day)

    result = run(session.check_availability(day.isoformat(), "any", False, "1pm"))
    assert result["exact_match"] is False  # 1 PM is not open; 2 PM is the closest, not an exact match


def test_availability_says_when_the_day_has_nothing(session):
    day = session.tomorrow.date() + timedelta(days=1)
    session.calcom.slots = [session.tomorrow.replace(hour=10) + timedelta(days=3)]  # only open 3 days after tomorrow
    result = run(session.check_availability(day.isoformat(), "any", False))
    assert result["same_day"] is False and "Nothing open on" in result["note"]
    assert len(result["options"]) == 1


def test_availability_understands_relative_days(session):
    result = run(session.check_availability("tomorrow", "morning", False))
    assert result["searched_day"] == tools.spoken_day(session.tomorrow.date())


# --- languages ----------------------------------------------------------------------------------

from pipecat.frames.frames import (  # noqa: E402
    LLMMessagesAppendFrame,
    ManuallySwitchServiceFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
)  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402
from pipecat.transcriptions.language import Language  # noqa: E402

from app import language as lang  # noqa: E402


def test_every_spoken_phrase_exists_in_every_supported_language():
    for kind, table in lang.PHRASES.items():
        assert set(table) == set(lang.SUPPORTED), f"{kind} is missing {set(lang.SUPPORTED) - set(table)}"
    assert all("911" in lang.phrase("emergency", code) for code in lang.SUPPORTED)  # safety line everywhere
    assert lang.phrase("check", "xx") == lang.phrase("check", "en")  # unknown language falls back to English


def test_switch_frames_only_touch_stt_for_urdu():
    state = lang.LanguageState("multi")
    multi, urdu = object(), object()
    state.stt_services = {"multi": multi, "ur": urdu}
    frames = lang.switch_frames(state, "es")
    assert [type(f) for f in frames] == [TTSUpdateSettingsFrame] and state.current == "es" and state.stt == "multi"

    frames = lang.switch_frames(state, "ur")
    assert [type(f) for f in frames] == [ManuallySwitchServiceFrame, TTSUpdateSettingsFrame]
    assert state.stt == "ur" and frames[0].service is urdu

    frames = lang.switch_frames(state, "en")  # back out of Urdu restores multilingual STT
    assert frames[0].service is multi and state.stt == "multi"


def _detect(state, text, code):
    detector = lang.LanguageDetector(state, None)
    frame = TranscriptionFrame(text, "u", "2030-01-01T00:00:00Z", Language(code) if code else None)
    detector._recent.append(text)  # process_frame does this before detecting
    return detector._detected(frame)


def test_detector_switches_on_a_clear_foreign_utterance():
    state = lang.LanguageState("multi")
    assert _detect(state, "hola quisiera una cita", "es") == "es"
    assert _detect(state, "hello I need an appointment", "en") is None  # already English


def test_detector_ignores_short_noisy_or_unsupported_utterances():
    state = lang.LanguageState("multi")
    assert _detect(state, "si", "es") is None                          # one word
    assert _detect(state, "bonjour oui merci", "fr") is None          # three words
    assert _detect(state, "ni hao wo yao yue", "zh") is None           # unsupported language
    assert _detect(state, "hola quisiera una cita", None) is None      # no language reported


def test_detector_leaves_language_alone_in_urdu_mode():
    state = lang.LanguageState("multi")
    state.stt_services = {"multi": object(), "ur": object()}
    lang.switch_frames(state, "ur")
    assert _detect(state, "hola quisiera una cita", "es") is None


def test_set_language_tool_applies_frames(session):
    applied = []

    async def apply(code):
        applied.append(code)
        session.language_state.current = code

    session.apply_language = apply
    assert run(session.set_language("ur"))["status"] == "ok" and applied == ["ur"]
    assert run(session.set_language("klingon"))["error"] == "unsupported_language"


def test_hindi_with_urdu_vocabulary_is_treated_as_urdu():
    state = lang.LanguageState("multi")
    assert _detect(state, "असामिलायक़म मुझे कल मुलाकात बुक करनी है", "hi") == "ur"     # Urdu greeting
    assert _detect(state, "मुझे कल अपॉइंटमेंट बुक करना है", "hi") == "hi"             # plain Hindi stays Hindi
    assert _detect(state, "मैं कल मुलाकात बुक करना चाहता हूं", "hi") == "hi"          # one weak marker is not enough
    assert _detect(state, "मेहरबानी करके कल मुलाकात बुक करें", "hi") == "ur"          # two weak markers are


def test_urdu_greeting_in_an_earlier_segment_counts():
    detector = lang.LanguageDetector(lang.LanguageState("multi"), None)
    for text in ["Aslam alaikum.", "मुझे अगले जुमेरात दोपहर दो बजे बुक करना है"]:
        detector._recent.append(text)
    frame = TranscriptionFrame("मुझे अगले जुमेरात दोपहर दो बजे बुक करना है", "u", "2030-01-01T00:00:00Z", Language("hi"))
    assert detector._detected(frame) == "ur"


def test_common_hindi_words_do_not_trigger_urdu():
    state = lang.LanguageState("multi")
    assert _detect(state, "जी हाँ शुक्रिया ज़रूर आऊँगा कल", "hi") == "hi"


@pytest.mark.parametrize("greeting", ["अस्लामायक़म", "असामिलायक़म", "अस्सलामु अलैकुम", "Aslam alaikum.", "assalamualaikum", "आदाब"])
def test_urdu_greeting_spellings_all_match(greeting):
    assert lang.looks_urdu(greeting)


@pytest.mark.parametrize("hindi", ["नमस्ते, मुझे अपॉइंटमेंट चाहिए", "जी हाँ शुक्रिया ज़रूर", "मैं कल मुलाकात book करना चाहता हूं"])
def test_hindi_phrases_do_not_match_urdu(hindi):
    assert not lang.looks_urdu(hindi)


def test_a_hindi_caller_is_not_switched_to_german_by_a_mislabeled_fragment():
    state = lang.LanguageState("multi")
    lang.switch_frames(state, "hi")
    assert _detect(state, "Mein Name ist Ahmed Khan", "de") is None


def test_switching_between_two_foreign_languages_needs_more_words():
    state = lang.LanguageState("multi")
    lang.switch_frames(state, "es")
    assert _detect(state, "bonjour je voudrais un rendez-vous", "fr") is None          # 5 words
    assert _detect(state, "bonjour je voudrais un rendez-vous demain matin", "fr") == "fr"  # 8 words
    assert _detect(state, "hello I would like an appointment", "en") == "en"           # back to English is easy


def test_stt_update_enters_at_pipeline_start_but_voice_update_goes_downstream():
    # STT is upstream of the detector, so a downstream STT frame would never reach it (a real bug caught live).
    state = lang.LanguageState("multi")
    state.stt_services = {"multi": object(), "ur": object()}
    queued, pushed = [], []
    detector = lang.LanguageDetector(state, lambda f: _collect(queued, f))

    async def push(frame, direction=None):
        pushed.append(frame)

    detector.push_frame = push
    run(detector._apply("ur", "बराए मेहरबानी मुलाकात उर्दू में बात करें", FrameDirection.DOWNSTREAM))
    assert [type(f) for f in queued] == [ManuallySwitchServiceFrame] and queued[0].service is state.stt_services["ur"]
    assert [type(f) for f in pushed] == [TTSUpdateSettingsFrame, LLMMessagesAppendFrame]
    assert "Urdu" in pushed[1].messages[0]["content"] and pushed[1].run_llm is False


async def _collect(bucket, frame):
    bucket.append(frame)
