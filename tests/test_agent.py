import asyncio
import base64
import json
import time
from datetime import date, datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from app import config, server, tools
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


def confirmed(session):
    """The caller heard the read-back and said yes, which book_appointment now insists on."""
    session.note_said("Is the number you're calling from the best one to reach you?")
    session.note_said("Just to confirm, I have you booked for that time. Is that correct?")
    session.note_heard("Yes, that's correct.")
    return session


async def _search_and_flush(session, now):
    """The emergency search, then wait for its background writes (production waits for them when the call ends)."""
    result = await session._emergency_slots(session.calcom, now)
    await session.flush()
    return result


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
    session.note_heard("my name is Jane Wilson")
    confirmed(session)
    result = run(session.book_appointment("Jane Wilson", "5125550199", "Cleaning", True, False, "2030-01-01T15:00:00Z"))
    assert result["error"] == "unknown_slot"
    assert session.calcom.bookings == []


def test_booking_offered_slot_records_appointment(session):
    async def go():
        option = (await session.check_availability(session.tomorrow.date().isoformat(), "morning"))["options"][0]
        session.note_heard("My name is Jane Wilson")
        confirmed(session)
        result = await session.book_appointment("Jane Wilson", "512-555-0199", "Cleaning", True, False, option["slot_id"])
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
    session.note_heard("Jane Wilson")
    confirmed(session)
    result = run(session.book_appointment("Jane Wilson", "5125550199", "Cleaning", False, False, option["slot_id"]))
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


def test_services_construct_with_configured_settings(monkeypatch):
    from app.bot import build_services

    assert config.DEEPGRAM_STT == "nova"  # the phone-proven default; Flux is opt-in
    monkeypatch.setattr(config, "DEEPGRAM_STT", "flux")
    stt, llm, tts = build_services(["Dr. Sarah Jenkins", "Delta Dental"])  # Deepgram Flux (English)
    assert llm._settings.model == "gpt-4o-mini" and llm._settings.temperature == config.OPENAI_TEMPERATURE <= 0.3
    assert llm._settings.max_tokens == config.OPENAI_MAX_TOKENS <= 150
    assert stt._settings.model == "flux-general-en"
    assert stt._settings.eager_eot_threshold == 0.5 and stt._settings.eot_threshold == 0.7
    assert stt._settings.eot_timeout_ms == 2000  # Flux's own default is 5000: a paused caller is not left waiting 5 s
    assert stt._settings.keyterm == ["Dr. Sarah Jenkins", "Delta Dental"]
    assert tts._settings.voice == config.CARTESIA_VOICE_ID
    assert tts._settings.generation_config.volume == config.CARTESIA_VOLUME == 1.4

    monkeypatch.setattr(config, "DEEPGRAM_STT", "nova")  # the rollback setting
    stt, _, _ = build_services()
    assert stt._settings.model == "nova-3-general" and stt._settings.smart_format is True


def test_keyterms_come_from_the_clinic_knowledge_base():
    from app.bot import keyterms_for

    terms = keyterms_for(APEX_KB)
    for expected in ["Dr. Sarah Jenkins", "Sarah Jenkins", "Michael Vance", "Delta Dental", "Premier Access", "dentists", "root canal"]:
        assert expected in terms, expected
    assert len(terms) == len({t.lower() for t in terms}) <= 30  # no duplicates, bounded
    assert keyterms_for(None)[:1] == ["dentist"] and keyterms_for({}) == keyterms_for(None)


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
    result = run(_search_and_flush(session, now))
    assert result["window"] == "within_two_hours" and len(result["options"]) == 1
    assert "urgent_callback" not in result and not flagged


def test_emergency_falls_back_to_later_today(session, monkeypatch):
    now, flagged, flag = _emergency_session(session, [6, 7, 27])
    monkeypatch.setattr(tools.db, "flag_urgent_callback", flag)
    result = run(_search_and_flush(session, now))
    assert result["window"] == "later_today" and len(result["options"]) == 2
    assert not flagged


def test_emergency_with_no_same_day_slot_flags_callback_in_code(session, monkeypatch):
    now, flagged, flag = _emergency_session(session, [27])  # tomorrow only
    monkeypatch.setattr(tools.db, "flag_urgent_callback", flag)
    result = run(_search_and_flush(session, now))
    assert result["window"] == "next_available_not_today" and result["options"]
    assert result["urgent_callback"] == "flagged" and len(flagged) == 1
    assert "another" not in result["say"].lower() and "emergency room" not in result["say"].lower()


def test_emergency_with_empty_calendar_still_flags_callback(session, monkeypatch):
    now, flagged, flag = _emergency_session(session, [])
    monkeypatch.setattr(tools.db, "flag_urgent_callback", flag)
    result = run(_search_and_flush(session, now))
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


# --- English-only by default, multilingual behind a flag ---------------------------------------

def _tool_names(session):
    return [t.name for t in session.tools_schema().standard_tools]


def test_english_only_by_default_has_no_multilingual_overhead(session):
    assert config.MULTILINGUAL_ENABLED is False and config.DEEPGRAM_LANGUAGE == "en"
    assert "set_language" not in _tool_names(session)
    prompt = build_system_prompt(make_tenant(), "+15125550101")
    assert "You speak English only" in prompt
    assert "Detect the caller's language" not in prompt and "Urdu" not in prompt and "Devanagari" not in prompt


def test_multilingual_flag_restores_language_rules_and_tool(session, monkeypatch):
    monkeypatch.setattr(config, "MULTILINGUAL_ENABLED", True)
    assert "set_language" in _tool_names(session)
    prompt = build_system_prompt(make_tenant(), "+15125550101")
    assert "Detect the caller's language" in prompt and "You speak English only" not in prompt


# --- opening hours guard and the clinic knowledge base ---------------------------------------------------------

import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

APEX_KB = _json.loads((_Path(__file__).parent.parent / "data" / "apex_dental_care.json").read_text(encoding="utf-8"))["knowledge_base"]


def _at(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=TZ)


@pytest.mark.parametrize("slot,expected", [
    # The Cal.com schedule is Mon-Sat 9 AM-8 PM Eastern, which is 8 AM-7 PM Central (the practice's time zone).
    (_at(2030, 1, 9, 8, 0), True),     # Wednesday opening time (9 AM Eastern)
    (_at(2030, 1, 9, 7, 0), False),    # before opening (8 AM Eastern)
    (_at(2030, 1, 9, 18, 0), True),    # last 60-minute slot on a weekday: ends exactly at 7 PM (8 PM Eastern)
    (_at(2030, 1, 9, 18, 30), False),  # would run past 7 PM
    (_at(2030, 1, 9, 19, 0), False),   # after closing
    (_at(2030, 1, 12, 8, 0), True),    # Saturday opens at the same time as weekdays
    (_at(2030, 1, 12, 18, 0), True),   # Saturday's last slot is the same as weekdays
    (_at(2030, 1, 12, 19, 0), False),  # Saturday evening is closed
    (_at(2030, 1, 13, 10, 0), False),  # Sunday is closed
])
def test_slot_in_hours(slot, expected):
    assert tools.slot_in_hours(APEX_KB, slot) is expected


def test_slot_in_hours_allows_everything_without_a_schedule():
    assert tools.slot_in_hours({}, _at(2030, 1, 13, 3, 0)) is True


def test_slots_outside_opening_hours_are_never_offered(session):
    session.tenant.knowledge_base = APEX_KB
    day = _at(2030, 1, 9, 0)  # a Wednesday
    session.calcom.slots = [day.replace(hour=h) for h in (7, 8, 12, 18, 19, 20)]
    options = session._offer(session._open_hours(session.calcom.slots))["options"]
    assert [o["spoken"].split(" at ")[1] for o in options] == ["8 AM", "12 PM", "6 PM"]


def test_an_after_hours_emergency_is_flagged_not_booked(session, monkeypatch):
    session.tenant.knowledge_base = APEX_KB
    flagged = []

    async def flag(call_id, reason):
        flagged.append(reason)

    monkeypatch.setattr(tools.db, "flag_urgent_callback", flag)
    now = _at(2030, 1, 9, 19, 30)  # 7:30 PM Wednesday: closed. A calendar that offered 8 PM would still not be booked.
    session.calcom.slots = [now + timedelta(minutes=30)]
    result = run(_search_and_flush(session, now))
    assert result["options"] == [] and result["urgent_callback"] == "flagged" and flagged


def test_prompt_carries_the_clinic_knowledge():
    prompt = build_system_prompt(make_tenant(name="Apex Dental Care", knowledge_base=APEX_KB), "+15125550101")
    for fact in ["Apex Dental Care", "742 Evergreen Terrace, Suite 100", "Monday to Saturday 8:00 AM to 7:00 PM Central Time",
                 "9:00 AM to 8:00 PM Eastern", "Sunday closed", "Dr. Sarah Jenkins", "Dr. Michael Vance",
                 "Orthodontist", "restorative and cosmetic", "Teeth whitening: price confirmed by the team",
                 "Emergency tooth pain care", "never promise a particular dentist"]:
        assert fact in prompt, fact



# --- bookings must use what the caller actually said -------------------------------------------------------------

def _book(session, name, phone="5125550199"):
    option = run(session.check_availability(session.tomorrow.date().isoformat(), "any"))["options"][0]
    confirmed(session)
    return run(session.book_appointment(name, phone, "Cleaning", True, False, option["slot_id"]))


def test_booking_is_refused_when_the_caller_never_gave_a_name(session):
    session.note_heard("I just knocked out my front tooth")
    result = _book(session, "Alex Morgan")  # the model invented it
    assert result["error"] == "name_not_confirmed" and session.calcom.bookings == []


@pytest.mark.parametrize("placeholder", ["Unknown", "Patient", "John Doe", "caller", "N/A", ""])
def test_placeholder_names_are_refused_even_if_said(session, placeholder):
    session.note_heard("unknown patient john doe caller")
    assert _book(session, placeholder)["error"] == "name_not_confirmed"


def test_a_name_the_caller_said_is_accepted_even_if_spelled_slightly_differently(session):
    session.note_heard("my name is Jonathon Whitfeld")        # as the speech recognizer wrote it
    assert _book(session, "Jonathan Whitfield")["status"] == "booked"


def test_the_caller_id_is_an_acceptable_phone_number_but_an_invented_one_is_not(session):
    session.note_heard("this is Maria Lopez")
    assert _book(session, "Maria Lopez", "(512) 555-0199")["status"] == "booked"      # matches the caller ID
    assert _book(session, "Maria Lopez", "415 555 0123")["error"] == "phone_not_confirmed"  # never mentioned
    session.note_heard("my number is 415-555-0123")
    assert _book(session, "Maria Lopez", "415 555 0123")["status"] == "booked"       # the caller said it


def test_guard_matches_the_calcom_schedule_exactly_in_every_season():
    """Cal.com's schedule is Mon-Sat 9 AM-8 PM Eastern. The guard (stored in Central) must accept exactly the
    60-minute slots that schedule can produce, and nothing else: checked every half hour across a whole year,
    including both daylight-saving changes, which move Eastern and Central together."""
    eastern = ZoneInfo("America/New_York")
    mismatches = []
    moment = datetime(2030, 1, 1, 0, 0, tzinfo=TZ)
    while moment.year == 2030:
        e = moment.astimezone(eastern)
        cal_offers = e.weekday() <= 5 and dtime(9, 0) <= e.time() and (e + timedelta(minutes=60)).time() <= dtime(20, 0) \
            and (e + timedelta(minutes=60)).date() == e.date()
        if tools.slot_in_hours(APEX_KB, moment) != cal_offers:
            mismatches.append(moment.isoformat())
        moment += timedelta(minutes=30)
    assert not mismatches, mismatches[:5]



# --- bookkeeping writes happen behind the reply -------------------------------------------------------------------

class _SlowDB:
    """Every database write takes 300 ms; reads of rows we already hold must not touch the database at all."""

    def __init__(self, inner):
        self.inner, self.delay, self.calls = inner, 0.3, []

    def __getattr__(self, name):
        real = getattr(self.inner, name)

        async def call(*a, **k):
            self.calls.append(name)
            await asyncio.sleep(self.delay)
            return await real(*a, **k)

        return call


def _slow_db(session, monkeypatch):
    slow = _SlowDB(session.fake_db)
    for name in ("insert_appointment", "find_upcoming_appointments", "get_appointment", "update_appointment", "flag_urgent_callback"):
        monkeypatch.setattr(tools.db, name, getattr(slow, name))
    return slow


def test_cancel_and_reschedule_do_not_wait_for_the_database(session, monkeypatch):
    run(session.fake_db.insert_appointment({"patient_name": "Bob Smith", "patient_phone": "+15125550199", "cal_booking_uid": "uid-x",
                                            "start_time": session.tomorrow.isoformat(), "visit_reason": "Crown"}))
    slow = _slow_db(session, monkeypatch)

    async def go():
        await session.find_appointments()                       # one real read (300 ms)
        opts = (await session.check_availability(session.tomorrow.date().isoformat(), "any"))["options"]
        t0 = time.perf_counter()
        moved = await session.reschedule_appointment("appt-1", opts[1]["slot_id"])
        cancelled = await session.cancel_appointment("appt-1", "changed my mind")
        took = time.perf_counter() - t0
        await session.flush()
        return moved, cancelled, took

    moved, cancelled, took = run(go())
    assert moved["status"] == "rescheduled" and cancelled["status"] == "cancelled"
    assert took < 0.25, f"two changes took {took:.2f}s: a 300 ms database write is on the caller's path"
    assert "get_appointment" not in slow.calls                  # served from the lookup already made in this call
    assert slow.calls.count("update_appointment") == 2          # both writes did happen, after the replies
    assert session.fake_db.appointments["appt-1"]["status"] == "cancelled"
    assert session.fake_db.appointments["appt-1"]["cal_booking_uid"] == "uid-x-r"


def test_a_cancelled_appointment_cannot_be_cancelled_twice_in_one_call(session):
    run(session.fake_db.insert_appointment({"patient_name": "Bob Smith", "patient_phone": "+15125550199", "cal_booking_uid": "uid-x",
                                            "start_time": session.tomorrow.isoformat(), "visit_reason": "Crown"}))

    async def go():
        await session.find_appointments()
        first = await session.cancel_appointment("appt-1", "x")
        second = await session.cancel_appointment("appt-1", "x")
        await session.flush()
        return first, second

    first, second = run(go())
    assert first["status"] == "cancelled" and second["error"] == "appointment_not_active"
    assert session.calcom.cancelled == ["uid-x"]


def test_flagging_an_urgent_callback_returns_at_once(session, monkeypatch):
    slow = _slow_db(session, monkeypatch)

    async def go():
        t0 = time.perf_counter()
        result = await session.flag_urgent_callback("test")
        took = time.perf_counter() - t0
        await session.flush()
        return result, took

    result, took = run(go())
    assert result["status"] == "flagged" and took < 0.1 and slow.calls == ["flag_urgent_callback"]


def test_session_is_busy_only_while_a_tool_runs(session):
    """The idle timer must not read a running calendar lookup as the caller going quiet."""
    seen = []

    async def handler():
        seen.append(session.busy)
        raise RuntimeError("boom")  # a failing tool still has to release the flag

    async def result_callback(result):
        seen.append(result["error"])

    params = type("P", (), {"arguments": {}, "result_callback": staticmethod(result_callback)})()
    assert not session.busy
    run(session._wrap("t", handler)(params))
    assert seen == [True, "internal_error"] and not session.busy



def test_broken_promise_detection():
    """The 23:17 failure: Mia said "Let me check for availability this Friday." and called no tool, leaving the caller in silence."""
    from app.bot import broke_promise

    said = "Thank you! One moment, please. Just to confirm, you're looking for a checkup. Let me check for availability this Friday."
    assert broke_promise(said, called_tool=False) == "promise"
    assert broke_promise(said, called_tool=True) is None  # promise kept
    assert broke_promise("We have nine AM or ten thirty on Friday. Which works better?", called_tool=False) is None
    assert broke_promise("Just to confirm, nine AM Friday for a cleaning. Is that correct?", called_tool=False) is None  # waits for the caller
    assert broke_promise("", called_tool=False) is None
    assert broke_promise("I'll check the calendar for you.", called_tool=False) == "promise"


def test_unsaved_booking_claim_detection():
    """Found by conversation_eval: Mia said "You're all set!" after the read-back and never called book_appointment."""
    from app.bot import broke_promise

    for claim in ("You're all set! Just bring yourself.", "I've booked your cleaning for Thursday at ten.",
                  "Your appointment is confirmed for Friday at nine.", "Great! I have you booked for a check-up on Friday at noon.", "I have rescheduled your visit to Monday."):
        assert broke_promise(claim, called_tool=False, saved=False) == "unsaved", claim
        assert broke_promise(claim, called_tool=False, saved=True) is None, claim  # a real booking was saved
        assert broke_promise(claim, called_tool=True, saved=False) is None
    assert broke_promise("Just to confirm, I have you booked for Friday at nine.", called_tool=False, saved=False) is None
    assert broke_promise("Thanks for calling, goodbye!", called_tool=False, saved=False) is None


def test_name_guard_accepts_hyphens_spelling_and_does_not_loop(session):
    """Found by conversation_eval: a hyphenated, spelled-out name was rejected on every attempt and the caller was stuck."""
    session.heard = ["it's siobhan fitzgerald-nowak. that's s-i-o-b-h-a-n f-i-t-z-g-e-r-a-l-d - n-o-w-a-k."]
    assert session._name_is_from_caller("Siobhan Fitzgerald-Nowak")
    session.heard = ["my name is joy smith"]
    assert not session._name_is_from_caller("Peter Jones")  # still blocks invented names
    assert not session._insist("name", "Peter Jones") and session._insist("name", "Peter Jones")  # but never forever


def test_nothing_open_flags_a_callback_in_code(session):
    session.calcom.slots = []
    result = run(session.check_availability("tomorrow", "any", False))
    run(session.flush())
    assert result["options"] == [] and "team member will call" in result["say"]
    assert session.fake_db.urgent  # recorded for the practice by code, not by the model remembering to


def test_part_of_day_never_hides_openings(session):
    result = run(session.check_availability("tomorrow", "evening", False))  # fixture slots are 9, 10, 14, 15
    assert result["options"] and "Nothing open in the evening" in result["note"]


def test_promise_guard_after_readback():
    """After "Just to confirm ... Is that correct?" the next reply must be a tool call or a question, never a bare statement."""
    from app.bot import PromiseGuard

    g = PromiseGuard()
    assert g.check("Just to confirm, Paul Young, Friday at noon for a checkup. Is that correct?", False, 0) is None
    assert g.check("Great, see you then. Have a lovely day.", False, 0) == "unsaved"  # a yes, no book_appointment call
    g = PromiseGuard()
    g.check("Just to confirm, Paul Young, Friday at noon. Is that correct?", False, 0)
    assert g.check("", True, 0) is None  # the tool was called
    assert g.check("Your appointment is set for Friday at noon.", False, 1) is None  # and a booking really was saved
    g = PromiseGuard()
    g.check("Just to confirm, Friday at noon. Is that correct?", False, 0)
    assert g.check("What name should I use?", False, 0) is None  # a question is fine; the stale read-back is dropped
    assert g.check("Thanks, one moment.", False, 0) == "promise"


def test_booking_needs_a_readback_and_a_yes(session):
    """Found by conversation_eval: Mia booked straight after the caller picked a time, with no read-back."""
    option = run(session.check_availability(session.tomorrow.date().isoformat(), "any"))["options"][0]
    session.note_heard("My name is Jane Wilson")
    args = ("Jane Wilson", "5125550199", "Cleaning", True, False, option["slot_id"])
    assert run(session.book_appointment(*args))["error"] == "confirmation_required"  # nothing was read back
    session.note_said("Is the number you're calling from the best one to reach you?")
    session.note_said("Which time works best?")
    session.note_heard("Nine please")
    assert run(session.book_appointment(*args))["error"] == "confirmation_required"  # a pick is not a yes
    session.note_said("Just to confirm, Jane Wilson, tomorrow at nine for a cleaning. Is that correct?")
    session.note_heard("No, actually make it ten")
    assert run(session.book_appointment(*args))["error"] == "confirmation_required"  # a no is not a yes
    session.note_said("Just to confirm, Jane Wilson, tomorrow at ten for a cleaning. Is that correct?")
    session.note_heard("Yeah, that's right")
    assert run(session.book_appointment(*args))["status"] == "booked"


def test_emergency_booking_skips_the_readback(session):
    option = run(session.check_availability(session.tomorrow.date().isoformat(), "any"))["options"][0]
    session.note_heard("My name is Jane Wilson")
    assert run(session.book_appointment("Jane Wilson", "5125550199", "Swelling", True, True, option["slot_id"]))["status"] == "booked"


def test_unsearched_availability_claim_is_flagged():
    from app.guards import PromiseGuard

    g = PromiseGuard()
    assert g.check("The three PM slot isn't available. Would you like a morning time?", False, 0, searched=False) == "unsearched"
    assert g.check("The three PM slot isn't available.", False, 0, searched=True) is None  # a search ran this turn
    assert g.check("We're closed on Sundays.", False, 0, searched=False) is None


def test_booking_needs_the_number_question_and_never_books_a_garbled_caller_id(session):
    """Found by conversation_eval: Mia booked without asking about the number, and once read the caller ID back wrong (+5122...)."""
    option = run(session.check_availability(session.tomorrow.date().isoformat(), "any"))["options"][0]
    session.note_heard("Jane Wilson")
    session.note_said("Just to confirm, Jane Wilson, tomorrow at nine. Is that correct?")
    session.note_heard("Yes")
    args = ("Jane Wilson", "5125550199", "Cleaning", True, False, option["slot_id"])
    assert run(session.book_appointment(*args))["error"] == "phone_not_confirmed"  # nobody asked about the number
    session.note_said("Is the number you're calling from the best one to reach you?")
    session.note_said("Just to confirm, Jane Wilson, tomorrow at nine. Is that correct?")
    session.note_heard("Yes")
    garbled = ("Jane Wilson", "5155550199", "Cleaning", True, False, option["slot_id"])  # one digit off the caller ID
    assert run(session.book_appointment(*garbled))["status"] == "booked"
    assert session.calcom.bookings[-1][2] == "+15125550199"  # the caller ID, not the garbled digits
