"""The demo line: what Mia may say about herself and Boltix AI, the callback tool, and the call-start calendar cache.

No network. Product claims are checked against the prompt text itself, because that is the only place a wrong claim can
come from: the model repeats what it is given.
"""
import asyncio
import re
from datetime import datetime
from pathlib import Path

import pytest

from app import config
from app.demo_facts import DEMO_FACTS
from app.emergency import classify
from app.language import FILLER_VARIANTS, LanguageState
from app.prompts import _demo_rules, build_system_prompt
from tests.test_agent import TZ, CallSession, FakeCalCom, make_tenant, run, session  # noqa: F401  (session is a fixture)

ROOT = Path(__file__).resolve().parents[1]
VENDORS = ["OpenAI", "GPT", "Deepgram", "Cartesia", "Telnyx", "Pipecat", "Railway", "Cal.com", "Supabase", "Resend"]
FORBIDDEN_CLAIMS = ["HIPAA compliant", "HIPAA-compliant", "certified", "integrates with", "integrates into", "end-to-end encrypted", "fully secure"]


def demo_tenant(**kw):
    return make_tenant(demo_mode=True, name="Apex Dental Care", **kw)


def test_brand_is_boltix_everywhere():
    demo = build_system_prompt(demo_tenant(), "+15125550199")
    assert "Boltix" in demo and "Boltex" not in demo
    for path in [*ROOT.glob("app/*.py"), ROOT / "README.md", ROOT / "schema.sql", ROOT / ".env.example"]:
        assert "boltex" not in path.read_text(encoding="utf-8").lower(), f"leftover 'Boltex' in {path.name}"


def test_production_prompt_has_no_demo_rules_and_no_boltix():
    prod = build_system_prompt(make_tenant(demo_mode=False), "+15125550199")
    assert "Boltix" not in prod and "# Demo line" not in prod and "request_boltix_callback" not in prod
    assert "I can have the team call you back" in prod  # a caller asking for a real person is handled
    assert "# Demo line" in build_system_prompt(demo_tenant(), "+15125550199")


def test_demo_prompt_never_makes_a_claim_we_do_not_stand_behind():
    section = _demo_rules(demo_tenant())
    full = build_system_prompt(demo_tenant(), "+15125550199")
    assert not re.search(r"\$\s?\d", section), "the demo section must not state a price"
    assert not re.search(r"\b\d+\s?(%|percent|clients|practices|customers)\b", section), "no invented numbers"
    for text in (section, full):
        for vendor in VENDORS:
            assert vendor.lower() not in text.lower(), f"vendor name {vendor!r} in the prompt"
        for claim in FORBIDDEN_CLAIMS:
            assert claim.lower() not in text.lower(), f"forbidden claim {claim!r} in the prompt"
    for value in DEMO_FACTS.values():
        assert not re.search(r"\$\s?\d", value) and not any(v.lower() in value.lower() for v in VENDORS)


def test_the_facts_come_from_the_one_editable_dict():
    section = _demo_rules(demo_tenant())
    for key in ("what_mia_does", "pricing", "setup", "setup_time", "patient_texts", "call_transfer", "languages",
                "trial_contract", "customization", "practice_software", "privacy", "next_step"):
        assert DEMO_FACTS[key] in section, f"{key} is not used by the prompt"
    assert "audio is not recorded, but a written transcript and a short summary" in section
    assert "no, this call is not recorded" not in section.lower()
    assert "may be recorded" in _demo_rules(make_tenant(demo_mode=True, record_calls=True))


@pytest.mark.parametrize("text", [
    "Can you handle a patient who has a broken tooth?",
    "How would you handle a patient with a swollen face?",
    "What happens if a patient's tooth is knocked out?",
])
def test_manager_questions_do_not_start_the_emergency_script(text):
    assert classify(text) is None


@pytest.mark.parametrize("text", [
    "My tooth is killing me",
    "I knocked out my front tooth",
    "Can you help me, my face is swollen",
    "my son broke his tooth",
])
def test_real_emergencies_still_start_it(text):
    assert classify(text) == "emergency"


def test_callback_tool_exists_only_in_demo_mode():
    def names(tenant):
        return {f.name for f in CallSession(tenant, "c", "+15125550199").tools_schema().standard_tools}

    assert "request_boltix_callback" in names(demo_tenant())
    assert "request_boltix_callback" not in names(make_tenant(demo_mode=False))
    prod = CallSession(make_tenant(demo_mode=False), "c", "+15125550199")
    assert run(prod.request_boltix_callback("Pat Lee", "Lee Dental", "+15125550199")) == {"error": "not_available"}


def _confirm(s):
    s.note_said("Is the number you're calling from the best one to reach you?")
    s.note_said("Just to confirm, Pat Lee from Lee Dental, at this number. Is that correct?")
    s.note_heard("Yes, that's correct.")


def test_callback_rejects_a_name_the_caller_never_said():
    s = CallSession(demo_tenant(), "c1", "+15125550199")
    s.note_heard("Hi, I run a practice and I'm interested.")
    _confirm(s)
    first = run(s.request_boltix_callback("Pat Lee", "Lee Dental", "+15125550199"))
    assert first["error"] == "name_not_confirmed" and s.callback is None


def test_callback_needs_readback_number_question_and_practice(monkeypatch):
    from app import db, notify

    saved, emails = [], []

    async def insert(row):
        saved.append(row)
        return row

    monkeypatch.setattr(db, "insert_demo_lead", insert)
    monkeypatch.setattr(notify, "notify_demo_lead", lambda lead: emails.append(lead) or True)
    s = CallSession(demo_tenant(), "c2", "+15125550199")
    s.note_heard("I'm Pat Lee from Lee Dental")
    assert run(s.request_boltix_callback("Pat Lee", "", "+15125550199"))["error"] == "practice_not_given"
    assert run(s.request_boltix_callback("Pat Lee", "Lee Dental", "+15125550199"))["error"] == "confirmation_required"
    s.note_said("Just to confirm, Pat Lee from Lee Dental. Is that correct?")
    s.note_heard("Yes")
    assert run(s.request_boltix_callback("Pat Lee", "Lee Dental", "+15125550199"))["error"] == "phone_not_confirmed"
    _confirm(s)

    async def finish():
        # a number one digit off the caller ID is the caller ID (the model garbles digits it reads aloud)
        result = await s.request_boltix_callback("Pat Lee", "Lee Dental", "5155550199", best_time="mornings", notes="after-hours calls")
        again = await s.request_boltix_callback("Pat Lee", "Lee Dental", "+15125550199")
        await s.flush()
        return result, again

    result, again = asyncio.run(finish())
    assert result["status"] == "saved" and "will give you a call back" in result["say"]
    assert "shortly" not in result["say"].lower().replace("never say shortly", "")
    assert saved[0]["phone"] == "+15125550199" and saved[0]["name"] == "Pat Lee" and saved[0]["best_time"] == "mornings"
    assert saved[0]["call_log_id"] == "c2"
    assert again["status"] == "saved" and len(saved) == 1 and len(emails) == 1  # asked twice, saved once


def test_filler_never_repeats_the_same_line_twice_in_a_row():
    for kind, options in FILLER_VARIANTS.items():
        assert len(options) == 3 and all(len(o.split()) <= 5 for o in options), "short, under a second spoken"
        for _ in range(20):  # different random starts
            state = LanguageState()
            lines = [state.filler(kind, "en") for _ in range(9)]
            assert all(a != b for a, b in zip(lines, lines[1:])), lines
            assert set(lines) == set(options)
    assert LanguageState().filler("check", "es") == "Un momento, por favor, voy a revisar."  # other languages unchanged


class CountingCal(FakeCalCom):
    def __init__(self, slots):
        super().__init__(slots)
        self.slot_calls = 0

    async def get_slots(self, *a, **k):
        self.slot_calls += 1
        return await super().get_slots(*a, **k)


def test_prefetch_serves_several_searches_from_one_calendar_call_and_a_write_drops_it(session):  # noqa: F811
    day = session.tomorrow.date().isoformat()
    session.calcom = cal = CountingCal(session.calcom.slots)

    async def go():
        session.prefetch_slots()
        a = await session.check_availability(day, "any")
        b = await session.check_availability(day, "afternoon")
        c = await session.check_availability("tomorrow", "any", preferred_time="2 PM")
        assert cal.slot_calls == 1, "one Cal.com call must cover the prefetch and all three searches"
        assert a["options"] and b["options"] and c["exact_match"]
        option = a["options"][0]
        session.note_heard("My name is Jane Wilson")
        session.note_said("Is the number you're calling from the best one to reach you?")
        session.note_said("Just to confirm, Jane Wilson, tomorrow at nine. Is that correct?")
        session.note_heard("Yes")
        booked = await session.book_appointment("Jane Wilson", "5125550199", "Cleaning", True, False, option["slot_id"])
        assert booked["status"] == "booked"
        assert session._prefetch is None, "a booking must drop the cached copy"
        await session.check_availability(day, "any")
        assert cal.slot_calls == 2, "after a write the next search asks Cal.com again"

    asyncio.run(go())


def test_prefetch_falls_back_to_cal_com_when_the_copy_is_stale_or_failed(session):  # noqa: F811
    day = session.tomorrow.date().isoformat()
    session.calcom = cal = CountingCal(session.calcom.slots)

    async def go():
        session.prefetch_slots()
        await asyncio.sleep(0)
        fetched_at, *rest = session._prefetch
        session._prefetch = (fetched_at - 1000, *rest)  # older than the cache allows
        assert (await session.check_availability(day, "any"))["options"]
        assert cal.slot_calls == 2

        real = cal.get_slots

        async def boom(*a, **k):
            raise RuntimeError("calendar down")

        cal.get_slots = boom
        session.prefetch_slots()  # this copy fails
        await asyncio.sleep(0.01)
        cal.get_slots = real  # Cal.com is back: the search must go straight to it
        assert (await session.check_availability(day, "any"))["options"]

    asyncio.run(go())


def test_demo_call_report_email_is_off_without_a_recipient(monkeypatch):
    from app import notify

    call = notify.DemoCallEmail("+15125550199", 75, "Manager asked about price.", False, [{"role": "mia", "content": "Hi"}])
    monkeypatch.setattr(config, "BOLTIX_LEADS_EMAIL", "")
    monkeypatch.setattr(config, "RESEND_API_KEY", "re_x")
    assert notify.notify_demo_call(call) is False
    subject, html, text = notify.render_demo_call(notify.DemoCallEmail(
        "+15125550199", 75, "Asked <b>price</b>", True, [{"role": "patient", "content": "<script>x</script>"}]))
    assert "CALLBACK REQUESTED" in subject and "<script>" not in html and "1m 15s" in text


def test_only_insults_aimed_at_mia_count():
    from app.tools import is_abusive

    for fine in ["This stupid tooth is killing me", "The pain is shit, I can't sleep", "I feel so dumb, I forgot my appointment",
                 "My old dentist was useless", "this crown is garbage", "Thank you, this crown is garbage",
                 "The weather is terrible today", "I have a stupidly long question about crowns"]:
        assert not is_abusive(fine), fine
    for rude in ["You're a useless robot", "Shut up", "You stupid machine", "Are you stupid?", "Screw you",
                 "Fuck you", "You're a useless piece of garbage robot.", "Shut up you stupid idiot, just answer me."]:
        assert is_abusive(rude), rude


@pytest.mark.parametrize("line", ["You're a useless robot", "Shut up", "You stupid machine"])
def test_each_insult_aimed_at_mia_is_one_strike(line):
    s = CallSession(demo_tenant(), "c4", "+15125550199")
    s.note_heard(line)
    assert s.abuse_strikes == 1 and s.abuse_count == 1


def test_swearing_about_pain_gives_no_strikes():
    s = CallSession(demo_tenant(), "c5", "+15125550199")
    for line in ["This stupid tooth is killing me", "The pain is shit, I can't sleep", "I feel so dumb, I forgot my appointment",
                 "My old dentist was useless", "this crown is garbage"]:
        s.note_heard(line)
    assert s.abuse_strikes == 0


def test_an_emergency_call_never_ends_on_rudeness():
    from app.guards import PromiseGuard

    s = CallSession(demo_tenant(), "c6", "+15125550199")
    s.note_heard("My tooth is killing me, I knocked out my front tooth")  # emergency report
    assert s.emergency_seen
    for line in ["You stupid machine", "Shut up", "You're a useless robot"]:  # even insults aimed at Mia, after the emergency
        s.note_heard(line)
    assert s.abuse_count == 0
    assert PromiseGuard().check("I'm here to help.", False, 0, abuse=s.abuse_count) is None
    # strikes before an emergency do not carry over either
    t = CallSession(demo_tenant(), "c7", "+15125550199")
    t.note_heard("You stupid machine")
    t.note_heard("You're a useless robot")
    assert t.abuse_count == 2
    t.note_heard("Sorry. My face is swollen and the pain is 10 out of 10")
    assert t.emergency_seen and t.abuse_count == 0
    # an emergency found by the calendar search (the model's own is_emergency path) counts too
    u = CallSession(demo_tenant(), "c8", "+15125550199")
    u.calcom = FakeCalCom([])
    run(u._emergency_slots(u.calcom, datetime.now(TZ)))
    assert u.emergency_seen


def test_second_abusive_message_ends_the_call_politely():
    from app.guards import NOTES, PromiseGuard

    s = CallSession(demo_tenant(), "c3", "+15125550199")
    s.note_heard("You're a useless robot")
    s.note_heard("Hello?")
    assert s.abuse_strikes == 1
    g = PromiseGuard()
    assert g.check("I'm here to help.", False, 0, abuse=1) is None  # the first time: one calm line
    s.note_heard("Shut up you stupid idiot")
    assert s.abuse_strikes == 2
    assert g.check("I'm here to help.", False, 0, abuse=2) == "goodbye"
    assert g.check("", True, 0, abuse=2) is None  # end_call was called: nothing more to do
    assert "end_call" in NOTES["goodbye"]


def test_the_readback_wording_is_recognised_and_does_not_sound_booked():
    from app.guards import PromiseGuard, broke_promise
    from app.tools import is_readback

    line = "So that's a routine cleaning on Tuesday, October 27 at noon, for Taylor Brooks. Is that correct?"
    assert is_readback(line)
    assert broke_promise(line, called_tool=False) is None  # a question is never a claim that it was saved
    g = PromiseGuard()
    assert g.check(line, False, 0) is None
    assert g.check("Great, see you then.", False, 0) == "unsaved"  # a yes followed by no booking is still caught
    s = CallSession(demo_tenant(), "c9", "+15125550199")
    s.note_said("Is the number you're calling from the best one to reach you?")
    s.note_said(line)
    s.note_heard("Yes")
    assert s._caller_confirmed()
    prompt = build_system_prompt(make_tenant(), "+15125550199")
    assert "So that's a routine cleaning" in prompt and "Do not mention texts or emails unless the caller asks" in prompt
    assert "How can I assist you today?" in prompt  # listed as a phrase never to say
    assert "asking how I'd handle it" in build_system_prompt(demo_tenant(), "+15125550199")
