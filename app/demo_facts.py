"""The facts Mia may state about herself and Boltix AI on the demo line.

Edit the strings below to change what Mia says. Nothing else needs to change: `app/prompts.py` builds the demo
section of her prompt from this dict, and `tests/test_demo.py` checks that no fact makes a claim we do not stand
behind (no prices, no certifications, no vendor names).

Rules for every value: plain spoken English, no numbers or promises beyond what is written here. If a fact is not
here, Mia says the Boltix AI team will cover it on a call. The website is written the way it is spoken
("boltix A I dot com") so the voice pronounces it correctly.
"""

DEMO_FACTS: dict[str, str] = {
    "company": "Boltix AI",
    "website": "boltix A I dot com",
    "what_mia_does": (
        "answers every call day or night with no hold time; answers questions about hours, services, prices, "
        "insurance and location from the practice's own information; books, reschedules and cancels appointments "
        "in the calendar; handles dental emergencies by finding the soonest opening and marking urgent callbacks; "
        "emails the practice the patient's details after every booking; and every call is saved with a short "
        "summary for the staff"
    ),
    "pricing": (
        "a monthly plan based on how many call minutes the practice needs, plus a one-time setup fee; "
        "the Boltix AI team gives an exact quote on a short call"
    ),
    "setup": (
        "the practice keeps its existing number and forwards calls to Mia; the team sets her up with the "
        "practice's hours, services, prices, insurance and calendar"
    ),
    "setup_time": "usually within a few days; the Boltix AI team confirms the timing for a practice on a call",
    "patient_texts": (
        "no: Mia does not send text messages to patients; the practice gets an email with the patient's details "
        "after every booking"
    ),
    "call_transfer": (
        "yes: for a practice, Mia can be set up to transfer calls to the front desk; in this demo there is no one "
        "to transfer to"
    ),
    "languages": "this demo speaks English; other languages can be discussed on a call",
    "trial_contract": "the Boltix AI team covers any free trial, contract length and cancellation on a call",
    "customization": "yes: a practice can choose Mia's name, voice and greeting",
    "practice_software": (
        "today Mia books into an online calendar; the team checks on a call whether she can connect to the "
        "practice's software"
    ),
    "privacy": "the team walks you through how patient data is handled on a call",
    "next_step": (
        "reply to the email from Boltix AI, or book a call at boltix A I dot com, or I can take your details and "
        "have the Boltix AI team call you back"
    ),
    "insurance_verification": (
        "Mia tells callers which insurance the practice accepts; checking a patient's exact coverage is something the "
        "Boltix AI team can go through on a call"
    ),
    "reminders": "the team can talk about that on a call",
    "call_summaries": (
        "every call is saved with a short summary for the practice team; the Boltix AI team can show you how on a call"
    ),
    "outages": "the Boltix AI team can walk you through how reliability is handled on a call",
    "specialties": (
        "Mia is built for dental practices; the Boltix AI team can talk about orthodontists or other specialties on a call"
    ),
    "clients": "the Boltix AI team can walk you through that on a call",
}
