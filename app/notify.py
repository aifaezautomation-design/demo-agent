"""Email the practice when Mia books an appointment (Resend).

Nothing here may ever slow down or break a call:
  * `notify_booking` only schedules a task and returns immediately (no awaiting, no I/O on the caller's path);
  * every failure is caught and logged, and a problem with email never reaches the voice pipeline;
  * it is off unless RESEND_API_KEY is set;
  * the tasks live in their own set, separate from the database writes the call waits on (`CallSession.flush`),
    so a slow email can never hold up a reply.
All patient-supplied text is HTML-escaped: a caller's name or reason becomes part of the email.
"""

import asyncio
import html
from dataclasses import dataclass
from datetime import datetime

import httpx
from loguru import logger

from app import config

RESEND_URL = "https://api.resend.com/emails"
_TIMEOUT = httpx.Timeout(8.0)
_tasks: set[asyncio.Task] = set()  # strong references, so a task is not garbage-collected mid-flight


@dataclass
class BookingEmail:
    practice: str
    patient_name: str
    patient_phone: str
    reason: str
    start: datetime  # timezone-aware, in the practice's time zone
    timezone_name: str  # e.g. America/Chicago
    is_emergency: bool = False
    is_new_patient: bool | None = None
    booking_uid: str = ""


def recipient_for(kb: dict) -> str | None:
    """A tenant can set `notify_email` in its knowledge base; otherwise the default address is used."""
    return (kb or {}).get("notify_email") or config.NOTIFY_EMAIL_TO or None


def _phone(raw: str) -> str:
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}" if len(digits) == 10 else raw


def _when(start: datetime, zone_name: str) -> tuple[str, str]:
    day = f"{start:%A, %B} {start.day}, {start.year}"
    clock = f"{start.hour % 12 or 12}:{start:%M %p}"
    return f"{day} at {clock}", f"{start.tzname() or ''} ({zone_name})".strip()


def render(b: BookingEmail) -> tuple[str, str, str]:
    """(subject, html, plain text)."""
    when, tz = _when(b.start, b.timezone_name)
    phone = _phone(b.patient_phone)
    subject = f"{'URGENT: ' if b.is_emergency else ''}New appointment: {b.patient_name}, {b.start:%a %b} {b.start.day}, {b.start.hour % 12 or 12}:{b.start:%M %p}"
    rows = [
        ("Patient", b.patient_name),
        ("Phone", phone),
        ("Service / reason", b.reason or "Not given"),
        ("Date and time", when),
        ("Time zone", tz),
    ]
    if b.is_new_patient is not None:
        rows.append(("Patient type", "New patient" if b.is_new_patient else "Returning patient"))
    e = html.escape
    cells = "".join(
        f'<tr><td style="padding:10px 16px;color:#6b7280;font-size:14px;border-bottom:1px solid #e5e7eb;white-space:nowrap">{e(k)}</td>'
        f'<td style="padding:10px 16px;color:#111827;font-size:15px;font-weight:600;border-bottom:1px solid #e5e7eb">{e(v)}</td></tr>'
        for k, v in rows
    )
    banner = (
        '<div style="background:#fef2f2;color:#991b1b;padding:10px 16px;border-radius:8px;margin-bottom:16px;font-weight:600">'
        "Emergency appointment: please prepare for a same-day patient.</div>"
        if b.is_emergency
        else ""
    )
    body = f"""<!doctype html>
<html><body style="margin:0;background:#f3f4f6;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
<div style="max-width:560px;margin:24px auto;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb">
  <div style="background:#0f766e;color:#ffffff;padding:20px 24px">
    <div style="font-size:13px;opacity:.85">{e(b.practice)}</div>
    <div style="font-size:20px;font-weight:700;margin-top:2px">New appointment booked</div>
  </div>
  <div style="padding:20px 24px">
    {banner}
    <table role="presentation" cellspacing="0" cellpadding="0" style="width:100%;border:1px solid #e5e7eb;border-radius:8px;border-collapse:separate;overflow:hidden">{cells}</table>
    <p style="color:#6b7280;font-size:12px;margin:16px 0 0">Booked by Mia, your AI receptionist, during a phone call. The booking is in your Cal.com calendar{f" (ref {e(b.booking_uid)})" if b.booking_uid else ""}.</p>
  </div>
</div>
</body></html>"""
    text = "New appointment booked at " + b.practice + "\n\n" + "\n".join(f"{k}: {v}" for k, v in rows)
    if b.is_emergency:
        text = "EMERGENCY APPOINTMENT\n" + text
    return subject, body, text


async def _post(payload: dict, idempotency_key: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        return await client.post(
            RESEND_URL,
            json=payload,
            headers={"Authorization": f"Bearer {config.RESEND_API_KEY}", "Idempotency-Key": idempotency_key},
        )


async def send(b: BookingEmail, to: str) -> bool:
    """Sends one email; returns whether it was accepted. Never raises."""
    subject, body, text = render(b)
    payload = {"from": config.NOTIFY_EMAIL_FROM, "to": [to], "subject": subject, "html": body, "text": text}
    key = f"booking-{b.booking_uid}" if b.booking_uid else f"booking-{b.patient_name}-{b.start.isoformat()}"
    for attempt in (1, 2):
        try:
            resp = await _post(payload, key)
        except Exception as exc:  # network error, timeout, anything: email must never hurt a call
            logger.warning(f"booking email attempt {attempt} failed to send: {exc!r}")
        else:
            if resp.status_code < 300:
                logger.info(f"booking email sent to {to} for {b.booking_uid or b.patient_name}")
                return True
            logger.warning(f"booking email attempt {attempt}: Resend answered {resp.status_code}: {resp.text[:300]}")
            if resp.status_code < 500 and resp.status_code != 429:
                return False  # a rejected request will be rejected again (bad key, unverified sender, ...)
        await asyncio.sleep(1.0)
    logger.error(f"booking email LOST for {b.booking_uid or b.patient_name}: Resend unavailable")
    return False


def notify_booking(b: BookingEmail, kb: dict | None = None) -> bool:
    """Fire and forget. Returns at once; True only if a send was scheduled."""
    to = recipient_for(kb or {})
    if not config.RESEND_API_KEY or not to:
        return False
    try:
        task = asyncio.get_running_loop().create_task(send(b, to))
    except Exception:
        logger.exception("could not schedule the booking email")
        return False
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True
