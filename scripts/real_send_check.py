"""Send ONE real, clearly labelled test notification email through the app's own sender.

Use it to check the Resend setup (key, verified sender domain, recipient) after changing .env or Railway variables:

    .venv\\Scripts\\python scripts\\real_send_check.py

The key comes from .env and is never printed.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from loguru import logger  # noqa: E402

from app import config, notify  # noqa: E402

logger.remove()
logger.add(sys.stdout, format="{level: <8} {message}")

TZ = "America/Chicago"


def main() -> int:
    print(f"recipient: {config.NOTIFY_EMAIL_TO or '(none)'} | sender: {config.NOTIFY_EMAIL_FROM} | key set: {bool(config.RESEND_API_KEY)}")
    if not config.RESEND_API_KEY or not config.NOTIFY_EMAIL_TO:
        print("RESULT: not configured (set RESEND_API_KEY and NOTIFY_EMAIL_TO in .env)")
        return 1
    notice = notify.BookingEmail(
        practice="Apex Dental Care",
        patient_name="TEST - Jane Example",
        patient_phone="+15125550123",
        reason="TEST EMAIL: routine cleaning (no real patient)",
        start=datetime(2026, 10, 22, 14, 0, tzinfo=ZoneInfo(TZ)),
        timezone_name=TZ,
        is_new_patient=True,
        booking_uid="test-notification-001",
    )
    if asyncio.run(notify.send(notice, config.NOTIFY_EMAIL_TO)):
        print(f"RESULT: accepted by Resend. Check the inbox of {config.NOTIFY_EMAIL_TO}.")
        return 0
    print("RESULT: NOT sent.")
    print("  If Resend said 403 'You can only send testing emails to your own email address', verify a domain at")
    print("  resend.com/domains and set NOTIFY_EMAIL_FROM to an address on it, e.g. Mia <mia@yourdomain.com>.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
