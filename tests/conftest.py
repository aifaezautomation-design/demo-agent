import os

# Dummy credentials so app.config imports; no test touches the network.
for name in [
    "WS_AUTH_TOKEN", "TELNYX_API_KEY", "DEEPGRAM_API_KEY", "OPENAI_API_KEY",
    "CARTESIA_API_KEY", "CARTESIA_VOICE_ID", "SUPABASE_SERVICE_ROLE_KEY",
]:
    os.environ.setdefault(name, f"test-{name.lower()}")
os.environ.setdefault("PUBLIC_HOST", "voice.test")
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ["TELNYX_PUBLIC_KEY"] = ""
# Tests load the real .env too: never let a test booking send a real notification email. tests/test_notify.py
# turns it back on with a fake transport where it needs to.
os.environ["RESEND_API_KEY"] = ""


# Shared fixture: a call session with a fake calendar and fake database (defined in test_agent.py).
from tests.test_agent import session  # noqa: E402,F401
