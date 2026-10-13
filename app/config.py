"""Environment-driven settings. Loaded once at import."""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# Public URL Railway assigns, e.g. "voice.boltex.ai" (no scheme).
PUBLIC_HOST = _require("PUBLIC_HOST")
# Shared secret embedded in the media-stream URL so only our TeXML can open /ws.
WS_AUTH_TOKEN = _require("WS_AUTH_TOKEN")

TELNYX_API_KEY = _require("TELNYX_API_KEY")
# Base64 Ed25519 key from the Telnyx portal. When set, webhooks are verified.
TELNYX_PUBLIC_KEY = os.getenv("TELNYX_PUBLIC_KEY")

DEEPGRAM_API_KEY = _require("DEEPGRAM_API_KEY")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3-general")

OPENAI_API_KEY = _require("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

CARTESIA_API_KEY = _require("CARTESIA_API_KEY")
CARTESIA_VOICE_ID = _require("CARTESIA_VOICE_ID")

SUPABASE_URL = _require("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _require("SUPABASE_SERVICE_ROLE_KEY")

CALCOM_BASE_URL = os.getenv("CALCOM_BASE_URL", "https://api.cal.com")

# Hard ceiling per call so a stuck line cannot drain a tenant's minute pool.
MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", "900"))
