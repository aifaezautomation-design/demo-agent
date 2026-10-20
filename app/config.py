"""Environment-driven settings. Loaded once at import."""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# Public URL Railway assigns, e.g. "voice.boltixai.com" (no scheme).
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
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

CARTESIA_API_KEY = _require("CARTESIA_API_KEY")
CARTESIA_VOICE_ID = _require("CARTESIA_VOICE_ID")

SUPABASE_URL = _require("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _require("SUPABASE_SERVICE_ROLE_KEY")

CALCOM_BASE_URL = os.getenv("CALCOM_BASE_URL", "https://api.cal.com")

# Hard ceiling per call so a stuck line cannot drain a tenant's minute pool.
MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", "900"))

# --- Latency tuning (override via env to A/B test without code changes) ---
# Send LLM text to Cartesia token by token instead of whole sentences. Off by default: no gain
# measured locally and it can change prosody. Try on Railway: CARTESIA_STREAM_TOKENS=1 CARTESIA_MAX_BUFFER_MS=100
CARTESIA_STREAM_TOKENS = os.getenv("CARTESIA_STREAM_TOKENS", "0") == "1"
# Cartesia's server-side buffer before it starts speaking in token mode (ms; unset = Cartesia's 3000).
_buf = os.getenv("CARTESIA_MAX_BUFFER_MS", "")
CARTESIA_MAX_BUFFER_MS = int(_buf) if _buf else None
# Deepgram's silence threshold for finalizing a transcript (ms; unset = Deepgram's default of 10).
_ep = os.getenv("DEEPGRAM_ENDPOINTING_MS", "")
DEEPGRAM_ENDPOINTING_MS = int(_ep) if _ep else None
# Silence before voice-activity detection ends a turn (seconds); Smart Turn then confirms.
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.2"))
# Smart Turn judges from the audio whether the caller is done. When it says "not yet" (a hesitant "I think... today"), it waits
# this long in silence before ending the turn anyway. Pipecat's default is 3 s, which a real caller felt as a 5 s delay.
SMART_TURN_STOP_SECS = float(os.getenv("SMART_TURN_STOP_SECS", "1.0"))
# If a turn gets cut by a late transcript fragment and no stop signal follows, how long until it is closed (default 5 s).
USER_TURN_STOP_TIMEOUT = float(os.getenv("USER_TURN_STOP_TIMEOUT", "1.5"))
# Seconds of caller silence before "Are you still there?"; a second stretch ends the call. Never counted during a tool call.
# Seconds after a caller's turn with no reply of any kind before the model is re-prompted.
STALL_SECS = float(os.getenv("STALL_SECS", "6"))
USER_IDLE_SECS = float(os.getenv("USER_IDLE_SECS", "20"))

# Cartesia volume multiplier (0.5-2.0, 1.0 = normal). The voice already peaks near full scale,
# so above ~1.5 speech starts to clip and sound distorted: 1.25 -> 0.1%, 1.5 -> 0.8%, 2.0 -> 3.6%.
CARTESIA_VOLUME = min(max(float(os.getenv("CARTESIA_VOLUME", "1.4")), 0.5), 2.0)

# Speech-to-text language. "multi" = Deepgram nova-3 multilingual (English, Spanish, French, German,
# Portuguese, Italian, Hindi, ...) with per-utterance language detection; Urdu is switched in by the
# set_language tool. Set to "en" to run English-only.
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "multi" if os.getenv("MULTILINGUAL_ENABLED") == "1" else "en")

# English only by default: no language detection, no second speech-to-text connection, and no language rules or
# set_language tool in the prompt. "1" turns the multilingual path back on (see app/language.py and the README).
MULTILINGUAL_ENABLED = os.getenv("MULTILINGUAL_ENABLED", "0") == "1"

# Speech-to-text engine for English calls. "nova" = Deepgram nova-3 with Pipecat's VAD + Smart Turn deciding when the
# caller is done. "flux" = Deepgram Flux (English-only), which predicts the end of the turn itself and lets the LLM
# start answering before the transcript is final.
DEEPGRAM_STT = os.getenv("DEEPGRAM_STT", "nova")

# Booking notification emails (Resend). Off unless RESEND_API_KEY is set. A tenant can override the recipient with
# `notify_email` in its knowledge base. Resend only lets an unverified sender (onboarding@resend.dev) email the
# address that owns the Resend account; verify a domain to send to anyone.
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO", "")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM", "Mia Receptionist <onboarding@resend.dev>")

# Where demo-line emails go: one per callback request and one after every demo call (summary and transcript).
# Empty (or no RESEND_API_KEY) turns them off. Only used when the practice has demo_mode on.
BOLTIX_LEADS_EMAIL = os.getenv("BOLTIX_LEADS_EMAIL", "")

# Reply length bound, in tokens. Real calls average ~30 and peak ~65; a spoken turn is one or two sentences, so this
# only ever cuts off a runaway answer.
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "150"))

# Deepgram Flux turn detection (DEEPGRAM_STT=flux). Flux predicts the end of the caller's turn itself.
#   eager threshold: how sure it must be to START the LLM early (lower = earlier, more discarded guesses)
#   eot threshold:   how sure it must be to END the turn
#   eot timeout:     the longest silence after which the turn ends whatever the confidence (Flux's default is 5 s)
FLUX_EAGER_EOT_THRESHOLD = float(os.getenv("FLUX_EAGER_EOT_THRESHOLD", "0.5"))
FLUX_EOT_THRESHOLD = float(os.getenv("FLUX_EOT_THRESHOLD", "0.7"))
FLUX_EOT_TIMEOUT_MS = int(os.getenv("FLUX_EOT_TIMEOUT_MS", "2000"))
