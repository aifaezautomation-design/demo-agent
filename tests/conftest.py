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
