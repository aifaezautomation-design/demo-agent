# Boltex AI — Dental Voice Agent (Phase 1)

"Mia", an inbound phone receptionist for dental practices. She books, reschedules, and cancels appointments, triages emergencies, writes to Cal.com, and logs every call to Supabase. One deployment serves every practice; the dialed number decides which practice a call belongs to.

```
Caller ─PSTN─▶ Telnyx ──POST /telnyx/texml──▶ tenant lookup by dialed number
                  │                            (unknown → reject, out of minutes → polite hang-up)
                  └──WS /ws (8 kHz PCMU)──▶ Pipecat pipeline, one per call
                        Deepgram STT → Silero VAD + Smart Turn → gpt-4o-mini + tools → Cartesia TTS
                                                                      │
                                                         Cal.com v2 ◀─┴─▶ Supabase
```

## Layout

| Path | Purpose |
|---|---|
| `app/server.py` | FastAPI: TeXML webhook, media WebSocket, recording callback, Telnyx signature check |
| `app/bot.py` | Per-call Pipecat pipeline, transcript capture, idle and max-duration guards, billing on hang-up |
| `app/tools.py` | LLM tools and their guardrails (see below) |
| `app/prompts.py` | Mia's system prompt, built per call from the practice's knowledge base |
| `app/calcom.py` | Cal.com v2 client (slots, create, reschedule, cancel) |
| `app/db.py` | Supabase access with a 30-second tenant cache |
| `schema.sql` | Tables, RLS, atomic billing function, recordings bucket, Apex Dental seed |

## Setup

1. **Supabase.** Run `schema.sql` in the SQL editor. Store the practice's Cal.com key server-side only:
   ```sql
   insert into client_secrets (client_id, cal_com_api_key) values ('apex-dental', 'cal_live_...');
   update clients set cal_com_event_type_id = 123456, phone_number = '+1512...' where id = 'apex-dental';
   ```
   Enable `pg_cron` and schedule the monthly reset (the command is in the comment above `reset_billing_cycles()`).
2. **Cal.com.** Create an event type per practice (and optionally a separate emergency event type), and note its numeric ID.
3. **Railway.** Deploy this repo (it builds the Dockerfile). Set the variables from `.env.example`. In service settings, pick the US East (Virginia) region and set a generous deploy draining time so redeploys don't cut live calls.
4. **Telnyx.** Create a TeXML application with the voice webhook `https://<PUBLIC_HOST>/telnyx/texml` (POST), assign the practice's number to it, and copy your account public key into `TELNYX_PUBLIC_KEY`.
5. **Dashboard login.** For each practice user: `insert into client_users (user_id, client_id, role) values ('<auth uuid>', 'apex-dental', 'owner');`. Boltex staff go in `platform_admins`.

Run locally: `uvicorn app.server:app --port 8000`, exposed with a tunnel (ngrok or cloudflared) and `PUBLIC_HOST` set to the tunnel's host.

Tests: `python -m pytest tests` (no network or credentials needed). For testing against real Supabase and Cal.com, including a typed chat with Mia, see [LOCAL_TESTING.md](LOCAL_TESTING.md).

## Onboarding a new practice

Insert a `clients` row (number, timezone, Cal.com event type, `knowledge_base` JSON in the same shape as the Apex seed), a `client_secrets` row, and point the Telnyx number at the TeXML app. No code changes or redeploys are needed.

## Design decisions

- **Guardrails live in code, not just the prompt.** `book_appointment` and `reschedule_appointment` only accept slot IDs that `check_availability` returned during the current call. Reschedule and cancel only act on appointments found during the call, and require the last four digits of the phone on file when it differs from the caller ID.
- **Appointments are mirrored in Supabase.** Cal.com v2 can't search bookings by phone, so lookups use the `appointments` table. Calendar writes plus their database records are shielded from barge-in cancellation, so an interruption can't leave the two out of sync.
- **Billing is atomic and idempotent.** `finalize_call()` closes the call log and deducts minutes in one transaction, and runs at most once per call. Calls to a practice that is out of minutes are refused at the TeXML step, before any AI cost.
- **No email or SMS is sent.** The booking API version `2026-02-25` makes attendee email optional, and none is sent. Also turn off attendee notifications on each Cal.com event type.
- **Recording is opt-in per practice** (`clients.record_calls`). When it's on, Mia's greeting discloses it, because recording consent laws differ by state. Files are copied into a private Supabase bucket, since Telnyx recording links expire.
- **The Boltex pitch only plays in demo mode** (`clients.demo_mode`). Production practices get an honest AI disclosure without a sales pitch.

## Latency

Responsiveness comes from several choices:

- The greeting goes straight to TTS, with no LLM call first.
- Audio stays at 8 kHz end to end.
- HTTP connections to Cal.com stay warm.
- End of turn is decided by Smart Turn plus a 200 ms VAD stop, not by a fixed silence wait.
- Models are baked into the image.
- LLM output is capped at 200 tokens.

Expect roughly **0.8–1.2 s caller-perceived response time** over the phone network. Telnyx transit, end-of-turn detection, and LLM time-to-first-token add up past 500 ms even when every component is fast. Measure with `enable_metrics` (already on) before quoting a number to clients. Keep the system prompt short: every token adds to time-to-first-token.

## Before real patient data

Transcripts and recordings are PHI. You need BAAs with Telnyx, Deepgram, OpenAI, Cartesia, and Supabase (HIPAA add-on), and Railway's terms need checking. Until those are in place, use demo tenants only.

## Languages and dates

- **Languages:** English, Spanish, French, German, Portuguese, Italian, Hindi, Urdu. Speech recognition starts in
  Deepgram nova-3 `multi`, which reports each utterance's language; the voice (Cartesia sonic-3.6, same voice for every
  language) switches before the LLM replies. Urdu is not in `multi` (it is heard as Hindi), so it runs on a second
  Deepgram service behind a `ServiceSwitcher`, selected by Urdu vocabulary in the transcript or by the `set_language`
  tool. Hindi and Urdu speech are the same language to a recognizer: a caller who uses no Urdu-specific words is
  answered in Hindi until they ask for Urdu. The spoken phrases in `app/language.py` should be reviewed by native speakers.
  Set `DEEPGRAM_LANGUAGE=en` to turn multilingual recognition off.
- **Dates:** the model passes the caller's own words ("next thursday", "friday next week", "the 26th") and code resolves
  them against the practice's clock (`resolve_day` in `app/tools.py`); the model never does calendar arithmetic.
