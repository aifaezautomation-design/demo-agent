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
- **Patients are never emailed or texted.** (The practice does get one staff notification email per booking, see below.) The booking API version `2026-02-25` makes attendee email optional, and none is sent. Also turn off attendee notifications on each Cal.com event type.
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

## How Mia behaves

- **English only, by default.** Speech recognition is Deepgram `nova-3` in English, the voice is Cartesia `sonic-3.6` (the
  fastest current Cartesia model; `sonic-english` has been shut down and `sonic-2` / `sonic-turbo` stop working on
  2026-10-20), and none of the multilingual machinery is loaded. `MULTILINGUAL_ENABLED=1` brings it back (language
  detection, a second Deepgram connection for Urdu, the `set_language` tool, localized spoken phrases; see `app/language.py`).
- **Dates are resolved in code.** The model passes the caller's own words ("next thursday", "friday next week", "the
  26th") and `resolve_day` in `app/tools.py` works out the date from the practice's clock. `check_availability` also takes a
  `preferred_time`, so Mia can say truthfully whether 2 PM is open.
- **Opening hours are enforced in code.** Cal.com decides what is bookable, and its schedule can disagree with the clinic's
  real hours (it did). `hours_schedule` in the tenant's `knowledge_base` is checked for every slot, so nothing outside
  opening hours is ever offered or booked. Keep the Cal.com schedule in line with it too.
- **Bookings use what the caller said.** `book_appointment` refuses a name or phone number the caller never gave
  (placeholders like "John Doe" and invented numbers included).
- **Emergencies are handled by code first (`app/emergency.py`).** Severe pain, swelling, bleeding, a knocked-out or broken
  tooth, or an abscess: code speaks a caring line (with first aid when it applies) and the 911-if-breathing line, finds the
  soonest same-day slot, states it, and hands the LLM only the job of asking for a name and booking. Mia never sends the
  caller to another dentist, urgent care, or the ER. The one exception is trouble breathing or swallowing: code tells
  the caller to call 911 and flags an urgent callback. Phrases the detector misses still go to the LLM and its prompt rules.
- **The clinic's knowledge lives in the database** (`clients.knowledge_base`): address, hours, dentists, services and prices,
  insurance. `data/apex_dental_care.json` is the Apex Dental Care record used for the schema seed and tests.

## Settings you may want to change (Railway variables or `.env`)

| Variable | Default | What it does |
|---|---|---|
| `DEEPGRAM_STT` | `flux` | `flux` = Deepgram Flux (English): it ends the caller's turn itself and starts the LLM while the last words are still being finalised, about 0.25-0.3 s faster on ordinary turns (no change on calendar-lookup turns). `nova` = Deepgram nova-3 with Pipecat's own turn detection: the one-line rollback |
| `FLUX_EAGER_EOT_THRESHOLD` / `FLUX_EOT_THRESHOLD` | `0.5` / `0.7` | How sure Flux must be to start the LLM early / to end the turn. More aggressive values made no difference in testing |
| `FLUX_EOT_TIMEOUT_MS` | `2000` | Longest silence before the turn ends whatever the confidence (Flux's own default is 5000) |
| `OPENAI_MODEL` / `OPENAI_TEMPERATURE` / `OPENAI_MAX_TOKENS` | `gpt-4o-mini` / `0.3` / `150` | Real replies average about 30 tokens and peak near 65, so the bound only cuts off a runaway answer. Temperature does not change latency |
| `MULTILINGUAL_ENABLED` | `0` | `1` turns on language detection and Urdu switching (and uses nova-3, not Flux) |
| `CARTESIA_VOLUME` | `1.4` | Voice volume, 0.5 to 2.0 (clips above about 1.5) |
| `CARTESIA_STREAM_TOKENS` | `0` | Token-by-token text to the voice. Measured: it made the first audio *slower* (0.58 s vs 0.33 s), so leave it off |
| `VAD_STOP_SECS` | `0.2` | Silence before voice-activity detection ends a turn (nova only) |

## Latency: what was measured

Per-stage times from real Railway calls: speech recognition about 0.33 s, LLM first token about 0.65 s (the largest single piece),
voice first byte about 0.12 s. Findings from `scripts/latency_benchmark.py` (phone-quality 8 kHz audio, relative numbers):

- Flux cut the median end-of-speech-to-first-audio time by 0.25-0.3 s on ordinary turns; calendar turns are unchanged, because
  Pipecat cancels a speculative answer that wants a tool and re-runs it once the turn is confirmed.
- Removing *all* ~2,400 prompt and tool tokens saves only about 100 ms of LLM first-token time, so the prompt is not worth
  trimming further at the risk of the dental rules.
- Bookkeeping writes (appointment rows, urgent-callback flags, cancel and reschedule records) and the Resend email all run behind
  the reply; the Cal.com request itself (0.8 s to look up, about 2 s to book) is the remaining cost of a tool turn.
- Sub-800 ms from the end of the caller's speech is not reachable with `gpt-4o-mini` on this stack: LLM first token plus voice
  first byte alone are about 0.8-0.9 s. Roughly 1.2-1.5 s is realistic for an ordinary turn.
