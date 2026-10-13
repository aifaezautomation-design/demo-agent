# Local testing

Work through these in order. Each step only needs the credentials listed for it, so you can test the database and calendar before you have any voice-provider keys. Run every command from the project root (`D:\demo agent`) in PowerShell.

## 0. Install dependencies

```powershell
.\.venv\Scripts\pip install -r requirements.txt pytest
.\.venv\Scripts\python -m pytest tests -q        # offline, needs no keys: expect 13 passed
```

## 1. Create `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` is in `.gitignore`. Never commit it, and never put the service-role key in dashboard or browser code.

| Variable | Where to find it | Needed for |
|---|---|---|
| `SUPABASE_URL` | Supabase → Project Settings → API → Project URL (`https://<ref>.supabase.co`) | everything |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Project Settings → API Keys → the **secret** key (`sb_secret_…`), or `service_role` on the Legacy tab. Not the publishable/anon key: RLS would hide every row. | everything |
| `OPENAI_API_KEY` | platform.openai.com → API keys | triage check, chat mode, live calls |
| `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`, `TELNYX_API_KEY`, `TELNYX_PUBLIC_KEY` | each provider's console | live calls only (step 6) |
| `WS_AUTH_TOKEN` | generate: `.\.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(32))"` | live calls only |
| `PUBLIC_HOST` | your tunnel's hostname, no `https://` | live calls only |

Leave `DEEPGRAM_MODEL=nova-3-general`, `OPENAI_MODEL=gpt-4o-mini`, and `OPENAI_TEMPERATURE=0.2` as they are. The Cal.com key does **not** go in `.env`: it lives in the database, per practice (step 3).

## 2. Apply `schema.sql`

1. Supabase dashboard → **SQL Editor** → **New query**.
2. Paste the entire contents of `schema.sql` and click **Run**. Expect `Success. No rows returned`.
3. Check the result:
   - **Table Editor** shows `clients`, `client_secrets`, `client_users`, `platform_admins`, `call_logs`, and `appointments`, and `clients` has the Apex Dental row.
   - **Storage** shows a private `recordings` bucket.

The script is safe to re-run. If the `read own recordings` policy errors with a permissions message, create that one policy under Storage → Policies instead; nothing else depends on it.

## 3. Connect a test calendar

Bookings made by the tests are **real** Cal.com bookings, so use a test Cal.com account or a dedicated test event type, not a practice's live calendar.

In Cal.com:

1. **Settings → Developer → API Keys**: create a key (`cal_live_…`).
2. Create an event type (e.g. "Dental Appointment", 45 min). Its numeric ID is in the URL: `app.cal.com/event-types/<ID>`.
3. On that event type:
   - **Advanced → Booking questions**: make **Email** optional, or hide it, and add Phone. Mia never collects email, so if Email stays required, every booking fails.
   - Turn off attendee emails and texts (Advanced settings and Workflows), so test patients don't get notifications.
   - **Limits → Minimum notice**: Cal.com defaults to 2 hours, which leaves no slots in Mia's 2-hour emergency window. Set it to about 15 minutes if you want emergency booking to find slots.
4. Make sure its availability schedule has open hours in the next 7 days.

Then in the Supabase SQL Editor (fill in your two values):

```sql
update public.clients
   set cal_com_event_type_id = 123456          -- your event type ID
 where id = 'apex-dental';

insert into public.client_secrets (client_id, cal_com_api_key)
values ('apex-dental', 'cal_live_xxx')         -- your Cal.com API key
on conflict (client_id) do update
   set cal_com_api_key = excluded.cal_com_api_key, updated_at = now();
```

Leave `phone_number` as the seeded `+15125550100` for now. You'll only change it when you hook up a real Telnyx number in step 6.

## 4. Run the call simulation

```powershell
.\.venv\Scripts\python scripts\simulate_call.py
```

The script runs the same tool code Mia uses on a live call, against your real Supabase and Cal.com:

| # | Step | Proves |
|---|---|---|
| 1 | Preflight | schema applied, tenant row and Cal.com key readable with the service key |
| 2 | Availability | Cal.com returns slots for your event type |
| 3 | Booking | a Cal.com booking plus an `appointments` row |
| 4 | Lookup | the booking is found again by caller ID |
| 5 | Reschedule | Cal.com reschedule plus a row update |
| 6 | Cancel | Cal.com cancel plus the row marked cancelled |
| 7 | Emergency | 2-hour emergency slot search, and the urgent-callback flag on the call log |
| 8 | Billing | 95 s bills as 1.58 min, `used_minutes` goes up by exactly that, and a second finalize is a no-op |
| 9 | LLM triage | with Mia's real prompt, a knocked-out tooth, airway swelling, and severe pain are handled as emergencies, and a cleaning isn't |

At the end, the script deletes its call log and refunds its minutes. Useful flags:

- `--keep`: leave the call log and billed minutes in Supabase, so you can inspect them in the Table Editor.
- `--book-emergency`: also book an emergency slot, then cancel it.
- `--skip-llm`: skip step 9.
- `--duration 300`: bill a different simulated length.

The script exits with code 0 only if every check passes.

**If something fails:**

| Message | Fix |
|---|---|
| `schema.sql has not been applied` | Step 2, in the same project as `SUPABASE_URL` |
| `no clients row with phone_number = …` | The `--number` you passed doesn't match `clients.phone_number` (must be E.164) |
| `…is the publishable key` / `has role 'anon'` | Use the secret/service-role key |
| `Cal.com 401` | Wrong or expired Cal.com key in `client_secrets` |
| `Cal.com returned open slots` fails | The event type has no availability in the next 7 days, or the event type ID is wrong |
| Booking fails, and a warning above it mentions email | Make Email optional on the event type (step 3) |
| A triage case fails once | LLM replies vary. Rerun it; if it fails repeatedly, the prompt needs work. |

## 5. Talk to Mia by typing

```powershell
.\.venv\Scripts\python scripts\simulate_call.py --chat
```

You play the patient. Mia uses the real prompt, tools, Cal.com, and Supabase, and prints each tool call and its LLM response time. When you type `quit` (or Mia ends the call), the call is summarized, billed, and logged like a real one, so it shows up in `call_logs`. Bookings you make here are real, so cancel them afterwards. Use `--caller +1…` to pretend to be a different patient.

This is the fastest way to tune the prompt: try rescheduling, canceling, emergencies, "are you a robot?", and off-topic questions.

## 6. Make a real phone call to your local machine

Telnyx sends calls to a public URL, and ngrok forwards that URL to the server on your PC:

```
your phone ─▶ Telnyx ─POST /telnyx/texml─▶ ngrok ─▶ localhost:8000 (uvicorn)
                     ─WSS  /ws/…  (audio)─▶ ngrok ─▶ localhost:8000
```

Every test call uses real Telnyx, Deepgram, OpenAI, and Cartesia time, and bills minutes to the tenant.

### 6.1 Get the voice keys

| Variable | Where |
|---|---|
| `DEEPGRAM_API_KEY` | console.deepgram.com → API Keys → Create a New API Key |
| `CARTESIA_API_KEY` | play.cartesia.ai → API Keys |
| `CARTESIA_VOICE_ID` | play.cartesia.ai → Voices: pick a warm US English female voice for Mia and copy its ID |
| `TELNYX_API_KEY` | portal.telnyx.com → Account Settings → Keys & Credentials → API Keys → Create API Key (`KEY…`) |
| `TELNYX_PUBLIC_KEY` | the same page → Public Key tab. Copy it now, but leave it out of `.env` until step 6.7. |

The server needs the Telnyx API key even for inbound calls: it uses it to hang up the call when Mia says goodbye.

### 6.2 Install and start ngrok

```powershell
winget install ngrok.ngrok              # or download from ngrok.com/download
ngrok config add-authtoken <token>      # dashboard.ngrok.com → Your Authtoken
```

Free accounts include one **static domain** (dashboard.ngrok.com → Domains). Use it, so the URL doesn't change every time you restart ngrok and you don't have to update Telnyx and `.env` each time:

```powershell
ngrok http 8000 --url=<your-name>.ngrok-free.app
```

Leave this window open. ngrok's request inspector at **http://127.0.0.1:4040** shows every webhook Telnyx sends and every response the server returns. It's the first place to look when a call misbehaves.

### 6.3 Fill in `.env`

```
PUBLIC_HOST=<your-name>.ngrok-free.app     # host only: no https://, no trailing slash
WS_AUTH_TOKEN=<output of the command below>
TELNYX_API_KEY=KEY...
TELNYX_PUBLIC_KEY=                         # leave empty for the first call
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
CARTESIA_VOICE_ID=...
```

```powershell
.\.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 6.4 Set up Telnyx

1. **Buy a number:** Numbers → Search & Buy → a US local number with Voice enabled.
2. **Create a TeXML application:** Voice → Programmable Voice → TeXML Applications → Add new.
   - Voice webhook URL: `https://<your-name>.ngrok-free.app/telnyx/texml`
   - Method: **POST**
   - Save.
3. **Attach the number:** Numbers → My Numbers → your number → set its connection/application to the TeXML application you just made. Save.

You don't need an outbound voice profile for inbound calls. The Telnyx portal reorganizes its menus from time to time; if a label differs, look for "TeXML" under Voice.

### 6.5 Point the tenant at your number

In the Supabase SQL Editor, using the full E.164 number you bought:

```sql
update public.clients set phone_number = '+1XXXXXXXXXX' where id = 'apex-dental';
```

`scripts\simulate_call.py` defaults to `+15125550100`, so pass `--number +1XXXXXXXXXX` from now on.

### 6.6 Start the server and call

```powershell
.\.venv\Scripts\python -m uvicorn app.server:app --port 8000
```

Check that it's reachable through the tunnel: open `https://<your-name>.ngrok-free.app/health` in a browser, click past ngrok's warning page, and you should see `{"ok":true}`. Telnyx never sees that warning page, because it only appears for browsers.

At startup you'll see `TELNYX_PUBLIC_KEY is not set: Telnyx webhooks are NOT verified`. That's expected for this first call.

Now call your Telnyx number from your phone. In the server window you should see, in order:

1. `inbound call for apex-dental: streaming to wss://…`: Telnyx reached the webhook.
2. `call start: client=apex-dental from=+1… encoding=PCMU`: the audio stream connected.
3. Pipecat pipeline logs, then Mia's greeting on your phone.
4. After you hang up: `call closed: 42s, billed 0.7 min, booked=False`.

The call then appears in `call_logs` with its transcript and summary. Try booking, rescheduling, canceling, and an emergency. Expect a short pause before the greeting while the voice models load.

Restart uvicorn after any change to `.env` or the `clients` table. The server reads `.env` once at startup and caches each tenant for 30 seconds.

### 6.7 Turn on webhook verification

1. Paste the public key from 6.1 into `TELNYX_PUBLIC_KEY`, restart uvicorn, and the startup warning disappears.
2. Call again. If the call goes through, signatures are verified.
3. If Telnyx's webhook gets **401** instead (in the ngrok inspector, with `rejected Telnyx webhook: …` in the server log), the reason is in that log line. `stale timestamp` means your PC's clock is off: sync it in Windows Settings → Time. `bad signature` means the key is wrong, or TeXML webhooks are signed differently than Telnyx's other webhooks. Tell me the exact log line and I'll adjust the check.

Don't skip this step. Without verification, anyone who finds your ngrok URL can request the stream token and open AI sessions billed to the tenant.

### Troubleshooting

| Symptom | Likely cause | Where to look |
|---|---|---|
| Fast busy, or the call is rejected immediately | No request reached the server (wrong webhook URL, ngrok not running), or `inbound call to unknown number` (the dialed number isn't in `clients`) | ngrok inspector; server log |
| "We're unable to take your call right now" | Tenant is out of minutes or not `active` | `clients.used_minutes`, `status` |
| "We're having technical difficulties" | The Supabase lookup failed | server log traceback |
| Call connects, then silence or a hang-up | The audio stream didn't connect. `403` on `/ws/…` means `PUBLIC_HOST` or `WS_AUTH_TOKEN` changed without a restart. | ngrok inspector (look for the `/ws/…` request) |
| Mia greets you but never responds | Deepgram key or model | server log, look for Deepgram errors |
| Logs show conversation turns but you hear nothing | Cartesia key or voice ID | server log |
| Mia doesn't hang up after her goodbye | `TELNYX_API_KEY` is wrong, so the hang-up API call fails | server log |

When you're done testing, stop ngrok (Ctrl+C). The URL is public for as long as it runs.
