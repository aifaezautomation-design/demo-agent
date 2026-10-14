"""HTTP entry points.

Call flow:
  1. Telnyx POSTs /telnyx/texml when a call arrives. We resolve the tenant by
     the dialed number and either reject the call or return TeXML that opens a
     bidirectional media stream to /ws.
  2. Telnyx connects to /ws; we run one Pipecat pipeline for the call.
  3. If the tenant records calls, Telnyx POSTs /telnyx/recording when the
     recording is ready and we copy it into Supabase Storage.
"""

import base64
import hmac
import json
import os
import time
from urllib.parse import parse_qs, quote, urlencode
from xml.sax.saxutils import escape, quoteattr

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import Response
from loguru import logger
from pipecat.runner.utils import parse_telephony_websocket

from app import config, db
from app.bot import run_bot
from app.tools import normalize_phone

app = FastAPI(title="Boltex Dental Voice Agent")

if not config.TELNYX_PUBLIC_KEY:
    # Without signature checks, anyone who finds this URL can fetch TeXML
    # (which contains the stream token) and open AI sessions on a tenant's minutes.
    logger.warning("TELNYX_PUBLIC_KEY is not set: Telnyx webhooks are NOT verified. Never run like this in production.")

_SIGNATURE_TOLERANCE_SECS = 300


def _texml(body: str) -> Response:
    return Response(
        content=f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>',
        media_type="application/xml",
    )


async def _verified_body(request: Request) -> bytes:
    """Reads the body, enforcing Telnyx's Ed25519 signature when a key is configured."""
    raw = await request.body()
    if not config.TELNYX_PUBLIC_KEY:
        return raw
    signature = request.headers.get("telnyx-signature-ed25519")
    timestamp = request.headers.get("telnyx-timestamp")
    if not signature or not timestamp:
        return _reject("missing signature headers")
    try:
        if abs(time.time() - int(timestamp)) > _SIGNATURE_TOLERANCE_SECS:
            return _reject("stale timestamp (is this machine's clock correct?)")
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(config.TELNYX_PUBLIC_KEY))
        key.verify(base64.b64decode(signature), f"{timestamp}|".encode() + raw)
    except (InvalidSignature, ValueError):
        return _reject("bad signature (check TELNYX_PUBLIC_KEY)")
    return raw


def _reject(reason: str):
    logger.warning(f"rejected Telnyx webhook: {reason}")
    raise HTTPException(status_code=401, detail=reason)


def _parse_params(request: Request, raw: bytes) -> dict[str, str]:
    """TeXML webhooks are form-encoded by default but may be JSON."""
    if request.headers.get("content-type", "").startswith("application/json"):
        data = json.loads(raw or b"{}")
        return {k: str(v) for k, v in data.items() if not isinstance(v, (dict, list))}
    return {k: v[0] for k, v in parse_qs(raw.decode()).items()}


if os.getenv("ENABLE_TEST_UI") == "1":
    from app.webtest import mount_test_ui

    mount_test_ui(app)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/telnyx/texml")
async def texml(request: Request):
    params = _parse_params(request, await _verified_body(request))
    to_number = normalize_phone(params.get("To"))
    call_sid = params.get("CallSid", "")

    try:
        tenant = await db.get_tenant_by_number(to_number) if to_number else None
    except Exception:
        # A 500 here makes the caller hear an error tone; apologize instead.
        logger.exception(f"tenant lookup failed for {to_number}")
        return _texml("<Say>Sorry, we're having technical difficulties. Please call back in a few minutes.</Say><Hangup/>")
    if not tenant:
        logger.warning(f"inbound call to unknown number {to_number}")
        return _texml("<Reject/>")
    if not tenant.can_take_calls:
        # Out of minutes or paused: don't start an AI session the tenant can't pay for.
        logger.warning(f"{tenant.id} cannot take calls (status={tenant.status}, remaining={tenant.remaining_minutes})")
        return _texml(
            f"<Say>Thank you for calling {escape(tenant.name)}. We're unable to take your call right now. "
            "Please try again later.</Say><Hangup/>"
        )

    # Everything the stream needs rides in the path: Telnyx doesn't document
    # whether it keeps query strings on stream URLs, and path segments always survive.
    path = "/".join(quote(part, safe="") for part in (config.WS_AUTH_TOKEN, to_number, call_sid or "-"))
    stream_url = f"wss://{config.PUBLIC_HOST}/ws/{path}"
    logger.info(f"inbound call for {tenant.id}: streaming to wss://{config.PUBLIC_HOST}/ws/…")
    recording = ""
    if tenant.record_calls:
        callback = f"https://{config.PUBLIC_HOST}/telnyx/recording?{urlencode({'call_sid': call_sid})}"
        recording = (
            f'<Start><Recording channels="dual" format="mp3" '
            f"recordingStatusCallback={quoteattr(callback)} recordingStatusCallbackEvent=\"completed\"/></Start>"
        )
    # <Connect> holds the call for as long as the stream is open. Reconnects are
    # off: a reconnected stream would start a second pipeline with no context.
    return _texml(
        f"{recording}<Connect><Stream url={quoteattr(stream_url)} "
        'bidirectionalMode="rtp" enableReconnect="false"/></Connect>'
    )


@app.websocket("/ws/{token}/{to_number}/{call_sid}")
async def media_stream(websocket: WebSocket, token: str, to_number: str, call_sid: str):
    if not hmac.compare_digest(token, config.WS_AUTH_TOKEN):
        logger.warning("media stream rejected: bad token")
        await websocket.close(code=1008)
        return
    await websocket.accept()

    transport_type, call_data = await parse_telephony_websocket(websocket)
    if transport_type != "telnyx":
        logger.error(f"unexpected transport {transport_type}")
        await websocket.close(code=1003)
        return

    # The dialed number comes from our own TeXML, already resolved once.
    tenant = await db.get_tenant_by_number(to_number)
    if not tenant:
        logger.error(f"media stream for unknown number {to_number}")
        await websocket.close(code=1008)
        return

    logger.info(f"call start: client={tenant.id} from={call_data.from_number} encoding={call_data.outbound_encoding}")
    await run_bot(websocket, call_data, tenant, None if call_sid == "-" else call_sid)


@app.post("/telnyx/recording")
async def recording_ready(request: Request, background: BackgroundTasks):
    params = _parse_params(request, await _verified_body(request))
    call_sid = request.query_params.get("call_sid") or params.get("CallSid")
    url = params.get("RecordingUrl") or params.get("PublicRecordingUrl")
    if not call_sid or not url:
        logger.warning(f"recording callback missing fields: {sorted(params)}")
        return {"ok": False}
    # Telnyx retries slow webhooks; acknowledge now, copy the file afterwards.
    background.add_task(_store_recording, call_sid, url)
    return {"ok": True}


async def _store_recording(call_sid: str, url: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code in (401, 403):
                resp = await client.get(url, headers={"Authorization": f"Bearer {config.TELNYX_API_KEY}"})
            resp.raise_for_status()
        if not await db.save_recording(call_sid, resp.content):
            logger.warning(f"no call log for recording {call_sid}")
    except Exception:
        logger.exception(f"failed to store recording for {call_sid}")
