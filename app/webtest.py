"""Local browser test page: talk to Mia with your microphone.

Runs the same pipeline as a phone call (Deepgram, gpt-4o-mini, Cartesia, Cal.com,
Supabase); only the phone line is replaced by WebRTC. Enabled with
ENABLE_TEST_UI=1 and served only to requests addressed to localhost, so the
ngrok URL can never reach it (it would spend your API credits and make real
bookings).
"""

import os
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from loguru import logger

from app import db
from app.bot import run_call

TEST_TENANT_ID = os.getenv("TEST_TENANT_ID", "apex-dental")
TEST_CALLER_PHONE = os.getenv("TEST_CALLER_PHONE", "+15125550199")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _local_only(request: Request) -> None:
    # ngrok forwards the public hostname as Host and adds X-Forwarded-For.
    if request.url.hostname not in _LOCAL_HOSTS or request.headers.get("x-forwarded-for"):
        raise HTTPException(status_code=404)


def mount_test_ui(app: FastAPI) -> None:
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
    from pipecat.transports.smallwebrtc.request_handler import (
        SmallWebRTCPatchRequest,
        SmallWebRTCRequest,
        SmallWebRTCRequestHandler,
    )
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    from pipecat_ai_prebuilt.frontend import PipecatPrebuiltUI

    handler = SmallWebRTCRequestHandler()

    @app.get("/test", include_in_schema=False)
    async def test_page(request: Request):
        _local_only(request)
        return RedirectResponse(url="/client/")

    sessions: set[str] = set()

    @app.get("/status", include_in_schema=False)
    async def status(request: Request):
        _local_only(request)
        return {"status": "ready", "transports": ["webrtc"]}

    @app.post("/start", include_in_schema=False)
    async def start(request: Request):
        # The prebuilt client opens a session here, then sends its WebRTC offer to it.
        _local_only(request)
        session_id = str(uuid.uuid4())
        sessions.add(session_id)
        return {"sessionId": session_id, "iceConfig": {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}}

    async def _offer(request: Request, body: SmallWebRTCRequest, background: BackgroundTasks):
        _local_only(request)

        async def on_connection(connection: SmallWebRTCConnection):
            tenant = await db.get_tenant_by_id(TEST_TENANT_ID)
            if tenant is None:
                logger.error(f"test UI: no tenant '{TEST_TENANT_ID}'")
                return
            transport = SmallWebRTCTransport(
                webrtc_connection=connection,
                params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
            )
            logger.info(f"browser test call: tenant={tenant.id} caller={TEST_CALLER_PHONE}")
            background.add_task(
                run_call, transport, tenant,
                caller_phone=TEST_CALLER_PHONE, call_control_id=None, call_sid=None, sample_rate=int(os.getenv("TEST_UI_SAMPLE_RATE", "16000")),
                rtvi=True,  # the prebuilt browser client needs RTVI
            )

        return await handler.handle_web_request(request=body, webrtc_connection_callback=on_connection)

    async def _ice(request: Request, body: SmallWebRTCPatchRequest):
        _local_only(request)
        await handler.handle_patch_request(body)
        return {"status": "success"}

    # Older clients post to /api/offer; the current one uses /sessions/<id>/api/offer.
    app.post("/api/offer", include_in_schema=False)(_offer)
    app.patch("/api/offer", include_in_schema=False)(_ice)

    @app.post("/sessions/{session_id}/api/offer", include_in_schema=False)
    async def session_offer(session_id: str, request: Request, body: SmallWebRTCRequest, background: BackgroundTasks):
        if session_id not in sessions:
            raise HTTPException(status_code=404)
        return await _offer(request, body, background)

    @app.patch("/sessions/{session_id}/api/offer", include_in_schema=False)
    async def session_ice(session_id: str, request: Request, body: SmallWebRTCPatchRequest):
        if session_id not in sessions:
            raise HTTPException(status_code=404)
        return await _ice(request, body)

    app.mount("/client", PipecatPrebuiltUI)
    logger.info("browser test UI enabled: http://localhost:8000/test")
