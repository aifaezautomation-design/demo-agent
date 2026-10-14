"""One Pipecat pipeline per call (phone via run_bot, browser test via run_call):

    Telnyx WS -> Deepgram STT -> user aggregator (Silero VAD + Smart Turn)
              -> gpt-4o-mini (+ tools) -> Cartesia TTS -> Telnyx WS

Audio stays at 8 kHz end to end to match the phone network and skip resampling.
"""

import asyncio
import time
import uuid

from fastapi import WebSocket
from loguru import logger
from openai import AsyncOpenAI
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndWorkerFrame, LLMFullResponseStartFrame, LLMTextFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import TelnyxCallData
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from app import config, db
from app.db import Tenant
from app.prompts import build_system_prompt
from app.tools import CallSession

PHONE_SAMPLE_RATE = 8000
_openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


class _SpokeTracker(FrameProcessor):
    """Remembers whether the LLM already produced speech text in its current response."""

    def __init__(self):
        super().__init__()
        self.spoke = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self.spoke = False
        elif isinstance(frame, LLMTextFrame) and frame.text.strip():
            self.spoke = True
        await self.push_frame(frame, direction)


def _greeting(tenant: Tenant) -> str:
    disclosure = " This call may be recorded." if tenant.record_calls else ""
    return f"Thanks for calling {tenant.name}, this is Mia.{disclosure} How can I help you today?"


def build_services() -> tuple[DeepgramSTTService, OpenAILLMService, CartesiaTTSService]:
    stt = DeepgramSTTService(
        api_key=config.DEEPGRAM_API_KEY,
        settings=DeepgramSTTService.Settings(
            model=config.DEEPGRAM_MODEL,
            language=Language.EN_US,
            smart_format=True,
            punctuate=True,
            interim_results=True,
            **({"endpointing": config.DEEPGRAM_ENDPOINTING_MS} if config.DEEPGRAM_ENDPOINTING_MS is not None else {}),
        ),
    )
    llm = OpenAILLMService(
        api_key=config.OPENAI_API_KEY,
        settings=OpenAILLMService.Settings(
            model=config.OPENAI_MODEL,
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=200,  # voice turns are short; caps runaway answers
        ),
    )
    tts = CartesiaTTSService(
        api_key=config.CARTESIA_API_KEY,
        settings=CartesiaTTSService.Settings(
            voice=config.CARTESIA_VOICE_ID,
            generation_config=GenerationConfig(volume=config.CARTESIA_VOLUME),
        ),
        text_aggregation_mode=TextAggregationMode.TOKEN if config.CARTESIA_STREAM_TOKENS else TextAggregationMode.SENTENCE,
        max_buffer_delay_ms=config.CARTESIA_MAX_BUFFER_MS,
    )
    return stt, llm, tts


async def run_bot(websocket: WebSocket, call_data: TelnyxCallData, tenant: Tenant, call_sid: str | None) -> None:
    """A phone call: Telnyx media stream over a WebSocket, 8 kHz."""
    encoding = call_data.outbound_encoding or "PCMU"
    serializer = TelnyxFrameSerializer(
        stream_id=call_data.stream_id,
        outbound_encoding=encoding,
        inbound_encoding=encoding,
        call_control_id=call_data.call_id,
        api_key=config.TELNYX_API_KEY,
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )
    await run_call(
        transport, tenant,
        caller_phone=call_data.from_number, call_control_id=call_data.call_id, call_sid=call_sid,
        sample_rate=PHONE_SAMPLE_RATE,
    )


async def run_call(
    transport,
    tenant: Tenant,
    *,
    caller_phone: str | None,
    call_control_id: str | None,
    call_sid: str | None,
    sample_rate: int,
    rtvi: bool = False,
) -> None:
    """The call itself, independent of how audio arrives (phone or browser test)."""
    started = time.monotonic()
    # The id is chosen here and the row inserted in the background: waiting on the database
    # before building the pipeline used to delay the greeting by about 0.8 s.
    call_id = str(uuid.uuid4())
    call_log_ready = asyncio.ensure_future(
        db.create_call_log(tenant.id, caller_phone or "unknown", call_control_id, call_sid, call_id=call_id)
    )
    session = CallSession(tenant, call_id, caller_phone, call_log_ready)
    transcript: list[dict] = []
    log = logger.bind(call_id=call_id, client_id=tenant.id)

    stt, llm, tts = build_services()
    session.register(llm)
    greeting = _greeting(tenant)
    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(tenant, caller_phone or "")}],
        tools=session.tools_schema(),
    )
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Short VAD stop + Smart Turn (the default stop strategy) decides
            # end-of-turn from audio, rather than waiting on long silences.
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=config.VAD_STOP_SECS)),
            user_idle_timeout=12.0,
        ),
    )

    spoke = _SpokeTracker()
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            spoke,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=sample_rate,
            audio_out_sample_rate=sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=60,
        enable_rtvi=rtvi,  # only the browser test client speaks RTVI; phone calls skip the extra processor
    )

    @llm.event_handler("on_function_calls_started")
    async def on_tool_started(service, function_calls):
        # Tools take 0.5-3 s (Cal.com, Supabase). A short spoken acknowledgement keeps the line
        # from going silent; it is not added to the LLM context.
        if spoke.spoke:  # Mia already said something before the tool call; don't talk over her
            return
        names = {c.function_name for c in function_calls}
        emergency = any(c.function_name == "check_availability" and (c.arguments or {}).get("is_emergency")
                        for c in function_calls)
        if emergency:
            # Spoken by code, not the LLM, so an emergency always gets an immediate human-sounding reply.
            filler = "I'm so sorry you're hurting. Let me get you in right away. If it's ever hard to breathe or swallow, call 911."
        elif names & {"book_appointment", "reschedule_appointment", "cancel_appointment"}:
            filler = "Got it, one moment."
        elif names & {"check_availability", "find_appointments"}:
            filler = "One moment while I check."
        else:
            return
        await service.push_frame(TTSSpeakFrame(filler, append_to_context=False))

    @transport.event_handler("on_client_connected")
    async def on_connected(_transport, _ws):
        # Spoken straight through TTS: no LLM round trip before the caller hears us.
        await worker.queue_frame(TTSSpeakFrame(greeting))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _ws):
        log.info("caller hung up")
        await worker.cancel()

    @aggregators.user().event_handler("on_user_turn_stopped")
    async def on_user_turn(_agg, _strategy, message):
        if message.content:
            transcript.append(session.transcript_entry("patient", message.content))

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn(_agg, message):
        if message.content:
            transcript.append(session.transcript_entry("mia", message.content))

    idle_prompts = 0

    @aggregators.user().event_handler("on_user_turn_idle")
    async def on_idle(_agg):
        nonlocal idle_prompts
        idle_prompts += 1
        if idle_prompts == 1:
            await worker.queue_frame(TTSSpeakFrame("Are you still there?"))
        else:
            await worker.queue_frames(
                [TTSSpeakFrame("I'll let you go for now. Feel free to call back anytime. Goodbye!"), EndWorkerFrame()]
            )

    async def enforce_max_duration():
        await asyncio.sleep(config.MAX_CALL_SECONDS)
        log.warning("max call duration reached")
        await worker.queue_frames(
            [TTSSpeakFrame("I'm sorry, I need to wrap up this call. Please call back if you need anything else."), EndWorkerFrame()]
        )

    duration_guard = asyncio.create_task(enforce_max_duration())
    try:
        await WorkerRunner(handle_sigint=False).run(worker)
    finally:
        duration_guard.cancel()
        await _close_call(call_id, started, transcript, session, call_log_ready, log)


async def _close_call(
    call_id: str, started: float, transcript: list[dict], session: CallSession, call_log_ready, log
) -> None:
    """Bill and summarize. Runs after the caller is gone, so it costs no latency."""
    duration = int(time.monotonic() - started)
    summary = await _summarize(transcript)
    try:
        await call_log_ready  # the row must exist before it can be finalized
        await session.flush()  # background appointment writes
        minutes = await db.finalize_call(call_id, duration, transcript, summary, session.appointment_booked)
        log.info(f"call closed: {duration}s, billed {minutes} min, booked={session.appointment_booked}")
    except Exception:
        # Never lose billing silently: the row stays 'in_progress' with
        # billed_minutes null, which a reconciliation job can pick up.
        log.exception("failed to finalize call")


async def _summarize(transcript: list[dict]) -> str | None:
    if not transcript:
        return None
    lines = "\n".join(f"{t['role']}: {t['content']}" for t in transcript)
    try:
        resp = await _openai.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=0,
            max_tokens=120,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this dental front-desk call in one or two sentences for clinic staff. "
                    "Include the patient's name, the intent (book/reschedule/cancel/emergency/question), the outcome, "
                    "and any follow-up needed.",
                },
                {"role": "user", "content": lines[:12000]},
            ],
        )
        return resp.choices[0].message.content
    except Exception:
        logger.exception("summary generation failed")
        return None
