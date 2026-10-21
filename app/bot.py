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
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndWorkerFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import (
    ServiceSwitcher,
    ServiceSwitcherStrategyManual,
)
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import TelnyxCallData
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from app import config, db, notify
from app import language as language_module
from app.db import Tenant
from app.emergency import EmergencyTriage
from app.guards import NOTES, PromiseGuard, broke_promise  # noqa: F401  (broke_promise is re-exported for tests)
from app.language import LanguageDetector
from app.prompts import build_system_prompt
from app.tools import CallSession
from app.trimmer import OfferTrimmer

PHONE_SAMPLE_RATE = 8000

_SUMMARY = (
    "Summarize this dental front-desk call in one or two sentences for clinic staff. "
    "Include the patient's name, the intent (book/reschedule/cancel/emergency/question), the outcome, "
    "and any follow-up needed. If the caller asked for a real person or a callback, say so clearly."
)
# The demo line is called by dental practice owners and managers trying Mia out; the summary is for the Boltix AI team.
_SUMMARY_DEMO = (
    "Summarize this call to a demo dental front desk in two or three sentences for the Boltix AI sales team. "
    "Say whether the caller seemed to be a practice owner or manager (or a patient), what they asked about, the outcome "
    "(for example a test booking or a callback request), and any follow-up needed. "
    "If the caller asked for a real person or a callback, say so clearly."
)
_openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


class _SpokeTracker(FrameProcessor):
    """Watches each LLM response: did it speak, did it call a tool, and did it promise an action it never took?"""

    def __init__(self):
        super().__init__()
        self.spoke = False
        self.called = False
        self.text = ""
        self.nudges = 0  # re-prompts allowed once per caller turn
        self.guard = PromiseGuard()
        self.on_broken_promise = None  # set by run_call, which owns the worker and the context
        self.searched_this_turn = lambda: True  # set by run_call; True means "do not second-guess availability claims"
        self.writes = lambda: 0  # set by run_call: bookings, changes and cancellations really saved so far
        self.abuse = lambda: 0  # set by run_call: how many times the caller has insulted Mia (demo line only)
        self.on_said = None  # set by run_call: gets every spoken LLM reply the moment it ends
        self.last_activity = time.monotonic()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self.spoke, self.called, self.text = False, False, ""
            self.last_activity = time.monotonic()
        elif isinstance(frame, LLMTextFrame) and frame.text.strip():
            self.spoke = True
            self.text += frame.text
            self.last_activity = time.monotonic()
        elif isinstance(frame, FunctionCallsStartedFrame):
            self.called = True
            self.last_activity = time.monotonic()
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self.on_said and self.text.strip():
                self.on_said(self.text.strip())
            kind = self.guard.check(self.text, self.called, self.writes(), self.searched_this_turn(), self.abuse())
            if kind and self.on_broken_promise and self.nudges < 1:
                self.nudges += 1
                asyncio.ensure_future(self.on_broken_promise(self.text.strip(), NOTES[kind]))
        await self.push_frame(frame, direction)


def _turn_strategies() -> UserTurnStrategies | None:
    """Smart Turn with a short silence cap. None for Flux, which decides the end of the turn itself."""
    if config.DEEPGRAM_STT == "flux" and not config.MULTILINGUAL_ENABLED:
        return None
    analyzer = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=config.SMART_TURN_STOP_SECS))
    return UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)])


def _greeting(tenant: Tenant) -> str:
    disclosure = " This call may be recorded." if tenant.record_calls else ""
    return f"Thanks for calling {tenant.name}, this is Mia.{disclosure} How can I help you today?"


def keyterms_for(kb: dict | None) -> list[str]:
    """Words Deepgram should favour: the practice's dentists, services, and insurers (names a generic model gets wrong)."""
    kb = kb or {}
    terms: list[str] = []
    for member in kb.get("team", []):
        name = member.get("name", "")
        terms += [name, name.replace("Dr. ", "").strip()]
    terms += list(kb.get("insurances", []))
    terms += [sv["name"] for sv in kb.get("services", []) if len(sv.get("name", "")) <= 30]
    terms += ["dentist", "dentists", "orthodontist", "root canal", "crown", "whitening"]
    seen: set[str] = set()
    return [t for t in terms if t and not (t.lower() in seen or seen.add(t.lower()))][:30]


def build_stt(language: str, keyterms: list[str] | None = None):
    if config.DEEPGRAM_STT == "flux" and language == "en" and not config.MULTILINGUAL_ENABLED:
        # Flux ends the turn itself and signals an *eager* end of turn, so the LLM can start while the caller's last
        # words are still being finalised (Pipecat withdraws the guess, and never runs a tool, if the caller goes on).
        return DeepgramFluxSTTService(
            api_key=config.DEEPGRAM_API_KEY,
            enable_eager_end_of_turn=True,
            settings=DeepgramFluxSTTService.Settings(
                model="flux-general-en",
                eager_eot_threshold=config.FLUX_EAGER_EOT_THRESHOLD,
                eot_threshold=config.FLUX_EOT_THRESHOLD,
                eot_timeout_ms=config.FLUX_EOT_TIMEOUT_MS,
                keyterm=keyterms or [],
            ),
        )
    return _build_nova_stt(language)


def _build_nova_stt(language: str) -> DeepgramSTTService:
    return DeepgramSTTService(
        api_key=config.DEEPGRAM_API_KEY,
        settings=DeepgramSTTService.Settings(
            model=config.DEEPGRAM_MODEL,
            language=language,
            smart_format=True,
            punctuate=True,
            interim_results=True,
            **({"endpointing": config.DEEPGRAM_ENDPOINTING_MS} if config.DEEPGRAM_ENDPOINTING_MS is not None else {}),
        ),
    )


def build_services(keyterms: list[str] | None = None) -> tuple[DeepgramSTTService, OpenAILLMService, CartesiaTTSService]:
    stt = build_stt(config.DEEPGRAM_LANGUAGE, keyterms)
    llm = OpenAILLMService(
        api_key=config.OPENAI_API_KEY,
        settings=OpenAILLMService.Settings(
            model=config.OPENAI_MODEL,
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=config.OPENAI_MAX_TOKENS,  # voice turns are short; caps runaway answers
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
    session.prefetch_slots()  # calendar openings load while the greeting plays
    transcript: list[dict] = []
    log = logger.bind(call_id=call_id, client_id=tenant.id)

    stt, llm, tts = build_services(keyterms_for(tenant.knowledge_base))
    session.register(llm)
    stt_node, language_detector = stt, None
    if config.MULTILINGUAL_ENABLED:
        # Urdu is outside Deepgram's multilingual mode, so it gets its own service behind a switcher.
        stt_urdu = build_stt("ur")
        stt_node = ServiceSwitcher(services=[stt, stt_urdu], strategy_type=ServiceSwitcherStrategyManual)
        session.language_state.stt_services = {session.language_state.stt: stt, "ur": stt_urdu}
        language_detector = LanguageDetector(session.language_state, lambda frame: worker.queue_frame(frame))
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
            user_idle_timeout=config.USER_IDLE_SECS,
            user_turn_stop_timeout=config.USER_TURN_STOP_TIMEOUT,
            user_turn_strategies=_turn_strategies(),
        ),
    )

    spoke = _SpokeTracker()
    pipeline = Pipeline(
        [
            transport.input(),
            stt_node,
            *([language_detector] if language_detector else []),
            EmergencyTriage(session, session.language_state),
            aggregators.user(),
            llm,
            OfferTrimmer(enabled=tenant.demo_mode),  # demo line: no stock closing offers (see app/trimmer.py)
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

    async def keep_promise(spoken: str, note: str) -> None:
        log.warning(f"guard fired on {spoken!r}: re-prompting the model")
        context.add_message({"role": "system", "content": note})
        await worker.queue_frame(LLMRunFrame())

    spoke.on_broken_promise = keep_promise
    spoke.searched_this_turn = lambda: session.turn_searched
    spoke.on_said = session.note_said
    spoke.writes = lambda: session.writes
    spoke.abuse = lambda: session.abuse_count if tenant.demo_mode else 0
    stall_watch: asyncio.Task | None = None

    async def stall_guard(stopped_at: float) -> None:
        # The caller finished speaking and nothing at all happened (no words, no tool): never leave them in silence.
        # First the model is asked again; if that also produces nothing (an OpenAI outage, a hung tool), a fixed line is spoken.
        await asyncio.sleep(config.STALL_SECS)
        if spoke.last_activity >= stopped_at or session.busy:
            return
        log.warning(f"no reply {config.STALL_SECS:.0f}s after the caller's turn: re-prompting the model")
        retried_at = time.monotonic()
        context.add_message({"role": "system", "content": NOTES["stall"]})
        await worker.queue_frame(LLMRunFrame())
        await asyncio.sleep(config.STALL_SECS)
        if spoke.last_activity < retried_at and not session.busy:
            log.error("the model produced nothing after a re-prompt: speaking the fallback line")
            await worker.queue_frame(TTSSpeakFrame(language_module.phrase("trouble", session.language_state.current)))

    async def apply_language(code: str) -> None:
        log.info(f"language set to {code}")
        await worker.queue_frames(language_module.switch_frames(session.language_state, code))

    if config.MULTILINGUAL_ENABLED:
        session.apply_language = apply_language

    @llm.event_handler("on_function_calls_started")
    async def on_tool_started(service, function_calls):
        # Tools take 0.5-3 s (Cal.com, Supabase). A short spoken acknowledgement keeps the line
        # from going silent; it is not added to the LLM context.
        if spoke.spoke:  # Mia already said something before the tool call; don't talk over her
            return
        names = {c.function_name for c in function_calls}
        emergency = any(c.function_name == "check_availability" and (c.arguments or {}).get("is_emergency")
                        for c in function_calls)
        lang = session.language_state.current
        if emergency:
            # Spoken by code, not the LLM, so an emergency always gets an immediate human-sounding reply.
            filler = language_module.phrase("emergency", lang)
        elif names & {"book_appointment", "reschedule_appointment", "cancel_appointment", "request_boltix_callback"}:
            filler = session.language_state.filler("book", lang)  # a different line from the last one this call
        elif names & {"check_availability", "find_appointments"}:
            filler = session.language_state.filler("check", lang)
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
        nonlocal stall_watch
        if message.content:
            log.info(f"patient: {message.content}")
            if stall_watch:
                stall_watch.cancel()
            stall_watch = asyncio.ensure_future(stall_guard(time.monotonic()))
            session.note_heard(message.content)
            transcript.append(session.transcript_entry("patient", message.content))

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn(_agg, message):
        if message.content:
            transcript.append(session.transcript_entry("mia", message.content))

    idle_prompts = 0

    @aggregators.user().event_handler("on_user_turn_started")
    async def on_user_started(_agg, _strategy):
        nonlocal idle_prompts, stall_watch
        idle_prompts = 0  # the caller is talking: the silence count starts over
        spoke.nudges = 0
        if stall_watch:
            stall_watch.cancel()

    @aggregators.user().event_handler("on_user_turn_idle")
    async def on_idle(_agg):
        nonlocal idle_prompts
        if session.busy:  # a tool call (calendar lookup, booking) is running: that silence is ours, not the caller's
            return
        idle_prompts += 1
        lang = session.language_state.current
        if idle_prompts == 1:
            await worker.queue_frame(TTSSpeakFrame(language_module.phrase("still_there", lang)))
        else:
            await worker.queue_frames(
                [TTSSpeakFrame(language_module.phrase("idle_goodbye", lang)), EndWorkerFrame()]
            )

    async def enforce_max_duration():
        await asyncio.sleep(config.MAX_CALL_SECONDS)
        log.warning("max call duration reached")
        await worker.queue_frames(
            [TTSSpeakFrame(language_module.phrase("max_duration", session.language_state.current)), EndWorkerFrame()]
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
    demo = session.tenant.demo_mode
    summary = await _summarize(transcript, demo=demo)
    try:
        await call_log_ready  # the row must exist before it can be finalized
        await session.flush()  # background appointment writes
        minutes = await db.finalize_call(call_id, duration, transcript, summary, session.appointment_booked)
        log.info(f"call closed: {duration}s, billed {minutes} min, booked={session.appointment_booked}")
    except Exception:
        # Never lose billing silently: the row stays 'in_progress' with
        # billed_minutes null, which a reconciliation job can pick up.
        log.exception("failed to finalize call")
    if demo:
        _email_demo_report(call_id, duration, transcript, summary, session, log)


def _email_demo_report(call_id: str, duration: int, transcript: list[dict], summary: str | None, session: CallSession, log) -> None:
    """Emails Boltix AI what happened on a demo call. Scheduled in the background after the caller is gone; any failure
    is logged and ignored, so it can never touch billing or the call."""
    try:
        cb = session.callback
        lead = (
            notify.DemoLeadEmail(
                name=cb["name"], practice_name=cb["practice_name"], phone=cb["phone"],
                best_time=cb.get("best_time") or "", notes=cb.get("notes") or "", call_id=call_id,
            )
            if cb else None
        )
        notify.notify_demo_call(
            notify.DemoCallEmail(
                caller_phone=session.caller_phone or "", duration_seconds=duration, summary=summary or "",
                callback_requested=cb is not None, transcript=transcript, call_id=call_id, lead=lead,
            )
        )
    except Exception:
        log.exception("could not schedule the demo call report email")


async def _summarize(transcript: list[dict], demo: bool = False) -> str | None:
    if not transcript:
        return None
    lines = "\n".join(f"{t['role']}: {t['content']}" for t in transcript)
    try:
        resp = await _openai.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=0,
            max_tokens=200 if demo else 120,
            messages=[
                {"role": "system", "content": _SUMMARY_DEMO if demo else _SUMMARY},
                {"role": "user", "content": lines[:12000]},
            ],
        )
        return resp.choices[0].message.content
    except Exception:
        logger.exception("summary generation failed")
        return None
