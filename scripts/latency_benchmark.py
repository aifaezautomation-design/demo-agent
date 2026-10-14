# ruff: noqa: ASYNC240, ASYNC110  (a benchmark script: one log-file read, and a deliberate polling loop)
"""Phone-realistic latency benchmark against the LOCAL server (WebRTC transport, but the caller's audio is first
squeezed through 8 kHz G.711 mu-law like a real phone line).

Start the server with the test page on, email off, an 8 kHz pipeline, and its log in a file:

    ENABLE_TEST_UI=1 RESEND_API_KEY= TEST_UI_SAMPLE_RATE=8000 DEEPGRAM_STT=flux python -m uvicorn app.server:app --port 8000 > server.log 2>&1
    python scripts/latency_benchmark.py 2                 # passes over the 8 questions; SERVER_LOG=... if the log is elsewhere
    BENCH_SET=tools python scripts/latency_benchmark.py   # questions that make Mia check the calendar

Numbers from a laptop are RELATIVE: compare settings against each other. Absolute timings depend on where the machine is
relative to the US servers; Railway's own logs give the real ones.

For every question: time from the end of the caller's speech to the first audible audio from Mia, plus the server's own
per-stage averages (speech recognition, LLM first token, text-to-speech). Test calls are billed to the tenant and appear
in call_logs like any other.
"""
import asyncio
import audioop
import fractions
import os
import re
import statistics
import sys
import time
from pathlib import Path

import av
import httpx
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack

LOG = Path(os.environ.get("SERVER_LOG") or ROOT / "server.log")
VOICE = os.environ["CARTESIA_VOICE_ID"]
QUESTIONS = [
    "What are your hours on Saturday?",
    "Do you accept Delta Dental insurance?",
    "Where is your office located?",
    "Who are your dentists?",
    "How much is a root canal?",
    "Do you do teeth whitening?",
    "Is there parking at the office?",
    "What kinds of insurance do you take?",
]
if os.environ.get("BENCH_SET") == "tools":  # every one of these makes Mia look at the calendar
    QUESTIONS = [
        "I'm a new patient and I need a cleaning. Do you have anything tomorrow morning?",
        "I'm a new patient and I need a filling. Do you have anything on Friday afternoon?",
        "I'm a new patient and I want a cleaning. What is your earliest opening on Saturday?",
        "I'm a new patient and I need a cleaning. Do you have time next Tuesday?",
    ]


def synth(text):
    r = httpx.post("https://api.cartesia.ai/tts/bytes", timeout=30,
                   headers={"X-API-Key": os.environ["CARTESIA_API_KEY"], "Cartesia-Version": "2025-04-16"},
                   json={"model_id": "sonic-3.6", "transcript": text, "language": "en", "voice": {"mode": "id", "id": VOICE},
                         "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 48000}})
    r.raise_for_status()
    return np.frombuffer(r.content, dtype=np.int16)


def resample(pcm, src, dst):
    frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = src
    rs = av.AudioResampler(format="s16", layout="mono", rate=dst)
    out = rs.resample(frame) + (rs.resample(None) or [])
    return np.concatenate([f.to_ndarray().reshape(-1) for f in out])


def telephone(pcm48):
    """48 kHz -> 8 kHz -> G.711 mu-law round trip -> 48 kHz, plus a touch of line noise."""
    pcm8 = resample(pcm48, 48000, 8000)
    pcm8 = np.frombuffer(audioop.ulaw2lin(audioop.lin2ulaw(pcm8.tobytes(), 2), 2), dtype=np.int16)
    noise = (np.random.default_rng(7).normal(0, 60, pcm8.shape)).astype(np.int16)
    return resample((pcm8.astype(np.int32) + noise).clip(-32768, 32767).astype(np.int16), 8000, 48000)


class Mic(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.pcm = np.zeros(0, np.int16)
        self.t = 0
        self.t_end = None

    async def recv(self):
        await asyncio.sleep(0.02)
        chunk, self.pcm = self.pcm[:960], self.pcm[960:]
        if len(chunk) and not len(self.pcm):
            self.t_end = time.monotonic()
        if len(chunk) < 960:
            chunk = np.pad(chunk, (0, 960 - len(chunk)))
        f = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        f.sample_rate, f.pts, f.time_base = 48000, self.t, fractions.Fraction(1, 48000)
        self.t += 960
        return f


async def main(passes):
    clips = [(q, telephone(synth(q))) for q in QUESTIONS] * passes
    offset = LOG.stat().st_size if LOG.exists() else 0
    mic, pc = Mic(), RTCPeerConnection()
    pc.addTrack(mic)
    st = {"first": None, "last_loud": time.monotonic()}

    @pc.on("track")
    def on_track(track):
        async def drain():
            try:
                while True:
                    a = (await track.recv()).to_ndarray()
                    if int(np.abs(a).max()) > 800:
                        now = time.monotonic()
                        st["last_loud"] = now
                        if mic.t_end and st["first"] is None and now > mic.t_end:
                            st["first"] = now
            except Exception:
                pass
        asyncio.ensure_future(drain())

    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient(timeout=30) as c:
        sid = (await c.post("http://localhost:8000/start", json={"transport": "webrtc"})).json()["sessionId"]
        r = await c.post(f"http://localhost:8000/sessions/{sid}/api/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"})
        await pc.setRemoteDescription(RTCSessionDescription(**{k: r.json()[k] for k in ("sdp", "type")}))

    async def quiet(secs):
        while time.monotonic() - st["last_loud"] < secs:
            await asyncio.sleep(0.2)

    await asyncio.sleep(4)
    await quiet(1.5)
    times, said = [], []
    for q, clip in clips:
        st["first"], mic.t_end = None, None
        mic.pcm = clip
        for _ in range(250):
            await asyncio.sleep(0.1)
            if st["first"]:
                break
        times.append(round(st["first"] - mic.t_end, 2) if st["first"] and mic.t_end else None)
        said.append(q)
        await quiet(1.8)
    await pc.close()
    await asyncio.sleep(2.5)

    print(f"  per-turn: {times}")
    ok = sorted(t for t in times if t)
    print(f"  turns answered: {len(ok)}/{len(times)}")
    if ok:
        print(f"  end-of-speech -> first audio: median {statistics.median(ok):.2f}s  mean {statistics.mean(ok):.2f}s  "
              f"p25 {ok[len(ok)//4]:.2f}s  p75 {ok[(3*len(ok))//4]:.2f}s  max {ok[-1]:.2f}s")
    txt = re.sub(r"\x1b\[[0-9;]*m", "", LOG.read_bytes()[offset:].decode("utf-8", "ignore"))
    for name, pat in [("Deepgram TTFB", r"DeepgramSTTService#\d+ TTFB: ([\d.]+)s"), ("OpenAI TTFB", r"OpenAILLMService#\d+ TTFB: ([\d.]+)s"),
                      ("TTS text aggregation", r"text aggregation time: ([\d.]+)"), ("Cartesia first audio", r"CartesiaTTSService#\d+ TTFA: ([\d.]+)s")]:
        v = [float(x) for x in re.findall(pat, txt)]
        v = [x for x in v if x > 0.02] if "OpenAI" in name else v
        if v:
            print(f"  {name:22} mean {statistics.mean(v):.2f}s  median {statistics.median(v):.2f}s  (n={len(v)})")
    return said


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
