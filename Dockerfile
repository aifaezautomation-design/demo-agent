FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Bake the Silero VAD and Smart Turn models into the image so the first call
# after a deploy doesn't pay the model-load cost.
RUN python -c "from pipecat.audio.vad.silero import SileroVADAnalyzer; SileroVADAnalyzer(); \
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3; LocalSmartTurnAnalyzerV3()"

ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --ws-ping-interval 20"]
