"""Caller-language handling.

How it works
  * Speech-to-text starts in Deepgram nova-3 `multi` mode, which understands English, Spanish, French,
    German, Portuguese, Italian, Hindi (and more) and reports the detected language per utterance.
  * `LanguageDetector` reads that and switches Mia's voice (Cartesia) to the same language before the
    LLM answers, so no LLM round trip is spent on it.
  * Urdu is not part of `multi`: it is transcribed as Hindi. The LLM tool `set_language("ur")` switches
    both speech-to-text and the voice to Urdu (Cartesia sonic-3.6 and Deepgram nova-3 both support it).
  * The LLM itself replies in the caller's language; this module only handles audio and the few
    sentences the code speaks (acknowledgements, the 911 safety line).
"""

import random
import re
from collections import deque

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    ManuallySwitchServiceFrame,
    ServiceSwitcherFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transcriptions.language import Language

from app import config

# Verified end to end with Mia's voice (Cartesia) and Deepgram: en, es, fr, hi, ur.
# de, pt, it follow the vendors' documented support and share the same code path.
SUPPORTED = ("en", "es", "fr", "de", "pt", "it", "hi", "ur")
# Languages covered by Deepgram nova-3 `multi`. Urdu is not, so it needs its own STT setting.
STT_MULTI = {"en", "es", "fr", "de", "pt", "it", "hi"}

# Deepgram's per-utterance language label is unreliable on short fragments (an Urdu "میرا نام ... ہے"
# came back as German "Mein Name ist"). So an utterance needs this many words to change the language,
# and moving between two non-English languages needs more evidence still.
MIN_WORDS_TO_SWITCH = 4
MIN_WORDS_FOREIGN_TO_FOREIGN = 6
HINDUSTANI = {"hi", "ur"}

# English acknowledgements spoken while a tool runs. A caller who needs two searches should not hear the identical
# sentence twice, so each has a few variants that LanguageState.filler() rotates through. Each is under a second spoken.
# "book" also covers reschedule, cancel, and the demo callback, so none of them names what is being done.
FILLER_VARIANTS: dict[str, list[str]] = {
    "check": ["Let me take a look.", "One sec, checking now.", "Sure, checking that now."],
    "book": ["Got it, one moment.", "Perfect, just a second.", "Okay, doing that now."],
}

# Sentences spoken by code (not the LLM). Review with a native speaker before relying on them.
PHRASES: dict[str, dict[str, str]] = {
    "check": {
        "en": FILLER_VARIANTS["check"][0],
        "es": "Un momento, por favor, voy a revisar.",
        "fr": "Un instant, je vérifie.",
        "de": "Einen Moment, ich prüfe das.",
        "pt": "Um momento, vou verificar.",
        "it": "Un attimo, controllo subito.",
        "hi": "एक पल रुकिए, मैं देखती हूँ।",
        "ur": "ایک لمحہ، میں دیکھتی ہوں۔",
    },
    "book": {
        "en": FILLER_VARIANTS["book"][0],
        "es": "Entendido, un momento.",
        "fr": "C'est noté, un instant.",
        "de": "Verstanden, einen Moment.",
        "pt": "Certo, um momento.",
        "it": "Perfetto, un attimo.",
        "hi": "ठीक है, एक पल रुकिए।",
        "ur": "ٹھیک ہے، ایک لمحہ۔",
    },
    "emergency": {
        "en": "I'm so sorry you're hurting. Let me get you in right away. If it's ever hard to breathe or swallow, call 911.",
        "es": "Lamento mucho que tenga dolor. Voy a conseguirle una cita de inmediato. Si le cuesta respirar o tragar, llame al 911.",
        "fr": "Je suis désolée que vous souffriez. Je vous trouve un rendez-vous tout de suite. Si vous avez du mal à respirer ou à avaler, appelez le 911.",
        "de": "Es tut mir leid, dass Sie Schmerzen haben. Ich besorge Ihnen sofort einen Termin. Wenn Sie schwer atmen oder schlucken können, rufen Sie 911 an.",
        "pt": "Sinto muito que você esteja com dor. Vou conseguir uma consulta para você agora mesmo. Se ficar difícil respirar ou engolir, ligue para o 911.",
        "it": "Mi dispiace che stia soffrendo. Le trovo subito un appuntamento. Se fa fatica a respirare o a deglutire, chiami il 911.",
        "hi": "मुझे बहुत अफ़सोस है कि आपको दर्द हो रहा है। मैं अभी आपके लिए अपॉइंटमेंट देखती हूँ। अगर साँस लेने या निगलने में तकलीफ़ हो, तो तुरंत 911 पर कॉल करें।",
        "ur": "مجھے بہت افسوس ہے کہ آپ کو تکلیف ہو رہی ہے۔ میں ابھی آپ کے لیے ملاقات دیکھتی ہوں۔ اگر سانس لینے یا نگلنے میں دشواری ہو تو فوراً 911 پر کال کریں۔",
    },
    "trouble": {
        "en": "Sorry, I missed that. Could you say it again?",
        "es": "Perdón, no le escuché bien. ¿Puede repetirlo?",
        "fr": "Désolée, je n'ai pas bien entendu. Pouvez-vous répéter ?",
        "de": "Entschuldigung, das habe ich nicht verstanden. Können Sie es wiederholen?",
        "pt": "Desculpe, não entendi. Pode repetir?",
        "it": "Scusi, non ho capito. Può ripetere?",
        "hi": "माफ़ कीजिए, मैं सुन नहीं पाई। क्या आप फिर से कह सकते हैं?",
        "ur": "معاف کیجیے، میں سن نہیں پائی۔ کیا آپ دوبارہ کہہ سکتے ہیں؟",
    },
    "still_there": {
        "en": "Are you still there?",
        "es": "¿Sigue ahí?",
        "fr": "Vous êtes toujours là ?",
        "de": "Sind Sie noch da?",
        "pt": "Você ainda está aí?",
        "it": "È ancora lì?",
        "hi": "क्या आप अभी भी लाइन पर हैं?",
        "ur": "کیا آپ ابھی لائن پر ہیں؟",
    },
    "idle_goodbye": {
        "en": "I'll let you go for now. Feel free to call back anytime. Goodbye!",
        "es": "Los dejo por ahora. Puede volver a llamar cuando quiera. ¡Adiós!",
        "fr": "Je vous laisse pour l'instant. N'hésitez pas à rappeler. Au revoir !",
        "de": "Ich lasse Sie jetzt gehen. Rufen Sie gern jederzeit wieder an. Auf Wiederhören!",
        "pt": "Vou deixá-lo por agora. Ligue de novo quando quiser. Tchau!",
        "it": "La lascio per ora. Può richiamare quando vuole. Arrivederci!",
        "hi": "अभी के लिए मैं फ़ोन रखती हूँ। आप जब चाहें दोबारा कॉल कर सकते हैं। अलविदा!",
        "ur": "فی الحال میں فون رکھتی ہوں۔ آپ جب چاہیں دوبارہ کال کر سکتے ہیں۔ خدا حافظ!",
    },
    "max_duration": {
        "en": "I'm sorry, I need to wrap up this call. Please call back if you need anything else.",
        "es": "Lo siento, tengo que terminar esta llamada. Por favor, vuelva a llamar si necesita algo más.",
        "fr": "Je suis désolée, je dois terminer cet appel. Rappelez-nous si vous avez besoin d'autre chose.",
        "de": "Es tut mir leid, ich muss dieses Gespräch beenden. Rufen Sie gern wieder an, wenn Sie noch etwas brauchen.",
        "pt": "Sinto muito, preciso encerrar esta ligação. Ligue de novo se precisar de mais alguma coisa.",
        "it": "Mi dispiace, devo concludere questa chiamata. Richiami se ha bisogno di altro.",
        "hi": "माफ़ कीजिए, मुझे यह कॉल समाप्त करनी होगी। कुछ और ज़रूरत हो तो दोबारा कॉल करें।",
        "ur": "معذرت، مجھے یہ کال ختم کرنی ہوگی۔ کسی اور مدد کے لیے دوبارہ کال کریں۔",
    },
}


# Spoken Hindi and Urdu are nearly the same language, and Deepgram's multilingual mode hears both as Hindi.
# Vocabulary is what separates them: these Urdu-leaning words, as Deepgram writes them in Devanagari or
# Latin letters, mark a caller as Urdu-speaking. One strong marker, or two weak ones, is enough.
# Deepgram spells the same greeting differently every time ("असामिलायक़म", "अस्लामायक़म"), so the
# strongest markers are patterns, matched after removing nukta dots. Words that Hindi speakers also
# use a lot ("zaroor", "shukriya", "ji haan") are deliberately left out.
_URDU_STRONG = re.compile(
    r"(अस्?सा?ला|अस्?ला|असामि|सलाम|वा?ल?[ऐैए]कुम|अल[ऐैे]कुम|लायकुम|लामायकम|आदाब|उर्दू|اردو|"
    r"a?s+a?lam|alaikum|alaykum|salaa?m|urdu|khuda ?hafiz)"
)
_URDU_WEAK = ("मुलाकात", "मेहरबानी", "बराह", "खुदा", "जनाब")

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "it": "Italian", "hi": "Hindi (Devanagari script)", "ur": "Urdu (Arabic script)",
}


def looks_urdu(text: str) -> bool:
    t = text.lower().replace("़", "")  # drop nukta dots: क़ -> क, ज़ -> ज
    return bool(_URDU_STRONG.search(t)) or sum(m in t for m in _URDU_WEAK) >= 2


def phrase(kind: str, lang: str) -> str:
    """A code-spoken sentence in the caller's language, falling back to English."""
    return PHRASES[kind].get(lang) or PHRASES[kind]["en"]


class LanguageState:
    """The caller's current language, shared by the detector, the tools, and the spoken phrases."""

    def __init__(self, stt_default: str | None = None):
        stt_default = stt_default or config.DEEPGRAM_LANGUAGE
        self.current = "en"
        self.stt = stt_default  # which speech-to-text is active: the default ("multi") or "ur"
        self._stt_default = stt_default
        # {"multi": <Deepgram multilingual>, "ur": <Deepgram Urdu>}; filled in by the pipeline. Two
        # services behind a ServiceSwitcher, because reconnecting one live Deepgram stream to a new
        # language mid-call silently stopped the audio from reaching the pipeline.
        self.stt_services: dict[str, object] = {}
        self._filler_last: dict[str, int] = {}  # which variant of each filler was used last this call

    def filler(self, kind: str, lang: str) -> str:
        """The spoken acknowledgement for `kind`. In English it cycles through FILLER_VARIANTS from a random start, so
        the same line is never used twice in a row; other languages keep their single phrase."""
        options = FILLER_VARIANTS.get(kind) if lang == "en" else None
        if not options:
            return phrase(kind, lang)
        last = self._filler_last.get(kind)
        index = random.randrange(len(options)) if last is None else (last + 1) % len(options)
        self._filler_last[kind] = index
        return options[index]

    def stt_language_for(self, lang: str) -> str:
        return "ur" if lang == "ur" else self._stt_default


def switch_frames(state: LanguageState, lang: str) -> list[Frame]:
    """Updates `state` and returns the settings frames that apply `lang` to speech-to-text and the voice."""
    frames: list[Frame] = []
    wanted_stt = state.stt_language_for(lang)
    if wanted_stt != state.stt:
        frames.append(ManuallySwitchServiceFrame(service=state.stt_services[wanted_stt]))
        state.stt = wanted_stt
    frames.append(TTSUpdateSettingsFrame(delta=CartesiaTTSService.Settings(language=Language(lang))))
    state.current = lang
    return frames


class LanguageDetector(FrameProcessor):
    """Sits right after speech-to-text and moves the voice to the language the caller just spoke."""

    def __init__(self, state: LanguageState, queue_at_start):
        """`queue_at_start(frame)` injects a frame at the head of the pipeline. The speech-to-text
        switcher is *upstream* of this processor, so only a frame entering from the start can reach it."""
        super().__init__()
        self._state = state
        self._queue_at_start = queue_at_start
        # Greetings like "assalam alaikum" often arrive as their own short segment, so Urdu evidence
        # is judged over the last few utterances rather than one.
        self._recent: deque[str] = deque(maxlen=3)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            self._recent.append(frame.text)
            code = self._detected(frame)
            if code:
                await self._apply(code, frame.text, direction)
        await self.push_frame(frame, direction)

    async def _apply(self, code: str, heard: str, direction: FrameDirection) -> None:
        logger.info(f"language: {self._state.current} -> {code} (heard: {heard[:40]!r})")
        for update in switch_frames(self._state, code):
            if isinstance(update, ServiceSwitcherFrame):
                await self._queue_at_start(update)
            else:
                # Pushed ahead of the transcript, so the voice is ready before the reply is spoken.
                await self.push_frame(update, direction)
        await self.push_frame(
            LLMMessagesAppendFrame(
                [{"role": "developer", "content": f"The caller is now speaking {LANGUAGE_NAMES[code]}. "
                  f"Reply in {LANGUAGE_NAMES[code]} from now on, using the polite form of address."}],
                run_llm=False,
            ),
            direction,
        )

    def _detected(self, frame: TranscriptionFrame) -> str | None:
        if not frame.language or len(frame.text.split()) < MIN_WORDS_TO_SWITCH:
            return None
        code = str(frame.language.value if isinstance(frame.language, Language) else frame.language)
        code = code.split("-")[0].lower()
        if self._state.stt == "ur":
            return None  # in Urdu mode only the LLM tool changes language (STT there is Urdu-only)
        if code == "hi" and looks_urdu(" ".join(self._recent) or frame.text):
            code = "ur"
        if code not in SUPPORTED or code == self._state.current:
            return None
        current = self._state.current
        if current in HINDUSTANI and code not in HINDUSTANI and code != "en":
            return None  # a Hindi/Urdu caller is never "really" speaking German: it's a mislabeled fragment
        if current != "en" and code != "en" and len(frame.text.split()) < MIN_WORDS_FOREIGN_TO_FOREIGN:
            return None
        return code
