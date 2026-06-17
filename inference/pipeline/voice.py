"""
voice.py — Voice-triggered shutter using Whisper speech recognition.

Listens on the default microphone for a trigger phrase ("scan", "capture",
"identify", "go"). When heard, returns control to the caller which fires
the camera capture + inference pipeline.

Optionally speaks the result back via pyttsx3 (offline TTS, no internet needed).
"""

import queue
import threading
import numpy as np

try:
    import whisper
    import sounddevice as sd
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────
TRIGGER_WORDS   = {"scan", "capture", "identify", "walking man", "what is this", "what is that"}
SAMPLE_RATE     = 16_000          # Whisper expects 16 kHz
CHUNK_SECONDS   = 2               # record this many seconds per listen window
WHISPER_MODEL   = "tiny.en"       # tiny.en is fast enough on Pi 5; use "base.en" for accuracy


class VoiceTrigger:
    """
    Blocking listener that returns when a trigger word is detected.

    Usage:
        trigger = VoiceTrigger()
        trigger.wait_for_trigger()   # blocks until user says a trigger word
        # ... run inference ...
        trigger.speak("Chanterelle. Confidence 91 percent. Safe to eat.")
    """

    def __init__(self, model_name: str = WHISPER_MODEL):
        self._model = None
        self._engine = None
        self._device_rate = SAMPLE_RATE  # may be overridden below

        if not _WHISPER_AVAILABLE:
            print("WARNING: whisper/sounddevice not installed — voice trigger disabled")
            print("  Run: pip install openai-whisper sounddevice")
        else:
            # Probe the default input device's native sample rate
            try:
                dev_info = sd.query_devices(kind="input")
                native_rate = int(dev_info["default_samplerate"])
                if native_rate != SAMPLE_RATE:
                    print(f"  Mic native rate: {native_rate} Hz (will resample to {SAMPLE_RATE})")
                    self._device_rate = native_rate
            except Exception:
                pass  # fall back to SAMPLE_RATE

            print(f"Loading Whisper model ({model_name}) ...")
            self._model = whisper.load_model(model_name)

        if not _TTS_AVAILABLE:
            print("WARNING: pyttsx3 not installed — TTS disabled")
            print("  Run: pip install pyttsx3")
        else:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 160)
                # Force-set an English voice — works around espeak misconfiguration
                voices = self._engine.getProperty("voices")
                en_voice = next(
                    (v for v in voices if "en" in (v.id or "").lower()),
                    voices[0] if voices else None,
                )
                if en_voice:
                    self._engine.setProperty("voice", en_voice.id)
            except Exception as e:
                print(f"WARNING: TTS init failed ({e}) — TTS disabled")
                print("  Try: sudo apt install espeak-ng espeak-ng-data")
                self._engine = None

        if self._model:
            print("Voice trigger ready.")
        else:
            print("Voice trigger unavailable — falling back to Enter key.")

    @property
    def can_listen(self) -> bool:
        return self._model is not None

    def wait_for_trigger(self) -> str:
        """
        Record CHUNK_SECONDS of audio in a loop until a trigger word is heard.
        Returns the full transcribed phrase that contained the trigger.
        Handles both single-word ("scan") and multi-word ("what is this") triggers.

        If whisper is unavailable, falls back to blocking on Enter key.
        """
        if not self.can_listen:
            print("  (voice unavailable — press Enter to scan)")
            try:
                input()
            except EOFError:
                pass
            return "scan"

        print(f"  Listening for: {sorted(TRIGGER_WORDS)} ...")

        while True:
            audio = self._record_chunk()
            text  = self._transcribe(audio)

            if not text:
                continue

            lower = text.lower()
            if any(trigger in lower for trigger in TRIGGER_WORDS):
                print(f"  Triggered by: '{text.strip()}'")
                return text.strip()

    def speak(self, message: str):
        """Speak a result aloud (blocking until audio finishes)."""
        if self._engine is not None:
            self._engine.say(message)
            self._engine.runAndWait()

    # ── Private ───────────────────────────────────────────────────────────────

    def _record_chunk(self) -> np.ndarray:
        """Record CHUNK_SECONDS of mono audio. Returns float32 array at 16 kHz."""
        frames = int(self._device_rate * CHUNK_SECONDS)
        audio  = sd.rec(frames, samplerate=self._device_rate, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        # Resample to 16 kHz if the mic records at a different rate
        if self._device_rate != SAMPLE_RATE:
            target_len = int(len(audio) * SAMPLE_RATE / self._device_rate)
            audio = np.interp(
                np.linspace(0, len(audio), target_len, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        return audio

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run Whisper on a raw audio chunk. Returns lowercased transcript."""
        result = self._model.transcribe(
            audio,
            language="en",
            fp16=False,    # Pi CPU doesn't support fp16
            verbose=False,
        )
        return result.get("text", "").strip().lower()


BOOT_MESSAGE = "Walking Man ready. Say scan to identify."
SCAN_CONFIRM = "Scanning."


def build_speech_message(result) -> str:
    """
    Build a natural-language TTS string from a ForagerResult.

    Safety-first: DEADLY findings are announced with a warning.
    Low-confidence or unknown results get a generic "try again" message.
    """
    from .convergence import ForagerResult

    safety_phrases = {
        "SAFE":    "appears safe.",
        "CAUTION": "use caution before consuming.",
        "DEADLY":  "Warning. This species may be deadly. Do not consume.",
        "UNKNOWN": "safety unknown. Do not consume.",
    }

    if result.is_unknown:
        if result.domain == "other":
            return "Not a foraging target. Point the camera at a berry, mushroom, or plant."
        else:
            # Router saw a domain but was below confidence threshold — likely positioning
            return "Could not identify. Move closer, 4 to 6 inches from the subject, and try again."

    if result.low_confidence:
        return "Too uncertain to identify. Ensure good lighting and a clear view, then try again."

    name   = result.species.replace("_", " ").title()
    phrase = safety_phrases.get(result.safety, "safety unknown.")
    conf   = int(result.confidence * 100)

    if result.is_deadly:
        return (
            f"Warning. Identified {name} in the {result.domain} domain. "
            f"This species may be deadly. Do not consume."
        )

    return (
        f"Identified {name} in the {result.domain} domain "
        f"at {conf} percent confidence. {phrase}"
    )
