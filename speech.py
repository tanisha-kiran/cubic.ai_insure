"""
Offline speech-to-text for the voice recorder (and uploaded audio), using faster-whisper.

The model loads once and is reused: the first dictation after the server starts pays the load
(a few seconds for "base"), later ones only the transcription. WHISPER_MODEL=tiny|base|small
trades speed for accuracy. Nothing leaves the machine.
"""
import os
import tempfile
import threading

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            from hardware_profile import PROFILE

            threads = PROFILE["num_thread"] or 0
            _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=threads)
        return _model


def transcribe_bytes(data: bytes, suffix: str = ".webm", language: str | None = None):
    """Returns (text, detected_language, duration_seconds). Handles webm/ogg/wav/mp3/m4a via PyAV."""
    if not data:
        return "", None, 0.0
    suffix = suffix if suffix.startswith(".") else "." + suffix
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        segments, info = _get_model().transcribe(
            path, beam_size=1, language=language, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500})
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language, float(info.duration)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
