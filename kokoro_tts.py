#!/usr/bin/env python3
"""
kokoro_tts.py — Text-to-speech via Kokoro-82M.

Lazily loads the model on first call. ~7s cold start, then fast.
Returns WAV bytes (24kHz mono PCM) ready to stream or send.
"""
import io
import logging
import threading
import time

import numpy as np
import soundfile as sf
from kokoro import KPipeline

log = logging.getLogger("sam.voice.tts")

_pipeline = None
_lock = threading.Lock()

DEFAULT_VOICE = "am_adam"  # default male voice. Other options: af_heart, bf_emma, am_michael
# sam-voice can switch providers at runtime via env SAM_VOICE_TTS_PROVIDER. We default to Kokoro
# for low-latency local synthesis. The "regular Sam voice" you hear in Hermes chat
# (text_to_speech tool) is Edge TTS en-US-AriaNeural — that path is available via
# provider="edge" if you set SAM_VOICE_TTS_PROVIDER=edge.


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                log.info("loading Kokoro KPipeline (lang_code='a')...")
                t0 = time.time()
                _pipeline = KPipeline(lang_code='a')
                log.info("Kokoro loaded in %.1fs", time.time() - t0)
    return _pipeline


def synth_to_wav_bytes(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """Synthesize text → WAV bytes (24kHz mono PCM)."""
    if not text or not text.strip():
        return b""
    pipeline = _get_pipeline()
    t0 = time.time()
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio)
    if not chunks:
        log.warning("Kokoro produced no chunks for text: %r", text[:60])
        return b""
    audio_concat = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, audio_concat, 24000, format='WAV', subtype='PCM_16')
    buf.seek(0)
    wav_bytes = buf.read()
    log.info(
        "tts: %.2fs audio for %d chars in %.2fs (voice=%s)",
        len(audio_concat) / 24000, len(text), time.time() - t0, voice,
    )
    return wav_bytes


def synth_to_pcm_bytes(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0,
                       sample_rate: int = 24000) -> bytes:
    """Synthesize text → raw PCM s16le mono bytes (no WAV header)."""
    if not text or not text.strip():
        return b""
    pipeline = _get_pipeline()
    t0 = time.time()
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio)
    if not chunks:
        return b""
    audio_concat = np.concatenate(chunks)
    # Float32 → int16
    pcm = (audio_concat * 32767).astype(np.int16).tobytes()
    log.info(
        "tts-pcm: %d bytes (%d samples, %.2fs) in %.2fs",
        len(pcm), len(audio_concat), len(audio_concat) / sample_rate, time.time() - t0,
    )
    return pcm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    print("=== testing kokoro_tts.synth_to_wav_bytes ===")
    wav = synth_to_wav_bytes("Hello Justin. This is sam-voice. Voice synthesis is working.")
    print(f"  produced {len(wav)} bytes of WAV")
    assert len(wav) > 1000
    with open("/tmp/sam-voice-test/kokoro_test.wav", "wb") as f:
        f.write(wav)
    print(f"  saved to /tmp/sam-voice-test/kokoro_test.wav")
    print("  PASS")
