"""
piper_tts.py — The "regular Sam voice" synthesizer.

Piper is MIT-licensed, ~70x realtime on CPU, no model download required.
This is the same voice (`en_US-libritts_r-medium`) that Hermes uses for
Telegram voice memos via ~/.hermes/scripts/sam_tts.py.

We shell out to the standalone piper CLI from
~/Desktop/LIFE_MEMORY/PROJECTS/VOICE_AGENT/local_agent/venv311/bin/piper
because that venv is already wired up with all of Piper's native deps
(onnxruntime). Keeps the voice venv light.

Output: 22050Hz mono s16le WAV bytes.
"""
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger("sam.voice.piper")

# Default voice = the "regular Sam voice" Justin picked on 2026-06-27
# after A/B testing 7 Piper voices. Warm mid-pitch, soft + confident female.
DEFAULT_MODEL = Path(os.path.expanduser("~/.hermes/piper/en_US-libritts_r-medium.onnx"))
DEFAULT_CONFIG = DEFAULT_MODEL.with_suffix(".onnx.json")

# Path to the piper CLI (already-installed in LIFE_MEMORY venv311)
PIPER_BIN = Path(
    "/Users/samintelligence/Desktop/LIFE_MEMORY/PROJECTS/VOICE_AGENT/local_agent/venv311/bin/piper"
)

# 22050Hz mono s16le — matches the model card.
SAMPLE_RATE = 22050


def is_available() -> bool:
    """True if piper CLI + model files are reachable."""
    return PIPER_BIN.exists() and DEFAULT_MODEL.exists() and DEFAULT_CONFIG.exists()


def synth_to_wav_bytes(text: str, voice: str = None, speed: float = 1.0) -> bytes:
    """
    Synthesize text → 22050Hz mono s16le WAV bytes via Piper CLI.

    Args:
        text:  what to say
        voice: unused (piper models are single-speaker; passed for API compat with kokoro_tts)
        speed: length scale — 1.0 is normal, lower = faster, higher = slower
               Piper calls it --length-scale (default 1.0).
    """
    if not text or not text.strip():
        return b""
    if not is_available():
        raise FileNotFoundError(
            f"piper unavailable: bin={PIPER_BIN.exists()} model={DEFAULT_MODEL.exists()}"
        )

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        # Build piper command. --length-scale inverts intuition: 1.0 is default,
        # 0.5 = twice as fast, 2.0 = twice as slow. We invert `speed` so callers
        # can pass the same `speed=1.2` as kokoro_tts and get the same result.
        length_scale = (1.0 / speed) if speed > 0 else 1.0
        proc = subprocess.run(
            [
                str(PIPER_BIN),
                "--model", str(DEFAULT_MODEL),
                "--config", str(DEFAULT_CONFIG),
                "--output_file", tmp_wav,
                "--length-scale", f"{length_scale:.3f}",
                "--noise-scale", "0.333",
                "--noise-w", "0.333",
            ],
            input=text,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace")[:500]
            raise RuntimeError(f"piper failed (rc={proc.returncode}): {stderr}")
        wav_bytes = Path(tmp_wav).read_bytes()
        log.info(
            "piper: %d bytes (%.2fs audio) for %d chars in %.2fs",
            len(wav_bytes),
            len(wav_bytes) / (SAMPLE_RATE * 2),  # 16-bit = 2 bytes/sample
            len(text),
            time.time() - t0,
        )
        return wav_bytes
    finally:
        try:
            Path(tmp_wav).unlink(missing_ok=True)
        except Exception:
            pass
