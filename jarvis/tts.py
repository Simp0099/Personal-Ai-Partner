"""Local Kokoro text-to-speech with punctuation pauses."""
from __future__ import annotations

import re
import threading
from typing import Iterator, Optional, Tuple

import numpy as np

from jarvis.config import KOKORO_LANG, KOKORO_SPEED, KOKORO_VOICE
from jarvis.logger import logger

KOKORO_RATE = 24000
_pipeline = None
_pipeline_lock = threading.Lock()
_BREAK = re.compile(r"\.\.\.+|…|—|–|[.!?](?=\s|$)|[,;:](?=\s|$)")
_PAUSE = {"…": .45, "—": .30, "–": .30, ".": .30, "!": .30, "?": .32,
          ",": .14, ";": .18, ":": .18}


def _segments(text: str) -> Iterator[Tuple[str, float]]:
    position = 0
    for match in _BREAK.finditer(text):
        piece = text[position:match.end()].strip()
        if piece:
            token = match.group(0)
            yield piece, .45 if token.startswith("...") else _PAUSE.get(token, .2)
        position = match.end()
    tail = text[position:].strip()
    if tail:
        yield tail, 0.0


def _get_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            try:
                from kokoro import KPipeline
            except ImportError as e:
                raise RuntimeError("Kokoro is not installed. Run: pip install kokoro soundfile") from e
            logger.info("Loading Kokoro TTS (lang=%s, voice=%s)", KOKORO_LANG, KOKORO_VOICE)
            _pipeline = KPipeline(lang_code=KOKORO_LANG)
    return _pipeline


def synthesize(text: str, exaggeration: Optional[float] = None) -> np.ndarray:
    """Return mono float32 Kokoro audio at 24 kHz, with punctuation pauses."""
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.float32)
    speed = KOKORO_SPEED if exaggeration is None else max(.85, min(1.15, KOKORO_SPEED + (float(exaggeration) - .5) * .3))
    parts = []
    for piece, pause in _segments(text):
        if re.search(r"\w", piece):
            for _, _, audio in _get_pipeline()(piece, voice=KOKORO_VOICE, speed=speed):
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if pause:
            parts.append(np.zeros(int(KOKORO_RATE * pause), dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def warm_up() -> None:
    try:
        synthesize("Ready.")
    except Exception as e:  # noqa: BLE001
        logger.warning("Kokoro warm-up failed: %s", e)
