"""TTS module for voice_agent."""

from .tts import (
    TTSBackend,
    EdgeTTSOptions,
    EdgeTTSBackend,
    create_tts_backend,
)

__all__ = [
    "TTSBackend",
    "EdgeTTSOptions",
    "EdgeTTSBackend",
    "create_tts_backend",
]
