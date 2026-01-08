"""Speech-to-Text backends for voice_agent."""

from abc import ABC, abstractmethod
import numpy as np
import librosa


class STTBackend(ABC):
    """Abstract base class for Speech-to-Text backends."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (int16 expected)
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass

    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Common audio preprocessing: convert to float32, normalize, resample to 16kHz."""
        audio = audio.astype(np.float32).flatten() / 32768.0
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        return audio


class WhisperSTTBackend(STTBackend):
    """STT backend using OpenAI Whisper."""

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        from whisper import load_model, Whisper

        self.model: Whisper = load_model(name=model_size, device=device)
        self._model_size = model_size

    @property
    def name(self) -> str:
        return f"whisper-{self._model_size}"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        audio = self._preprocess_audio(audio, sample_rate)
        result = self.model.transcribe(audio, fp16=False)
        return result["text"]


class FasterWhisperSTTBackend(STTBackend):
    """STT backend using faster_whisper (CTranslate2 based, ~4x faster)."""

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._model_size = model_size

    @property
    def name(self) -> str:
        return f"faster-whisper-{self._model_size}"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        audio = self._preprocess_audio(audio, sample_rate)
        segments, _ = self.model.transcribe(audio, beam_size=5)
        return "".join(segment.text for segment in segments)


class GroqSTTBackend(STTBackend):
    """STT backend using Groq Whisper API (cloud-based, very fast)."""

    def __init__(self, model: str = "whisper-large-v3-turbo", api_key: str | None = None):
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        from groq import Groq

        load_dotenv(Path(__file__).parents[3] / ".env")
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = Groq(api_key=self._api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"groq-{self._model}"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        import io
        import wave

        # Preprocess audio to 16kHz
        audio = self._preprocess_audio(audio, sample_rate)

        # Convert to int16 for WAV format
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create in-memory WAV file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())

        wav_buffer.seek(0)

        # Call Groq API
        transcription = self.client.audio.transcriptions.create(
            file=("audio.wav", wav_buffer, "audio/wav"),
            model=self._model,
        )
        return transcription.text


def create_stt_backend(
    backend: str = "faster_whisper",
    model_size: str | None = None,
    device: str | None = None,
    api_key: str | None = None,
) -> STTBackend:
    """Factory function to create an STT backend.

    Args:
        backend: Backend type ("whisper", "faster_whisper", or "groq")
        model_size: Model size (e.g., "tiny", "small", "medium", "large"). Defaults to "small".
        device: Device to run on ("cpu" or "cuda"). Defaults to "cpu".
        api_key: API key for cloud backends (e.g., Groq). Can also use env vars.

    Returns:
        Configured STT backend instance
    """
    model_size = model_size or "small"
    device = device or "cpu"

    if backend == "whisper":
        return WhisperSTTBackend(model_size=model_size, device=device)
    elif backend == "faster_whisper":
        return FasterWhisperSTTBackend(model_size=model_size, device=device)
    elif backend == "groq":
        groq_model = model_size if model_size != "small" else "whisper-large-v3-turbo"
        return GroqSTTBackend(model=groq_model, api_key=api_key)
    else:
        raise ValueError(f"Unknown STT backend: {backend}")
