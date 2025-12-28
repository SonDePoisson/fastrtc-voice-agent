from fastrtc import ReplyOnPause, Stream
import numpy as np
from numpy.typing import NDArray
from whisper import load_model, load_audio, Whisper
import ollama
import librosa
import edge_tts
import asyncio
import io
import re
from pydub import AudioSegment
from typing import Generator, AsyncGenerator
from dataclasses import dataclass

STT_MODEL = "small"
OLLAMA_MODEL = "ministral-3"
TTS_VOICE = "en-US-AvaMultilingualNeural"


class STTModel:
    def __init__(self):
        self.model: Whisper = load_model(name=STT_MODEL, device="cpu")

    def stt(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        # Convert int16 to float32 and normalize to [-1, 1]
        audio = audio.astype(np.float32).flatten() / 32768.0
        # Resample to 16kHz (Whisper's expected sample rate)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        result = self.model.transcribe(audio, fp16=False)
        return result["text"]

    def load_audio(self, path: str) -> np.ndarray:
        return load_audio(path)


@dataclass
class EdgeTTSOptions:
    voice: str = TTS_VOICE
    rate: str = "+0%"
    pitch: str = "+0Hz"


class EdgeTTSModel:
    SAMPLE_RATE = 24000

    def __init__(self, voice: str = TTS_VOICE):
        self.voice = voice

    def _decode_mp3(self, mp3_bytes: bytes) -> NDArray[np.float32]:
        """Decode MP3 bytes to numpy array at 24kHz mono float32."""
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        audio = audio.set_frame_rate(self.SAMPLE_RATE).set_channels(1)
        return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

    async def _generate_sentence(self, text: str, options: EdgeTTSOptions) -> bytes:
        """Generate audio bytes for a single sentence."""
        communicate = edge_tts.Communicate(text, options.voice, rate=options.rate, pitch=options.pitch)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return audio_bytes

    def tts(self, text: str, options: EdgeTTSOptions | None = None) -> tuple[int, NDArray[np.float32]]:
        """Generate complete audio from text."""
        options = options or EdgeTTSOptions(voice=self.voice)
        loop = asyncio.new_event_loop()
        try:
            audio_bytes = loop.run_until_complete(self._generate_sentence(text, options))
            return self.SAMPLE_RATE, self._decode_mp3(audio_bytes)
        finally:
            loop.close()

    async def stream_tts(
        self, text: str, options: EdgeTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        """Async generator yielding audio chunks per sentence for lower latency."""
        options = options or EdgeTTSOptions(voice=self.voice)

        # Split by sentences for faster first-chunk delivery
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Generate and decode audio for this sentence
            audio_bytes = await self._generate_sentence(sentence, options)
            audio = self._decode_mp3(audio_bytes)

            # Yield in chunks
            chunk_size = self.SAMPLE_RATE // 5  # 200ms chunks
            for i in range(0, len(audio), chunk_size):
                yield self.SAMPLE_RATE, audio[i : i + chunk_size]

    def stream_tts_sync(
        self, text: str, options: EdgeTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        """Sync generator yielding audio chunks."""
        loop = asyncio.new_event_loop()
        iterator = self.stream_tts(text, options).__aiter__()
        try:
            while True:
                try:
                    yield loop.run_until_complete(iterator.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()


def response(audio: tuple[int, np.ndarray]):
    sample_rate, audio_data = audio
    user_prompt = stt_model.stt(audio_data, sample_rate)
    print(f"User: {user_prompt}")
    ollama_response = ollama.generate(model=OLLAMA_MODEL, prompt=user_prompt)
    print(f"Assistant: {ollama_response.response}")
    for audio_chunk in tts_model.stream_tts_sync(ollama_response.response):
        yield audio_chunk


stt_model = STTModel()
tts_model = EdgeTTSModel()

stream = Stream(ReplyOnPause(response), modality="audio", mode="send-receive")


if __name__ == "__main__":
    stream.ui.launch()
