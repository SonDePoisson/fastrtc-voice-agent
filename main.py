from fastrtc import get_tts_model, ReplyOnPause, Stream
import numpy as np
from whisper import load_model, load_audio, Whisper
import ollama
import librosa

STT_MODEL = "small"
OLLAMA_MODEL = "ministral-3"


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


def response(audio: tuple[int, np.ndarray]):
    sample_rate, audio_data = audio
    user_prompt = stt_model.stt(audio_data, sample_rate)
    print(f"User: {user_prompt}")
    ollama_response = ollama.generate(model=OLLAMA_MODEL, prompt=user_prompt)
    print(f"Assistant: {ollama_response.response}")
    # for audio_chunk in tts_model.stream_tts_sync(prompt):
    #     yield audio_chunk


stt_model = STTModel()

stream = Stream(ReplyOnPause(response), modality="audio", mode="send")


if __name__ == "__main__":
    stream.ui.launch()
