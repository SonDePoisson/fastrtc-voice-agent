from fastrtc import get_tts_model, ReplyOnPause, Stream
import numpy as np
from whisper import load_model, load_audio, Whisper

STT_MODEL = "small"


class STTModel:
    def __init__(self):
        self.model: Whisper = load_model(name=STT_MODEL, device="cpu")

    def stt(self, audio: np.ndarray) -> str:
        result = self.model.transcribe(audio, fp16=False)
        return result["text"]

    def load_audio(self, path: str) -> np.ndarray:
        return load_audio(path)


def main():
    # Create the model (downloads from HF if not cached)
    stt_model = STTModel()

    # print("Recording ...")
    # audio_data = np.zeros(3 * 16000, dtype=np.float16)  # 3 second of silence

    audio_data = stt_model.load_audio("Spiderman_AudioTestFile.mp3")

    # Transcribe
    text = stt_model.stt(audio_data)
    print(f"Transcription: {text}")


if __name__ == "__main__":
    main()
