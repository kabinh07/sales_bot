from transformers import pipeline
from datasets import load_dataset
import scipy
import torch
import asyncio
import io
import numpy as np
from app.config import HF_TOKEN, HF_WHISPER_MODEL, HF_SPEECH_T5_NAME

class SpeechService:
    def __init__(self):
        self.stt = pipeline(
            "automatic-speech-recognition",
            model=HF_WHISPER_MODEL,
            token=HF_TOKEN,
            device=0
        )
        self.tts = pipeline(
            "text-to-speech",
            model=HF_SPEECH_T5_NAME,
            token=HF_TOKEN,
            device=0
        )
        # Load speaker embedding from HF dataset on init (can change index for different voices)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    async def process_audio_stream(self, audio_data: bytes):
        try:
            audio_file = io.BytesIO(audio_data)
            result = await asyncio.to_thread(self.stt, audio_file)
            text = result["text"] if "text" in result else ""
            yield text
        except Exception as e:
            print(f"Error in process_audio_stream: {e}")
            yield ""

    async def generate_speech_stream(self, text: str):
        try:
            # Pass speaker_embedding with forward_params as in the pipeline reference
            result = await asyncio.to_thread(
                self.tts,
                text,
                forward_params={"speaker_embeddings": self.speaker_embedding}
            )
            audio_array = result["audio"]  # This is a numpy array
            sample_rate = result.get("sampling_rate", 16000)  # Default to 16kHz if not present

            # Convert numpy array to WAV bytes
            with io.BytesIO() as wav_io:
                scipy.io.wavfile.write(wav_io, sample_rate, audio_array.astype(np.float32))
                wav_io.seek(0)
                yield wav_io.read()
        except Exception as e:
            print(f"Error in generate_speech_stream: {e}")
            yield b""
