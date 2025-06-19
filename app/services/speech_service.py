from transformers import pipeline
from datasets import load_dataset
import scipy
import torch
import torchaudio
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
            # Load audio from bytes (supports OGG, WAV, etc.)
            audio_file = io.BytesIO(audio_data)
            waveform, sample_rate = torchaudio.load(audio_file)
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample to 16000 Hz if needed
            target_sr = 16000
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                waveform = resampler(waveform)
            audio_np = waveform.squeeze().numpy()
            # Pass only the numpy array to the pipeline (no sampling_rate kwarg)
            result = await asyncio.to_thread(self.stt, audio_np)
            text = result["text"] if "text" in result else ""
            print(f"text: {text}")
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
