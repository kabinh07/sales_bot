import torch
import torchaudio
import whisper
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import numpy as np
from typing import AsyncGenerator, Union
import asyncio
import io
import tempfile
from pathlib import Path
from app.config import HF_WHISPER_MODEL, HF_SPEECH_T5_NAME

class SpeechService:
    def __init__(self):
        # Initialize Whisper for STT
        self.whisper_model = whisper.load_model("base")

        # Initialize SpeechT5 for TTS
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Create speaker embeddings (needed for generation)
        self.speaker_embeddings = torch.zeros(1, 512)

        # Audio processing parameters
        self.sample_rate = 16000
        self.chunk_size = 4096

    async def process_audio_stream(self, audio_data: bytes) -> AsyncGenerator[str, None]:
        try:
            # Save audio data to a temporary file for Whisper processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)

            # Process with Whisper
            result = await asyncio.to_thread(
                self.whisper_model.transcribe, 
                temp_path
            )

            # Clean up temporary file
            Path(temp_path).unlink()

            if result and "text" in result:
                yield result["text"]
        except Exception as e:
            print(f"Error processing audio stream: {e}")
            yield ""

    async def generate_speech_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        try:
            # Generate speech in a non-blocking way
            inputs = self.processor(text=text, return_tensors="pt")

            speech = await asyncio.to_thread(
                self.model.generate_speech,
                inputs["input_ids"], 
                self.speaker_embeddings
            )

            # Convert to waveform with vocoder
            waveform = await asyncio.to_thread(self.vocoder, speech)

            # Split into chunks for streaming
            chunks = torch.split(waveform.squeeze(0), self.chunk_size)

            for chunk in chunks:
                if chunk.numel() > 0:  # Ensure chunk is not empty
                    # Convert to bytes
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, chunk.unsqueeze(0), self.sample_rate, format="wav")
                    yield buffer.getvalue()

                    # Small delay to simulate streaming
                    await asyncio.sleep(0.05)

        except Exception as e:
            print(f"Error generating speech stream: {e}")
            yield b""
