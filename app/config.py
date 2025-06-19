import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_LLM_NAME = os.getenv("OLLAMA_LLM_NAME", "qwen3:14b")
OLLAMA_LLM_TEMPERATURE = float(os.getenv("OLLAMA_LLM_TEMPERATURE", "0.0"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
HF_WHISPER_MODEL = os.getenv("HF_WHISPER_MODEL", "openai/whisper-base")
HF_SPEECH_T5_NAME = os.getenv("HF_SPEECH_T5_NAME", "microsoft/speecht5_tts")
HF_TOKEN= os.getenv("HF_TOKEN", "")
