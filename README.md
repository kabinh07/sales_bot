# Sales Bot

A voice-enabled AI sales agent for education products, powered by LLMs, Whisper, and Qdrant.  
Supports speech-to-text, LLM-driven conversation, and text-to-speech, all via API and a web frontend.

---

## 🚀 Quick Start (Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/kabinh07/sales_bot.git
cd sales_bot
```

### 2. Configure Environment

Copy the example environment file and fill in your HuggingFace and Ollama credentials:

```bash
cp app/.env.example app/.env
# Edit app/.env with your keys and model names
```

### 3. Build and Run with Docker Compose

```bash
docker-compose up --build
```

- The FastAPI app will be available at [http://localhost:8000](http://localhost:8000)
- Qdrant vector DB will run at [http://localhost:6333](http://localhost:6333)

---

## 🌐 Web Frontend

The HTML client is served automatically by FastAPI.

- Visit: [http://localhost:8000/static/client.html](http://localhost:8000/static/client.html)
- Use the interface to start a call, send text or voice messages, and interact with the AI agent.

---

## 🛠️ API Documentation

FastAPI provides interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Key Endpoints

#### `POST /start-call`

Start a new conversation.

**Request Body (JSON):**
```json
{
  "customer_name": "Alice",
  "phone_number": "+123456789"
}
```

**Response:**
- Returns a WAV audio stream (greeting message).
- `X-Call-Id` header contains the unique call ID.

#### `POST /respond/{call_id}`

Send an audio message and get the agent's audio reply.

**Request:**
- `call_id`: The call ID from `/start-call`.
- `file`: Audio file (WAV/OGG/Opus) as multipart/form-data.

**Example with `curl`:**
```bash
curl -X POST -F "file=@your_audio.wav" http://localhost:8000/respond/<call_id> --output reply.wav
```

**Response:**
- Returns a WAV audio stream (agent's reply).

---

## 📝 Environment Variables

Set these in `app/.env`:

| Variable              | Description                                 |
|-----------------------|---------------------------------------------|
| OLLAMA_LLM_NAME       | Ollama LLM model name                       |
| OLLAMA_LLM_TEMPERATURE| LLM temperature (float)                     |
| OLLAMA_BASE_URL       | Ollama server URL                           |
| HF_WHISPER_MODEL      | HuggingFace Whisper model name              |
| HF_SPEECH_T5_NAME     | HuggingFace SpeechT5 model name             |
| HF_TOKEN              | HuggingFace API token                       |
| QDRANT_URL            | Qdrant DB URL                               |
| EMB_MODEL_NAME        | Embedding model for Qdrant                  |

---

## 🗂️ Project Structure

```
app/
  services/         # Core logic: LLM, speech, conversation
  static/           # client.html (web frontend)
  utils/            # Qdrant vector store setup
  routes.py         # FastAPI endpoints
  main.py           # App entrypoint
  config.py         # Environment config
docker-compose.yaml
Dockerfile
requirements.txt
```

---

## 🧑‍💻 Development

- To run locally without Docker, install requirements and run `python app/main.py`.
- For API testing, use `/docs` or tools like Postman.

---

## 📝 Notes

- The web client (`client.html`) must be accessed via the `/static/client.html` endpoint, **not** as a local file.
- All CORS is enabled by default for easy integration.

---

## 📞 Support

For issues or feature requests, please open an issue on GitHub.

---