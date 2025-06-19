from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from .models import CallRequest, CallResponse, MessageRequest, MessageResponse, ConversationHistory
from .services.conversation_manager import ConversationManager
from .services.conversation_manager import ConversationManager
from typing import Dict, List, AsyncGenerator, Union, Any
import asyncio
import json
import io

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Templates directory
templates = Jinja2Templates(directory=Path(__file__).parent / "static")

@app.get("/", response_class=HTMLResponse)
async def get_client(request: Request):
    return templates.TemplateResponse("client.html", {"request": request})

# Initialize managers
conversation_manager = ConversationManager()
streaming_manager = ConversationManager()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, call_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[call_id] = websocket

    def disconnect(self, call_id: str):
        if call_id in self.active_connections:
            del self.active_connections[call_id]

    async def send_audio(self, call_id: str, audio_data: bytes):
        if call_id in self.active_connections:
            await self.active_connections[call_id].send_bytes(audio_data)

    async def send_text(self, call_id: str, message: str):
        if call_id in self.active_connections:
            await self.active_connections[call_id].send_text(message)

manager = ConnectionManager()

# Regular REST endpoints
@app.post("/start-call", response_model=CallResponse)
async def start_call(request: CallRequest):
    try:
        response = await streaming_manager.start_conversation(
            request.phone_number,
            request.customer_name
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/respond/{call_id}", response_model=MessageResponse)
async def respond(call_id: str, request: MessageRequest):
    try:
        # Use non-streaming version for compatibility
        response_text = ""
        should_end = False

        async for chunk in streaming_manager.process_text_message(call_id, request.message):
            response_text += chunk

        # Check if should end call
        should_end = call_id in streaming_manager.conversations and \
                   len(streaming_manager.conversations[call_id]) > 6 and \
                   streaming_manager._should_end_conversation(response_text)

        return {
            "reply": response_text,
            "should_end_call": should_end
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{call_id}", response_model=ConversationHistory)
async def get_conversation(call_id: str):
    try:
        # Both managers share the same interface for this method
        return streaming_manager.get_conversation_history(call_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints for streaming
@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await manager.connect(call_id, websocket)
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive()

            # Handle different message types
            if "bytes" in data:  # Audio data
                audio_data = data["bytes"]

                # Process the audio stream and stream back responses
                async for response in streaming_manager.process_audio_message(call_id, audio_data):
                    if isinstance(response, bytes):
                        await manager.send_audio(call_id, response)
                    else:
                        await manager.send_text(call_id, response)

            elif "text" in data:  # Text data
                message = data["text"]

                # Check if it's a JSON message
                try:
                    msg_data = json.loads(message)
                    # Handle control messages or commands
                    if "command" in msg_data:
                        if msg_data["command"] == "end_call":
                            await manager.send_text(call_id, json.dumps({"status": "call_ended"}))
                            break
                except json.JSONDecodeError:
                    # Treat as regular text message
                    async for text_chunk in streaming_manager.process_text_message(call_id, message):
                        await manager.send_text(call_id, text_chunk)

    except WebSocketDisconnect:
        manager.disconnect(call_id)
    except Exception as e:
        # Send error message and disconnect
        try:
            await manager.send_text(call_id, json.dumps({"error": str(e)}))
        except:
            pass
        manager.disconnect(call_id)
