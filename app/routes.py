from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from .services.conversation_manager import ConversationManager
from .models import CallRequest, CallResponse, MessageRequest, MessageResponse, ConversationHistory
import io

router = APIRouter()
conversation_manager = ConversationManager()

@router.post("/start-call")
async def start_call(request: CallRequest):
    """
    Starts a new call using provided phone number and customer name.
    Returns the conversation's first agent message as audio.
    """
    try:
        # Get call_id and greeting text
        call_id, greeting_text = await conversation_manager.start_conversation(
            request.phone_number, request.customer_name
        )

        # Convert greeting text to audio
        audio_chunks = []
        async for chunk in conversation_manager.speech_service.generate_speech_stream(greeting_text):
            audio_chunks.append(chunk)
        audio_bytes = b"".join(audio_chunks)

        print(f"greeting text: {greeting_text}")
        print(f"Audio chunks: {len(audio_chunks)}, bytes: {len(audio_bytes)}")

        # Return audio as StreamingResponse with call_id in header
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"X-Call-Id": call_id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/respond/{call_id}")
async def respond(call_id: str, file: UploadFile = File(...)):
    """
    Accepts an audio input for a given call_id, then uses LLM and TTS to return the agent's response as audio.
    """
    try:
        audio_bytes = await file.read()

        # 1. Recognize user's audio message as text.
        recognized_text = ""
        async for chunk in conversation_manager.process_audio_message(call_id, audio_bytes):
            recognized_text += chunk

        # 2. Process this text through the LLM to get a response.
        response_text = ""
        async for chunk in conversation_manager.process_text_message(call_id, recognized_text):
            response_text += chunk

        # 3. Convert agent's response text back to audio.
        response_audio = await conversation_manager.speech_service.generate_speech_stream(response_text)

        return StreamingResponse(
            io.BytesIO(b"".join([chunk async for chunk in response_audio])),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=reply_{call_id}.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
