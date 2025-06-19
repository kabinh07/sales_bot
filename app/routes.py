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
    Accepts audio input, transcribes it, sends it to LLM, and returns the response as audio.
    """
    try:
        audio_bytes = await file.read()
        
        # Transcribe user's audio - process_audio_message should return transcribed text
        recognized_chunks = []
        async for chunk in conversation_manager.process_audio_message(call_id, audio_bytes):
            # Handle different chunk types from STT service
            if isinstance(chunk, str):
                recognized_chunks.append(chunk)
            elif isinstance(chunk, bytes):
                # Only decode if you're certain these are text bytes, not audio bytes
                try:
                    decoded_text = chunk.decode("utf-8")
                    recognized_chunks.append(decoded_text)
                except UnicodeDecodeError:
                    # If decode fails, this might be audio data that shouldn't be decoded
                    print(f"Warning: Received non-text bytes in STT response: {len(chunk)} bytes")
                    continue
            else:
                print(f"Warning: Unexpected chunk type in STT: {type(chunk)}")
                continue
        
        recognized_text = ''.join(recognized_chunks)
        recognized_text = recognized_text.split("</think>")[-1].strip()
        print(f"Recognized text: {recognized_text}")
        
        # LLM generates response
        response_chunks = []
        async for chunk in conversation_manager.process_text_message(call_id, recognized_text):
            if isinstance(chunk, str):
                response_chunks.append(chunk)
            elif isinstance(chunk, bytes):
                try:
                    decoded = chunk.decode("utf-8")
                    response_chunks.append(decoded)
                except UnicodeDecodeError:
                    print(f"Warning: Received invalid UTF-8 bytes from LLM: {repr(chunk[:40])}")
                    continue
            elif hasattr(chunk, "content"):  # e.g., LangChain AIMessage
                response_chunks.append(chunk.content)
            else:
                print(f"Warning: Unknown chunk type from LLM: {type(chunk)}")
                continue
        
        response_text = ''.join(response_chunks)
        print(f"Response text: {response_text}")
        
        # Convert agent response to audio
        audio_chunk_list = []
        async for chunk in conversation_manager.speech_service.generate_speech_stream(response_text):
            if isinstance(chunk, bytes):
                audio_chunk_list.append(chunk)
            elif isinstance(chunk, str):
                # If TTS returns string, encode it to bytes
                audio_chunk_list.append(chunk.encode("utf-8"))
            else:
                print(f"Warning: Unexpected chunk type in TTS: {type(chunk)}")
                continue
        
        response_audio_bytes = b''.join(audio_chunk_list)
        print(f"Response audio bytes: {len(response_audio_bytes)}")
        
        return StreamingResponse(
            io.BytesIO(response_audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=reply_{call_id}.wav"}
        )
    
    except Exception as e:
        print(f"Error in respond endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")