from uuid import uuid4
from typing import Dict, List, AsyncGenerator, Union, Any
import asyncio
import os
from .speech_service import SpeechService
from .llm_service import LLMService
from langchain.prompts import PromptTemplate

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.speech_service = SpeechService()
        self.llm_service = LLMService()

    async def start_conversation(self, phone_number: str, customer_name: str):
        call_id = str(uuid4())
        self.conversations[call_id] = []

        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "prompts", "initial_prompt.txt"
        )
        with open(prompt_path, "r") as f:
            prompt = f.read()

        initial_prompt = PromptTemplate(
            input_variables=["customer_name"],
            template=prompt
        )
        initial_prompt = initial_prompt.format(customer_name=customer_name)
        greeting = ""
        async for chunk in self.llm_service.generate_response_stream(
            initial_prompt, [], "introduction"
        ):
            greeting += chunk
        greeting = greeting.split("</think>")[-1].strip()
        self.conversations[call_id].append({
            "role": "assistant",
            "content": greeting
        })

        # Return both call_id and greeting text
        return call_id, greeting
    
    async def process_audio_message(self, 
                                 call_id: str, 
                                 audio_data: bytes) -> AsyncGenerator[Union[str, bytes], None]:
        if call_id not in self.conversations:
            raise ValueError("Invalid call ID")

        # Process speech to text
        user_message = ""
        async for text_chunk in self.speech_service.process_audio_stream(audio_data):
            user_message += text_chunk

        if not user_message:
            yield "I couldn't hear you clearly. Could you please repeat?"
            return

        # Add user message to history
        self.conversations[call_id].append({
            "role": "user",
            "content": user_message
        })

        # Generate text response
        current_state = self._determine_state(call_id)

        # Generate and stream the response
        assistant_response = ""
        async for text_chunk in self.llm_service.generate_response_stream(
            user_message, 
            self.conversations[call_id],
            current_state
        ):
            assistant_response += text_chunk

            # Convert this chunk to speech
            async for audio_chunk in self.speech_service.generate_speech_stream(text_chunk):
                yield audio_chunk

            # Also yield the text for display
            yield text_chunk

        # Store full response in conversation history
        self.conversations[call_id].append({
            "role": "assistant",
            "content": assistant_response
        })

    async def process_text_message(self, 
                               call_id: str, 
                               message: str) -> AsyncGenerator[str, None]:
        if call_id not in self.conversations:
            raise ValueError("Invalid call ID")

        # Add user message to history
        self.conversations[call_id].append({
            "role": "user",
            "content": message
        })

        # Determine current state and generate response
        current_state = self._determine_state(call_id)

        # Generate and stream the response
        assistant_response = ""
        async for text_chunk in self.llm_service.generate_response_stream(
            message, 
            self.conversations[call_id],
            current_state
        ):
            assistant_response += text_chunk
            yield text_chunk

        # Store full response in conversation history
        self.conversations[call_id].append({
            "role": "assistant",
            "content": assistant_response
        })

    def get_conversation_history(self, call_id: str) -> Dict:
        if call_id not in self.conversations:
            raise ValueError("Invalid call ID")

        return {
            "call_id": call_id,
            "history": self.conversations[call_id]
        }

    def _determine_state(self, call_id: str) -> str:
        # Simple state determination based on conversation length
        history_length = len(self.conversations[call_id])
        if history_length <= 2:
            return "qualification"
        elif history_length <= 4:
            return "pitch"
        elif history_length <= 6:
            return "objection_handling"
        else:
            return "closing"

    def _should_end_conversation(self, response: str) -> bool:
        # Logic to determine if conversation should end
        return "schedule" in response.lower() or "follow up" in response.lower()
