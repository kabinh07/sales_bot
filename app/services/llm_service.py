from langchain_ollama import ChatOllama
import asyncio
from typing import AsyncGenerator, Dict, Any
import json

class LLMService:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen3:14b", 
            temperature=0.0, 
            base_url="http://192.168.27.207:11434",
            streaming=True
        )

        # Conversation prompts for different stages
        self.prompts = {
            "introduction": "You are a sales agent for our education company. Introduce yourself and the company briefly and professionally.",
            "qualification": "Based on the conversation so far, ask 2-3 relevant questions to understand the customer's needs and learning goals.",
            "pitch": "Based on the customer's responses, recommend the most suitable course and explain its benefits.",
            "objection_handling": "Address any concerns the customer has raised about the course, such as price, time commitment, or relevance.",
            "closing": "Try to schedule a follow-up call or get a commitment from the customer. Provide next steps."
        }

    async def generate_response_stream(self, 
                                   user_message: str, 
                                   conversation_history: list,
                                   state: str) -> AsyncGenerator[str, None]:
        try:
            # Format conversation history
            messages = []
            for entry in conversation_history:
                if entry["role"] == "user":
                    messages.append({"role": "user", "content": entry["content"]})
                else:
                    messages.append({"role": "assistant", "content": entry["content"]})

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            # Add system message with appropriate prompt for the current state
            system_message = {"role": "system", "content": self.prompts.get(state, self.prompts["introduction"])}
            messages.insert(0, system_message)

            # Stream the response
            response_buffer = ""
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    response_buffer += chunk.content
                    yield chunk.content

        except Exception as e:
            print(f"Error in streaming LLM response: {e}")
            yield f"I apologize, but I'm having trouble processing your request. {str(e)}"
