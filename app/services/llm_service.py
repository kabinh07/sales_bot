from langchain_ollama import ChatOllama
import asyncio
from typing import AsyncGenerator, Dict, Any
import json
from app.utils.vd import VECTOR_STORE
from app.config import OLLAMA_LLM_NAME, OLLAMA_LLM_TEMPERATURE, OLLAMA_BASE_URL

class LLMService:
    def __init__(self):
        self.llm = ChatOllama(
            model=OLLAMA_LLM_NAME, 
            temperature=OLLAMA_LLM_TEMPERATURE, 
            base_url=OLLAMA_BASE_URL,
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
    
    def __retrieve_context(self, user_message: str, k: int = 3) -> str:
        docs = VECTOR_STORE.similarity_search(user_message, k=k)
        return "\n".join([doc.page_content for doc in docs])

    async def generate_response_stream(self, 
                                   user_message: str, 
                                   conversation_history: list,
                                   state: str) -> AsyncGenerator[str, None]:
        try:
            messages = []
            for entry in conversation_history:
                if entry["role"] == "user":
                    messages.append({"role": "user", "content": entry["content"]})
                else:
                    messages.append({"role": "assistant", "content": entry["content"]})

            messages.append({"role": "user", "content": user_message})

            prompt = self.prompts.get(state, self.prompts["introduction"])
            if state == "pitch":
                context = self.__retrieve_context(user_message)
                prompt = f"Context:\n{context}\n\n{prompt}"
            system_message = {"role": "system", "content": prompt}
            messages.insert(0, system_message)

            response_buffer = ""
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    response_buffer += chunk.content
                    yield chunk.content

        except Exception as e:
            print(f"Error in streaming LLM response: {e}")
            yield f"I apologize, but I'm having trouble processing your request. {str(e)}"
