from pydantic import BaseModel
from typing import List, Optional, Dict

class CallRequest(BaseModel):
    phone_number: str
    customer_name: str

class CallResponse(BaseModel):
    call_id: str
    message: str
    first_message: str

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    reply: str
    should_end_call: bool

class ConversationHistory(BaseModel):
    call_id: str
    history: List[Dict[str, str]]