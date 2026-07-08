from pydantic import BaseModel

# What the user sends in POST /chat
class ChatRequest(BaseModel):
    session_id: str       # unique ID for this conversation
    message: str          # what the user typed

# What we send back
class ChatResponse(BaseModel):
    session_id: str
    reply: str            # agent's response
    stage: str            # "assessing" or "done"