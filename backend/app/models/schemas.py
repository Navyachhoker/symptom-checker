from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from enum import Enum


class UrgencyLevel(str, Enum):
    LOW       = "low"
    MODERATE  = "moderate"
    HIGH      = "high"
    EMERGENCY = "emergency"


class MessageRole(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


# ── Request models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         description="User's symptom description or reply")
    session_id: Optional[UUID] = Field(None,
                         description="Omit to start a new session")

    model_config = {"json_schema_extra": {
        "example": {
            "message": "I have a severe headache and my vision is blurry",
            "session_id": None
        }
    }}


# ── Response models ─────────────────────────────────────────

class MessageOut(BaseModel):
    id:         UUID
    role:       MessageRole
    content:    str
    order:      int
    created_at: datetime

    model_config = {"from_attributes": True}


class TriageOutcomeOut(BaseModel):
    id:               UUID
    urgency:          UrgencyLevel
    advice_text:      str
    symptoms_summary: Optional[str]
    created_at:       datetime

    model_config = {"from_attributes": True}


class SessionOut(BaseModel):
    id:         UUID
    created_at: datetime
    updated_at: Optional[datetime]
    message_count: Optional[int] = 0

    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    session_id:     UUID
    reply:          str                          # assistant's latest message
    triage_outcome: Optional[TriageOutcomeOut]   # set only when triage is complete
    is_complete:    bool = False                 # True = conversation is done

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    session_id:      UUID
    messages:        List[MessageOut]
    triage_outcome:  Optional[TriageOutcomeOut]

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    sessions: List[SessionOut]
    total:    int