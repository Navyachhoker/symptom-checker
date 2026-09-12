from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Any
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


class ChatRequest(BaseModel):
    message:    str            = Field(..., min_length=1, max_length=2000)
    session_id: Optional[UUID] = Field(None)


class MessageOut(BaseModel):
    id:         UUID
    role:       MessageRole
    content:    str
    order:      int
    created_at: datetime
    model_config = {"from_attributes": True}


class TriageOutcomeOut(BaseModel):
    id:                UUID
    urgency:           UrgencyLevel
    confidence:        Optional[int] = None
    advice_text:       str
    symptoms_summary:  Optional[str]
    specialist_called: Optional[str] = None
    created_at:        datetime
    model_config = {"from_attributes": True}


class SessionOut(BaseModel):
    id:            UUID
    created_at:    datetime
    updated_at:    Optional[datetime]
    message_count: Optional[int] = 0
    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    session_id:     UUID
    reply:          str
    triage_outcome: Optional[TriageOutcomeOut]
    is_complete:    bool          = False
    agent_trace:    List[Any]     = []
    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    session_id:     UUID
    messages:       List[MessageOut]
    triage_outcome: Optional[TriageOutcomeOut]


class SessionListResponse(BaseModel):
    sessions: List[SessionOut]
    total:    int