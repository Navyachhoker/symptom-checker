import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Integer,
    DateTime, ForeignKey, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.database import Base
import enum


class UrgencyLevel(str, enum.Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    EMERGENCY = "emergency"


class MessageRole(str, enum.Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


class Session(Base):
    __tablename__ = "sessions"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_identifier = Column(String(255), nullable=True)  # optional: IP, user_id, etc.

    # Relationships
    messages        = relationship("Message", back_populates="session",
                                   cascade="all, delete-orphan", order_by="Message.order")
    triage_outcomes = relationship("TriageOutcome", back_populates="session",
                                   cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session id={self.id} created_at={self.created_at}>"


class Message(Base):
    __tablename__ = "messages"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    role       = Column(SAEnum(MessageRole), nullable=False)
    content    = Column(Text, nullable=False)
    order      = Column(Integer, nullable=False)          # message position in conversation
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    session = relationship("Session", back_populates="messages")

    def __repr__(self):
        return f"<Message role={self.role} order={self.order}>"


class TriageOutcome(Base):
    __tablename__ = "triage_outcomes"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id  = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    urgency     = Column(SAEnum(UrgencyLevel), nullable=False)
    advice_text = Column(Text, nullable=False)
    symptoms_summary = Column(Text, nullable=True)    # brief summary of reported symptoms
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    session = relationship("Session", back_populates="triage_outcomes")

    def __repr__(self):
        return f"<TriageOutcome urgency={self.urgency}>"