from sqlalchemy import Column, String, Text, DateTime, Integer
from sqlalchemy.sql import func
from db.database import Base

# Table 1: one row per conversation session
class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=func.now())

# Table 2: every message in a conversation
class Message(Base):
    __tablename__ = "messages"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)           # links to Session
    role      = Column(String)            # "user" or "assistant"
    content   = Column(Text)             # the actual message text
    created_at = Column(DateTime, default=func.now())

# Table 3: final triage result when agent finishes
class TriageOutcome(Base):
    __tablename__ = "triage_outcomes"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    urgency    = Column(String)   # "low" / "medium" / "high"
    advice     = Column(Text)     # what the agent recommended
    created_at = Column(DateTime, default=func.now())