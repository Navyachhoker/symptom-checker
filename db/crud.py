from db.models import Session, Message, TriageOutcome

# Create a new session if it doesn't exist
def create_session(db, session_id: str):
    existing = db.query(Session).filter(Session.session_id == session_id).first()
    if not existing:
        db.add(Session(session_id=session_id))
        db.commit()

# Save a single message (user or assistant)
def save_message(db, session_id: str, role: str, content: str):
    db.add(Message(session_id=session_id, role=role, content=content))
    db.commit()

# Get all messages for a session — returns list of dicts
def get_history(db, session_id: str):
    messages = db.query(Message).filter(Message.session_id == session_id).all()
    return [{"role": m.role, "content": m.content} for m in messages]

# Save the final triage result
def save_triage_outcome(db, session_id: str, urgency: str, advice: str):
    db.add(TriageOutcome(session_id=session_id, urgency=urgency, advice=advice))
    db.commit()