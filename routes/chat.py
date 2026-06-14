from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession
from db.database import get_db
from db import crud
from schemas.models import ChatRequest, ChatResponse
from agent.graph import triage_graph

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: DBSession = Depends(get_db)):
    # 1. Ensures session exists in DB
    crud.create_session(db, request.session_id)

    # 2. Saves user message to DB
    crud.save_message(db, request.session_id, "user", request.message)

    # 3. Loads full history from DB
    history = crud.get_history(db, request.session_id)

    # 4. Runs LangGraph agent with current state
    result = triage_graph.invoke({
    "session_id": request.session_id,
    "messages": history,
    "symptoms": "",
    "stage": "assess",
    "reply": "",
    "urgency": "",
    "advice": "",
    "questions_asked": sum(1 for m in history if m["role"] == "assistant")  # counts how many times assistant already replied
})

    # 5. Saves agent reply to DB
    crud.save_message(db, request.session_id, "assistant", result["reply"])

    # 6. If triage is done, saves the outcome
    if result["stage"] == "done":
        crud.save_triage_outcome(db, request.session_id, result["urgency"], result["advice"])

    return ChatResponse(
        session_id=request.session_id,
        reply=result["reply"],
        stage=result["stage"]
    )