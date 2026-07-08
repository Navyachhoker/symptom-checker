from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession
from db.database import get_db
from db import crud

router = APIRouter()

@router.get("/history/{session_id}")
def get_history(session_id: str, db: DBSession = Depends(get_db)):
    history = crud.get_history(db, session_id)
    return {"session_id": session_id, "messages": history}