from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from langchain_core.messages import HumanMessage, AIMessage
from uuid import UUID, uuid4
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.db.database import get_db
from app.db.models import Session as DBSession, Message, TriageOutcome, MessageRole, UrgencyLevel
from app.models.schemas import (
    ChatRequest, ChatResponse,
    HistoryResponse, MessageOut,
    TriageOutcomeOut, SessionOut, SessionListResponse,
)
from app.agent import triage_graph, TriageState

router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def get_user_token(x_user_token: Optional[str] = Header(None)) -> Optional[str]:
    return x_user_token


async def load_conversation(session_id: UUID, db: AsyncSession):
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.order)
    )
    messages = result.scalars().all()
    lc_messages = []
    for m in messages:
        if m.role == MessageRole.USER:
            lc_messages.append(HumanMessage(content=m.content))
        elif m.role == MessageRole.ASSISTANT:
            lc_messages.append(AIMessage(content=m.content))
    return messages, lc_messages


async def save_message(db, session_id, role, content, order):
    msg = Message(
        id=uuid4(),
        session_id=session_id,
        role=role,
        content=content,
        order=order,
    )
    db.add(msg)
    return msg


# ── POST /chat — 20/minute ────────────────────────────────────
@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    body: ChatRequest,          # ← renamed from request to body
    request: Request,           # ← this is now the FastAPI Request for slowapi
    db: AsyncSession = Depends(get_db),
    user_token: Optional[str] = Depends(get_user_token),
):
    if body.session_id:         # ← use body.session_id
        session_result = await db.execute(
            select(DBSession).where(DBSession.id == body.session_id)
        )
        db_session = session_result.scalar_one_or_none()
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found")
        if user_token and db_session.user_identifier != user_token:
            raise HTTPException(status_code=403, detail="Access denied")
    else:
        db_session = DBSession(
            id=uuid4(),
            user_identifier=user_token or request.client.host,
        )
        db.add(db_session)
        await db.flush()

    session_id = db_session.id
    existing_msgs, lc_history = await load_conversation(session_id, db)
    current_order = len(existing_msgs)

    await save_message(db, session_id, MessageRole.USER, body.message, current_order)
    current_order += 1

    agent_state: TriageState = {
        "messages":            lc_history + [HumanMessage(content=body.message)],
        "symptoms":            [],
        "duration":            None,
        "severity":            None,
        "age":                 None,
        "existing_conditions": [],
        "step_count":          0,
        "confidence":          0,
        "differential":        [],
        "specialist_called":   None,
        "tool_calls":          [],
        "needs_escalation":    False,
        "trace":               [],
        "awaiting_user_input": False,
        "triage_complete":     False,
        "questions_asked":     max(0, (current_order // 2) - 1),
        "urgency":             None,
        "safety_approved":     False,
        "advice":              None,
        "symptoms_summary":    None,
    }

    result = await triage_graph.ainvoke(agent_state)

    all_ai = [
        m for m in result["messages"]
        if isinstance(m, AIMessage) and m.content.strip()
    ]
    if not all_ai:
        raise HTTPException(status_code=500, detail="Agent returned no response")

    reply = all_ai[-1].content
    await save_message(db, session_id, MessageRole.ASSISTANT, reply, current_order)

    triage_out = None
    if result.get("triage_complete") and result.get("urgency"):
        urgency_map = {
            "low":       UrgencyLevel.LOW,
            "moderate":  UrgencyLevel.MODERATE,
            "high":      UrgencyLevel.HIGH,
            "emergency": UrgencyLevel.EMERGENCY,
        }
        outcome = TriageOutcome(
            id=uuid4(),
            session_id=session_id,
            urgency=urgency_map.get(result["urgency"], UrgencyLevel.MODERATE),
            confidence=result.get("confidence"),
            advice_text=result.get("advice", reply),
            symptoms_summary=result.get("symptoms_summary"),
            specialist_called=result.get("specialist_called"),
        )
        db.add(outcome)
        await db.flush()

        triage_out = TriageOutcomeOut(
            id=outcome.id,
            urgency=result["urgency"],
            confidence=result.get("confidence"),
            advice_text=outcome.advice_text,
            symptoms_summary=outcome.symptoms_summary,
            specialist_called=result.get("specialist_called"),
            created_at=outcome.created_at,
        )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        triage_outcome=triage_out,
        is_complete=result.get("triage_complete", False),
        agent_trace=result.get("trace", []),
    )


# ── GET /history/{session_id} — 30/minute ────────────────────
@router.get("/history/{session_id}", response_model=HistoryResponse)
@limiter.limit("30/minute")
async def get_history(
    session_id: UUID,
    request: Request,                                # ← renamed from req
    db: AsyncSession = Depends(get_db),
    user_token: Optional[str] = Depends(get_user_token),
):
    session_result = await db.execute(
        select(DBSession).where(DBSession.id == session_id)
    )
    db_session = session_result.scalar_one_or_none()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if user_token and db_session.user_identifier != user_token:
        raise HTTPException(status_code=403, detail="Access denied")

    msgs_result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.order)
    )
    messages = msgs_result.scalars().all()

    outcome_result = await db.execute(
        select(TriageOutcome)
        .where(TriageOutcome.session_id == session_id)
        .order_by(TriageOutcome.created_at.desc())
    )
    outcome = outcome_result.scalars().first()

    return HistoryResponse(
        session_id=session_id,
        messages=[MessageOut.model_validate(m) for m in messages],
        triage_outcome=TriageOutcomeOut.model_validate(outcome) if outcome else None,
    )


# ── GET /sessions — 30/minute ─────────────────────────────────
@router.get("/sessions", response_model=SessionListResponse)
@limiter.limit("30/minute")
async def list_sessions(
    request: Request,                                # ← renamed from req
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user_token: Optional[str] = Depends(get_user_token),
):
    if user_token:
        count_result = await db.execute(
            select(func.count(DBSession.id))
            .where(DBSession.user_identifier == user_token)
        )
        total = count_result.scalar()
        sessions_result = await db.execute(
            select(DBSession)
            .where(DBSession.user_identifier == user_token)
            .order_by(DBSession.created_at.desc())
            .limit(limit).offset(offset)
        )
    else:
        count_result = await db.execute(select(func.count(DBSession.id)))
        total = count_result.scalar()
        sessions_result = await db.execute(
            select(DBSession)
            .order_by(DBSession.created_at.desc())
            .limit(limit).offset(offset)
        )

    sessions = sessions_result.scalars().all()
    out = []
    for s in sessions:
        count_result = await db.execute(
            select(func.count(Message.id)).where(Message.session_id == s.id)
        )
        out.append(SessionOut(
            id=s.id,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=count_result.scalar(),
        ))

    return SessionListResponse(sessions=out, total=total)


# ── POST /token — 10/minute ───────────────────────────────────
@router.post("/token")
@limiter.limit("10/minute")
async def issue_token(request: Request):             # ← renamed from req
    return {"token": str(uuid4())}