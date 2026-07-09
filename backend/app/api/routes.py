from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from langchain_core.messages import HumanMessage, AIMessage
from uuid import UUID, uuid4

from app.db.database import get_db
from app.db.models import Session as DBSession, Message, TriageOutcome, MessageRole, UrgencyLevel
from app.models.schemas import (
    ChatRequest, ChatResponse,
    HistoryResponse, MessageOut,
    TriageOutcomeOut, SessionOut, SessionListResponse,
)
from app.agent import intake_graph, followup_graph, TriageState

router = APIRouter()


# ── Helper: load conversation from DB ────────────────────────
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


# ── Helper: save a message ────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# POST /chat
# ─────────────────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    # 1. Resolve or create session
    if request.session_id:
        session_result = await db.execute(
            select(DBSession).where(DBSession.id == request.session_id)
        )
        db_session = session_result.scalar_one_or_none()
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        db_session = DBSession(id=uuid4(), user_identifier=req.client.host)
        db.add(db_session)
        await db.flush()

    session_id = db_session.id

    # 2. Load history
    existing_msgs, lc_history = await load_conversation(session_id, db)
    current_order = len(existing_msgs)

    # 3. Save user message
    await save_message(db, session_id, MessageRole.USER, request.message, current_order)
    current_order += 1

    # 4. Build agent state
    from langchain_core.messages import HumanMessage as HM
    agent_state: TriageState = {
        "messages":            lc_history + [HM(content=request.message)],
        "symptoms":            [],
        "duration":            None,
        "severity":            None,
        "age":                 None,
        "existing_conditions": [],
        "follow_up_count":     max(0, (current_order // 2) - 1),
        "awaiting_user_input": False,
        "triage_complete":     False,
        "urgency":             None,
        "advice":              None,
        "symptoms_summary":    None,
    }

    # 5. Pick graph — first message uses intake, rest use followup
    is_first = current_order <= 1
    graph = intake_graph if is_first else followup_graph
    result = await graph.ainvoke(agent_state)

    # 6. Extract last non-empty AI reply
    from langchain_core.messages import AIMessage as AM
    all_ai = [m for m in result["messages"] if isinstance(m, AM) and m.content.strip()]
    if not all_ai:
        raise HTTPException(status_code=500, detail="Agent returned no response")
    reply = all_ai[-1].content

    # 7. Save assistant message
    await save_message(db, session_id, MessageRole.ASSISTANT, reply, current_order)

    # 8. Save triage outcome if complete
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
            advice_text=result.get("advice", reply),
            symptoms_summary=result.get("symptoms_summary"),
        )
        db.add(outcome)
        await db.flush()
        triage_out = TriageOutcomeOut(
            id=outcome.id,
            urgency=result["urgency"],
            advice_text=outcome.advice_text,
            symptoms_summary=outcome.symptoms_summary,
            created_at=outcome.created_at,
        )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        triage_outcome=triage_out,
        is_complete=result.get("triage_complete", False),
    )


# ─────────────────────────────────────────────────────────────
# GET /history/{session_id}
# ─────────────────────────────────────────────────────────────
@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    session_result = await db.execute(
        select(DBSession).where(DBSession.id == session_id)
    )
    if not session_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Session not found")

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


# ─────────────────────────────────────────────────────────────
# GET /sessions
# ─────────────────────────────────────────────────────────────
@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    total_result = await db.execute(select(func.count(DBSession.id)))
    total = total_result.scalar()

    sessions_result = await db.execute(
        select(DBSession)
        .order_by(DBSession.created_at.desc())
        .limit(limit)
        .offset(offset)
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