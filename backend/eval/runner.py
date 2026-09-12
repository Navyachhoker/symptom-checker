import time
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.nodes import triage_decision_node
from app.agent.state import TriageState


async def run_single(case: dict) -> dict:
    state: TriageState = {
        "messages":            [HumanMessage(content=case["input"])],
        "symptoms":            [case["description"]],
        "duration":            "as described in message",
        "severity":            "as described in message",
        "age":                 "as described in message",
        "existing_conditions": [],
        "step_count":          3,
        "confidence":          0,
        "differential":        [],
        "specialist_called":   None,
        "tool_calls":          [],
        "needs_escalation":    False,
        "trace":               [],
        "awaiting_user_input": False,
        "triage_complete":     False,
        "questions_asked":     2,
        "urgency":             None,
        "safety_approved":     False,
        "advice":              None,
        "symptoms_summary":    None,
    }

    start  = time.time()
    result = await triage_decision_node(state)
    elapsed = round(time.time() - start, 2)

    ai_messages = [
        m for m in result.get("messages", [])
        if isinstance(m, AIMessage) and m.content.strip()
    ]
    reply = ai_messages[-1].content if ai_messages else ""

    return {
        "id":               case["id"],
        "description":      case["description"],
        "expected_urgency": case["expected_urgency"],
        "actual_urgency":   result.get("urgency"),
        "confidence":       result.get("confidence"),
        "advice":           result.get("advice", ""),
        "symptoms_summary": result.get("symptoms_summary", ""),
        "reply":            reply,
        "is_complete":      result.get("triage_complete", False),
        "elapsed_seconds":  elapsed,
    }