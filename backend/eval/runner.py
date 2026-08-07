"""
Runs a single test case directly through the triage decision node.
In the eval we skip intake and followup — test cases already contain
complete clinical information, so we go straight to triage.
"""

import time
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.nodes import triage_decision_node
from app.agent.state import TriageState


async def run_single(case: dict) -> dict:
    """
    Builds a fully populated state from the test case input
    and invokes the triage decision node directly.
    This avoids the intake/followup pipeline which requires
    multi-turn conversation to extract symptoms.
    """

    # Pre-populate state as if intake + followup already ran
    # The test case inputs contain all clinical info inline
    state: TriageState = {
        "messages": [HumanMessage(content=case["input"])],

        # Extract key fields directly from the test input
        # so triage_decision_node has what it needs
        "symptoms":            [case["description"]],
        "duration":            "as described in message",
        "severity":            "as described in message",
        "age":                 "as described in message",
        "existing_conditions": [],

        "follow_up_count":     3,       # signals followup is done
        "awaiting_user_input": False,
        "triage_complete":     False,
        "urgency":             None,
        "confidence":          None,
        "advice":              None,
        "symptoms_summary":    None,
    }

    start = time.time()

    # Call triage node directly
    result = await triage_decision_node(state)

    elapsed = round(time.time() - start, 2)

    # Extract last non-empty AI reply
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