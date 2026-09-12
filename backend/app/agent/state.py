from typing import TypedDict, List, Optional, Annotated, Dict, Any
from langchain_core.messages import BaseMessage
import operator


class AgentTrace(TypedDict):
    """Records a single decision made during the planning loop."""
    step:        int
    agent:       str        # orchestrator / cardiac / respiratory / safety / tool
    action:      str        # ask_question / call_specialist / call_tool / conclude
    reasoning:   str        # why this action was taken
    output:      str        # what was produced
    confidence:  Optional[int]


class TriageState(TypedDict):
    # ── Conversation ──────────────────────────────────────────
    messages: Annotated[List[BaseMessage], operator.add]

    # ── Extracted clinical fields ─────────────────────────────
    symptoms:            List[str]
    duration:            Optional[str]
    severity:            Optional[str]
    age:                 Optional[str]
    existing_conditions: List[str]

    # ── Orchestrator planning state ───────────────────────────
    step_count:          int          # how many planning steps taken
    confidence:          int          # 0-100, orchestrator's current confidence
    differential:        List[str]    # possible conditions being considered
    specialist_called:   Optional[str]  # which specialist was invoked
    tool_calls:          List[str]    # tools invoked this session
    needs_escalation:    bool         # safety agent flagged for human handoff

    # ── Agent trace (full reasoning log) ─────────────────────
    trace: Annotated[List[AgentTrace], operator.add]

    # ── Conversation control ──────────────────────────────────
    awaiting_user_input: bool
    triage_complete:     bool
    questions_asked:     int

    # ── Final output ──────────────────────────────────────────
    urgency:             Optional[str]
    safety_approved:     bool         # safety agent signed off
    advice:              Optional[str]
    symptoms_summary:    Optional[str]
