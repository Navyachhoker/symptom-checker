from langgraph.graph import StateGraph, END
from app.agent.state import TriageState
from app.agent.nodes import (
    symptom_intake_node,
    followup_node,
    triage_decision_node,
)


def route_after_followup(state: TriageState) -> str:
    if state.get("awaiting_user_input"):
        return END
    return "triage_decision"


def _build(entry: str):
    graph = StateGraph(TriageState)
    graph.add_node("symptom_intake",  symptom_intake_node)
    graph.add_node("followup",        followup_node)
    graph.add_node("triage_decision", triage_decision_node)

    graph.set_entry_point(entry)

    # symptom_intake always stops after one reply
    graph.add_edge("symptom_intake", END)

    graph.add_conditional_edges(
        "followup",
        route_after_followup,
        {"triage_decision": "triage_decision", END: END}
    )
    graph.add_edge("triage_decision", END)

    return graph.compile()


# Two compiled graphs — routes.py picks the right one
intake_graph   = _build("symptom_intake")
followup_graph = _build("followup")