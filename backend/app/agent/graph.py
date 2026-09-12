from langgraph.graph import StateGraph, END
from app.agent.state import TriageState
from app.agent.orchestrator import orchestrator_node
from app.agent.safety import safety_review


def route_after_orchestrator(state: TriageState) -> str:
    """
    If orchestrator is waiting for user input → END this turn.
    If triage is complete → END.
    Otherwise → run orchestrator again.
    """
    if state.get("awaiting_user_input"):
        return END
    if state.get("triage_complete"):
        return END
    return "orchestrator"


def route_after_safety(state: TriageState) -> str:
    """Safety agent always ends the turn."""
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(TriageState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("safety_agent", safety_review)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "orchestrator": "orchestrator",
            "safety_agent": "safety_agent",
            END:            END,
        }
    )

    graph.add_conditional_edges(
        "safety_agent",
        route_after_safety,
        {END: END}
    )

    return graph.compile()


triage_graph = build_graph()
