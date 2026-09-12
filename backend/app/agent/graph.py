from langgraph.graph import StateGraph, END
from app.agent.state import TriageState
from app.agent.orchestrator import orchestrator_node


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


def build_graph() -> StateGraph:
    graph = StateGraph(TriageState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"orchestrator": "orchestrator", END: END}
    )

    return graph.compile()


triage_graph = build_graph()