from langgraph.graph import StateGraph, END
from agent.state import TriageState
from agent.nodes import assess_node, recommend_node

def build_graph():
    graph = StateGraph(TriageState)

    # Add both nodes
    graph.add_node("assess", assess_node)
    graph.add_node("recommend", recommend_node)

    # Start at assess
    graph.set_entry_point("assess")

    # Conditional edge — where to go after assess_node runs
    def route_after_assess(state: TriageState):
        if state["stage"] == "recommend":
            return "recommend"   # enough info → give recommendation
        return END               # need more info → stop, wait for user reply

    graph.add_conditional_edges("assess", route_after_assess, {
        "recommend": "recommend",
        END: END
    })

    graph.add_edge("recommend", END)

    return graph.compile()

# Single instance used across the app
triage_graph = build_graph()