import asyncio
from langchain_core.messages import HumanMessage
from app.agent import triage_graph, TriageState

async def test():
    initial_state: TriageState = {
        "messages":            [HumanMessage(content="I have a bad headache and blurry vision for 2 hours")],
        "symptoms":            [],
        "duration":            None,
        "severity":            None,
        "age":                 None,
        "existing_conditions": [],
        "follow_up_count":     0,
        "awaiting_user_input": False,
        "triage_complete":     False,
        "urgency":             None,
        "advice":              None,
        "symptoms_summary":    None,
    }

    result = await triage_graph.ainvoke(initial_state)

    print("\n=== Agent reply ===")
    for msg in result["messages"]:
        if hasattr(msg, "content"):
            print(f"[{msg.__class__.__name__}]: {msg.content}\n")

    print(f"Triage complete: {result['triage_complete']}")
    print(f"Urgency: {result.get('urgency')}")

asyncio.run(test())