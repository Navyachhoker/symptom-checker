from typing import TypedDict, List

# This is the shared state passed between every LangGraph node
class TriageState(TypedDict):
    session_id: str
    messages: List[dict]      # full conversation so far
    symptoms: str             # symptoms extracted from conversation
    stage: str                # "assess" or "recommend"
    reply: str                # agent's latest reply to user
    urgency: str              # "low" / "medium" / "high" (filled at end)
    advice: str               # final advice (filled at end)
    questions_asked: int      # tracks how many follow-ups asked