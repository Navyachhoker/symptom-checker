from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator


class TriageState(TypedDict):
    # Full conversation history (LangChain message objects)
    messages: Annotated[List[BaseMessage], operator.add]

    # Extracted symptom info
    symptoms: List[str]                  # e.g. ["headache", "blurry vision"]
    duration: Optional[str]              # e.g. "2 days"
    severity: Optional[str]              # e.g. "7/10"
    age: Optional[str]
    existing_conditions: List[str]       # e.g. ["diabetes", "hypertension"]

    # Conversation control
    follow_up_count: int                 # how many follow-ups asked so far
    awaiting_user_input: bool            # pause graph, wait for next message
    triage_complete: bool                # True = ready to give final verdict

    # Final output
    urgency: Optional[str]               # low / moderate / high / emergency
    advice: Optional[str]                # full advice text
    symptoms_summary: Optional[str]      # brief summary for DB storage