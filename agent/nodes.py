from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import TriageState

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Node 1: assess symptoms, decide if we need more info or can recommend
def assess_node(state: TriageState) -> TriageState:
    questions_asked = state.get("questions_asked", 0)

    system_prompt = f"""You are a medical triage assistant collecting information.

    You have asked {questions_asked} clarifying question(s) so far.
    You are allowed to ask a maximum of 3 clarifying questions total.

    What you need to collect before making a recommendation:
    1. How long the symptoms have been present (duration)
    2. Severity on a scale of 1-10
    3. Any additional symptoms like nausea, dizziness, chest pain

    Rules:
    - Check the conversation history to see what has already been answered
    - If something from the list above is still unknown AND questions_asked < 3, ask about it
    - Ask ONE question at a time, most important first
    - If you have collected all 3 pieces of info OR questions_asked >= 3, proceed to recommendation
    - Start with FOLLOWUP: if asking a question
    - Start with READY: if you have enough info

    Example:
    FOLLOWUP: How long have you been experiencing these symptoms?
    READY: I have enough information to assess your condition."""

    messages = [SystemMessage(content=system_prompt)]
    for m in state["messages"]:
        messages.append(
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else SystemMessage(content=m["content"])
        )

    response = llm.invoke(messages).content

    if response.startswith("FOLLOWUP:"):
        return {
            **state,
            "reply": response.replace("FOLLOWUP:", "").strip(),
            "stage": "assess",
            "questions_asked": questions_asked + 1   # increment counter
        }
    else:
        return {
            **state,
            "reply": response.replace("READY:", "").strip(),
            "stage": "recommend",
            "questions_asked": questions_asked
        }


# Node 2: give final recommendation
def recommend_node(state: TriageState) -> TriageState:
    system_prompt = """You are a medical triage assistant.
    Based on the symptoms discussed, provide:
    1. Urgency level: LOW, MEDIUM, or HIGH
    2. Clear advice in 2-3 sentences.
    
    Format your response exactly like this:
    URGENCY: <LOW/MEDIUM/HIGH>
    ADVICE: <your advice here>"""

    messages = [SystemMessage(content=system_prompt)]
    for m in state["messages"]:
        messages.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                       else SystemMessage(content=m["content"]))

    response = llm.invoke(messages).content

    # Parse urgency and advice from response
    lines = response.strip().split("\n")
    urgency = "medium"
    advice = response

    for line in lines:
        if line.startswith("URGENCY:"):
            urgency = line.replace("URGENCY:", "").strip().lower()
        if line.startswith("ADVICE:"):
            advice = line.replace("ADVICE:", "").strip()

    return {**state, "urgency": urgency, "advice": advice,
            "reply": f"Urgency: {urgency.upper()}\n{advice}", "stage": "done"}
