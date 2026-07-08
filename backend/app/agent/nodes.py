from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from app.agent.state import TriageState
from app.config import settings

# ── Shared LLM instance ─────────────────────────────────────
llm = ChatGroq(
    api_key=settings.groq_api_key,
    model="llama-3.3-70b-versatile",   # fast + smart, free on Groq
    temperature=0.3,
    max_tokens=1024,
)

MAX_FOLLOW_UPS = 3   # ask at most 3 clarifying questions before triaging


# ── Node 1: Symptom Intake ───────────────────────────────────
async def symptom_intake_node(state: TriageState) -> dict:
    """
    First node. Greets the user, acknowledges their symptoms,
    and extracts key facts into state fields.
    """
    system_prompt = SystemMessage(content="""
You are a compassionate AI medical triage assistant.
Your job is to understand the user's symptoms clearly.

From the conversation so far, extract what you know and respond warmly.
- Acknowledge what they've told you
- Ask ONE clarifying question if critical info is missing
  (e.g. duration, severity on 1-10, age, relevant medical history)
- Do NOT diagnose. Do NOT prescribe. You are a triage helper only.
- Keep responses concise and empathetic (2-4 sentences max)

After your response, on a NEW LINE output a JSON block like:
<extract>
{
  "symptoms": ["symptom1", "symptom2"],
  "duration": "2 days or null",
  "severity": "7/10 or null",
  "age": "35 or null",
  "existing_conditions": ["diabetes"] 
}
</extract>
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw = response.content

    # Parse extraction block
    symptoms         = state.get("symptoms", [])
    duration         = state.get("duration")
    severity         = state.get("severity")
    age              = state.get("age")
    existing_conds   = state.get("existing_conditions", [])

    if "<extract>" in raw and "</extract>" in raw:
        import json, re
        try:
            json_str = re.search(r"<extract>(.*?)</extract>", raw, re.DOTALL).group(1)
            data = json.loads(json_str.strip())
            symptoms       = data.get("symptoms", symptoms)
            duration       = data.get("duration") or duration
            severity       = data.get("severity") or severity
            age            = data.get("age") or age
            existing_conds = data.get("existing_conditions", existing_conds)
        except Exception:
            pass  # extraction failed, carry forward existing state

    # Strip the <extract> block from the visible reply
    visible_reply = raw.split("<extract>")[0].strip()

    return {
        "messages":           [AIMessage(content=visible_reply)],
        "symptoms":           symptoms,
        "duration":           duration,
        "severity":           severity,
        "age":                age,
        "existing_conditions": existing_conds,
        "awaiting_user_input": True,
        "triage_complete":    False,
    }


# ── Node 2: Follow-up Questions ──────────────────────────────
async def followup_node(state: TriageState) -> dict:
    """
    Asks targeted clarifying questions — runs up to MAX_FOLLOW_UPS times.
    Once we have enough info (or hit the limit), signals to move to triage.
    """
    follow_up_count = state.get("follow_up_count", 0)

    # Decide if we have enough info already
    has_enough = (
        len(state.get("symptoms", [])) >= 1
        and state.get("severity") is not None
        and state.get("duration") is not None
    )

    if has_enough or follow_up_count >= MAX_FOLLOW_UPS:
        # Skip asking, go straight to triage
        return {
            "follow_up_count":   follow_up_count,
            "awaiting_user_input": False,
            "triage_complete":   False,
        }

    system_prompt = SystemMessage(content=f"""
You are a medical triage assistant. You have collected these details so far:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'unknown')}
- Severity (1-10): {state.get('severity', 'unknown')}
- Age: {state.get('age', 'unknown')}
- Existing conditions: {state.get('existing_conditions', [])}

Ask ONE focused follow-up question to fill in the most critical missing gap.
Priority order: severity → duration → age → existing conditions → other symptoms.
Be brief and kind. Do not repeat questions already answered.
Do NOT diagnose or prescribe.
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])

    return {
        "messages":            [AIMessage(content=response.content)],
        "follow_up_count":     follow_up_count + 1,
        "awaiting_user_input": True,
        "triage_complete":     False,
    }


# ── Node 3: Triage Decision ───────────────────────────────────
async def triage_decision_node(state: TriageState) -> dict:
    """
    Final node. Reviews all collected info and produces:
    - urgency level (low / moderate / high / emergency)
    - clear, actionable advice
    - a brief symptoms summary for the DB
    """
    system_prompt = SystemMessage(content=f"""
You are an AI medical triage assistant making a triage assessment.

Patient information collected:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'not specified')}
- Severity (1-10): {state.get('severity', 'not specified')}
- Age: {state.get('age', 'not specified')}
- Existing conditions: {state.get('existing_conditions', [])}

Based on this, provide a triage assessment in this EXACT format:

URGENCY: <low|moderate|high|emergency>

ADVICE:
<2-4 sentences of clear, compassionate, actionable advice.
Tell the user what to do next: rest at home / see a GP / go to urgent care / call 999 now.
Remind them this is AI guidance, not a diagnosis.>

SUMMARY:
<One sentence summarising the main symptoms and urgency for record-keeping.>

Urgency definitions:
- low: minor symptoms, self-care at home
- moderate: see a GP within 24-48 hours
- high: go to urgent care / A&E today
- emergency: call emergency services immediately (999/112/911)
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw = response.content

    # Parse structured output
    import re
    urgency = "moderate"
    advice  = raw
    summary = ""

    urgency_match = re.search(r"URGENCY:\s*(\w+)", raw, re.IGNORECASE)
    advice_match  = re.search(r"ADVICE:\s*(.*?)(?=SUMMARY:|$)", raw, re.DOTALL | re.IGNORECASE)
    summary_match = re.search(r"SUMMARY:\s*(.*?)$", raw, re.DOTALL | re.IGNORECASE)

    if urgency_match:
        urgency = urgency_match.group(1).strip().lower()
    if advice_match:
        advice  = advice_match.group(1).strip()
    if summary_match:
        summary = summary_match.group(1).strip()

    # Friendly final message shown to user
    urgency_labels = {
        "low":       "🟢 Low urgency",
        "moderate":  "🟡 Moderate urgency",
        "high":      "🔴 High urgency",
        "emergency": "🚨 Emergency",
    }
    label = urgency_labels.get(urgency, "🟡 Moderate urgency")
    final_message = f"**Triage Assessment — {label}**\n\n{advice}"

    return {
        "messages":         [AIMessage(content=final_message)],
        "urgency":          urgency,
        "advice":           advice,
        "symptoms_summary": summary,
        "triage_complete":  True,
        "awaiting_user_input": False,
    }