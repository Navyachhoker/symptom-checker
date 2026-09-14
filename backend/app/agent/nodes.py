import re
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from app.agent.state import TriageState
from app.config import settings

# ── Shared LLM instance ───────────────────────────────────────
llm = ChatGroq(
    api_key=settings.groq_api_key,
    model="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1024,
)

MAX_FOLLOW_UPS = 3


# ── Node 1: Symptom Intake ────────────────────────────────────
async def symptom_intake_node(state: TriageState) -> dict:
    system_prompt = SystemMessage(content="""
You are a compassionate AI medical triage assistant.
Your job is to understand the user's symptoms clearly.

From the conversation so far, extract what you know and respond warmly.
- Acknowledge what they have told you
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

    symptoms       = state.get("symptoms", [])
    duration       = state.get("duration")
    severity       = state.get("severity")
    age            = state.get("age")
    existing_conds = state.get("existing_conditions", [])

    if "<extract>" in raw and "</extract>" in raw:
        try:
            json_str = re.search(
                r"<extract>(.*?)</extract>", raw, re.DOTALL
            ).group(1)
            data           = json.loads(json_str.strip())
            symptoms       = data.get("symptoms", symptoms)
            duration       = data.get("duration") or duration
            severity       = data.get("severity") or severity
            age            = data.get("age") or age
            existing_conds = data.get("existing_conditions", existing_conds)
        except Exception:
            pass

    visible_reply = re.sub(
        r"<extract>.*?</extract>", "", raw, flags=re.DOTALL
    ).strip()
    visible_reply = re.sub(r"\{[\s\S]*?\}", "", visible_reply).strip()

    return {
        "messages":            [AIMessage(content=visible_reply)],
        "symptoms":            symptoms,
        "duration":            duration,
        "severity":            severity,
        "age":                 age,
        "existing_conditions": existing_conds,
        "awaiting_user_input": True,
        "triage_complete":     False,
    }


# ── Node 2: Follow-up Questions ───────────────────────────────
async def followup_node(state: TriageState) -> dict:
    follow_up_count = state.get("follow_up_count", 0)

    has_enough = (
        len(state.get("symptoms", [])) >= 1
        and state.get("severity") is not None
        and state.get("duration") is not None
    )

    if has_enough or follow_up_count >= MAX_FOLLOW_UPS:
        return {
            "follow_up_count":     follow_up_count,
            "awaiting_user_input": False,
            "triage_complete":     False,
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
    system_prompt = SystemMessage(content=f"""
You are an AI medical triage assistant making a triage assessment.
Be compassionate, warm, and empathetic in your response.
For mental health symptoms, acknowledge the person's feelings before giving advice.

Patient information:

Patient information collected:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'not specified')}
- Severity (1-10): {state.get('severity', 'not specified')}
- Age: {state.get('age', 'not specified')}
- Existing conditions: {state.get('existing_conditions', [])}

Provide your assessment in this EXACT format with these EXACT labels:

URGENCY: <low|moderate|high|emergency>

CONFIDENCE: <integer between 0 and 100>

ADVICE:
<2-4 sentences of clear, compassionate, actionable advice.
Tell the user exactly what to do next.
Remind them this is AI guidance, not a medical diagnosis.>

SUMMARY:
<One sentence summarising the main symptoms and urgency for record-keeping.>

Urgency level definitions — apply strictly:
- low: mild symptoms, clearly safe to manage at home, severity under 4/10
- moderate: symptoms need attention but not urgent, see GP within 24-48 hours
- high: go to urgent care or A&E today — do NOT use emergency unless life-threatening
- emergency: call emergency services immediately — ONLY for life-threatening situations:
  chest pain + arm/jaw, breathing + cyanosis, unconsciousness, seizure, stroke

Do NOT classify as emergency unless symptoms match the emergency list above.
Do NOT classify as low if severity is above 5/10.

SAFETY RULE: The following symptoms MUST result in emergency urgency:
- Chest pain radiating to arm or jaw
- Difficulty breathing with blue lips or cyanosis
- Loss of consciousness
- Suspected stroke symptoms
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw = response.content

    urgency    = "moderate"
    confidence = 70
    advice     = ""
    summary    = ""

    urgency_match    = re.search(r"URGENCY:\s*(\w+)", raw, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", raw, re.IGNORECASE)
    advice_match     = re.search(
        r"ADVICE:\s*(.*?)(?=SUMMARY:|$)", raw, re.DOTALL | re.IGNORECASE
    )
    summary_match    = re.search(r"SUMMARY:\s*(.*?)$", raw, re.DOTALL | re.IGNORECASE)

    if urgency_match:
        urgency = urgency_match.group(1).strip().lower()
    if confidence_match:
        confidence = max(0, min(100, int(confidence_match.group(1).strip())))
    if advice_match:
        advice = advice_match.group(1).strip()
    if summary_match:
        summary = summary_match.group(1).strip()

    urgency_labels = {
        "low":       "Low urgency",
        "moderate":  "Moderate urgency",
        "high":      "High urgency",
        "emergency": "Emergency",
    }
    label         = urgency_labels.get(urgency, "Moderate urgency")
    final_message = f"Triage Assessment — {label}\n\n{advice}"

    return {
        "messages":            [AIMessage(content=final_message)],
        "urgency":             urgency,
        "confidence":          confidence,
        "advice":              advice,
        "symptoms_summary":    summary,
        "triage_complete":     True,
        "awaiting_user_input": False,
    }

