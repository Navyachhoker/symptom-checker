"""
Specialist agents invoked conditionally by the orchestrator.
Each specialist has a focused clinical prompt and returns
a structured assessment the orchestrator uses to update case state.

Specialists are NOT called in a fixed pipeline — the orchestrator
decides which one to invoke based on symptoms and confidence level.
"""

import re
from langchain_core.messages import SystemMessage, AIMessage
from langchain_groq import ChatGroq
from app.agent.state import TriageState, AgentTrace
from app.config import settings

# ── Shared LLM ────────────────────────────────────────────────
llm = ChatGroq(
    api_key=settings.groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
)


# ── Shared parser ─────────────────────────────────────────────
def parse_specialist_response(raw: str) -> dict:
    """
    Parses structured output from any specialist agent.
    All specialists use the same output format for consistency.
    """
    urgency    = "moderate"
    confidence = 70
    assessment = raw
    red_flags  = []

    urgency_match    = re.search(r"URGENCY:\s*(\w+)", raw, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", raw, re.IGNORECASE)
    assessment_match = re.search(
        r"ASSESSMENT:\s*(.*?)(?=RED FLAGS:|$)", raw, re.DOTALL | re.IGNORECASE
    )
    red_flags_match  = re.search(r"RED FLAGS:\s*(.*?)$", raw, re.DOTALL | re.IGNORECASE)

    if urgency_match:
        urgency = urgency_match.group(1).strip().lower()
    if confidence_match:
        confidence = max(0, min(100, int(confidence_match.group(1).strip())))
    if assessment_match:
        assessment = assessment_match.group(1).strip()
    if red_flags_match:
        flags_text = red_flags_match.group(1).strip()
        red_flags  = [
            f.strip() for f in flags_text.split("\n")
            if f.strip() and f.strip() != "none"
        ]

    return {
        "urgency":    urgency,
        "confidence": confidence,
        "assessment": assessment,
        "red_flags":  red_flags,
    }


# ── Specialist 1: Cardiac ─────────────────────────────────────
async def cardiac_specialist(state: TriageState) -> dict:
    """
    Evaluates cardiac risk from collected symptoms.
    Invoked when chest pain, palpitations, or arm/jaw pain are present.
    """
    system_prompt = SystemMessage(content=f"""
You are a specialist AI assistant with expertise in cardiac triage.

Patient information:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'not specified')}
- Severity (1-10): {state.get('severity', 'not specified')}
- Age: {state.get('age', 'not specified')}
- Existing conditions: {state.get('existing_conditions', [])}

Assess this case from a cardiac perspective.
Consider: ACS risk, STEMI indicators, arrhythmia, heart failure, pericarditis.

Respond in this EXACT format:

URGENCY: <low|moderate|high|emergency>

CONFIDENCE: <0-100>

ASSESSMENT:
<2-3 sentences on cardiac risk assessment and recommended action.>

RED FLAGS:
<List any cardiac red flags present, one per line. Write 'none' if none.>
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw      = response.content
    parsed   = parse_specialist_response(raw)

    trace: AgentTrace = {
        "step":       state.get("step_count", 0),
        "agent":      "cardiac_specialist",
        "action":     "specialist_assessment",
        "reasoning":  "Cardiac specialist invoked due to chest/cardiac symptoms",
        "output":     parsed["assessment"],
        "confidence": parsed["confidence"],
    }

    return {
        "specialist_called": "cardiac",
        "confidence":        parsed["confidence"],
        "urgency":           parsed["urgency"],
        "trace":             [trace],
    }


# ── Specialist 2: Respiratory ─────────────────────────────────
async def respiratory_specialist(state: TriageState) -> dict:
    """
    Evaluates respiratory risk from collected symptoms.
    Invoked when breathing difficulty, cough, or lung symptoms are present.
    """
    system_prompt = SystemMessage(content=f"""
You are a specialist AI assistant with expertise in respiratory triage.

Patient information:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'not specified')}
- Severity (1-10): {state.get('severity', 'not specified')}
- Age: {state.get('age', 'not specified')}
- Existing conditions: {state.get('existing_conditions', [])}

Assess this case from a respiratory perspective.
Consider: asthma exacerbation, pneumonia, COPD, pulmonary embolism, anaphylaxis.

Respond in this EXACT format:

URGENCY: <low|moderate|high|emergency>

CONFIDENCE: <0-100>

ASSESSMENT:
<2-3 sentences on respiratory risk assessment and recommended action.>

RED FLAGS:
<List any respiratory red flags present, one per line. Write 'none' if none.>
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw      = response.content
    parsed   = parse_specialist_response(raw)

    trace: AgentTrace = {
        "step":       state.get("step_count", 0),
        "agent":      "respiratory_specialist",
        "action":     "specialist_assessment",
        "reasoning":  "Respiratory specialist invoked due to breathing symptoms",
        "output":     parsed["assessment"],
        "confidence": parsed["confidence"],
    }

    return {
        "specialist_called": "respiratory",
        "confidence":        parsed["confidence"],
        "urgency":           parsed["urgency"],
        "trace":             [trace],
    }


# ── Specialist 3: Mental Health ───────────────────────────────
async def mental_health_specialist(state: TriageState) -> dict:
    """
    Evaluates mental health risk from collected symptoms.
    Invoked when anxiety, panic, depression, or self-harm are mentioned.
    """
    system_prompt = SystemMessage(content=f"""
You are a specialist AI assistant with expertise in mental health triage.

Patient information:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'not specified')}
- Severity (1-10): {state.get('severity', 'not specified')}
- Age: {state.get('age', 'not specified')}
- Existing conditions: {state.get('existing_conditions', [])}

Assess this case from a mental health perspective.
Consider: acute anxiety, panic disorder, depression severity, crisis risk, self-harm risk.
Be compassionate and non-judgmental.

Respond in this EXACT format:

URGENCY: <low|moderate|high|emergency>

CONFIDENCE: <0-100>

ASSESSMENT:
<2-3 sentences on mental health risk assessment and recommended action.>

RED FLAGS:
<List any mental health red flags present, one per line. Write 'none' if none.>
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw      = response.content
    parsed   = parse_specialist_response(raw)

    trace: AgentTrace = {
        "step":       state.get("step_count", 0),
        "agent":      "mental_health_specialist",
        "action":     "specialist_assessment",
        "reasoning":  "Mental health specialist invoked due to psychological symptoms",
        "output":     parsed["assessment"],
        "confidence": parsed["confidence"],
    }

    return {
        "specialist_called": "mental_health",
        "confidence":        parsed["confidence"],
        "urgency":           parsed["urgency"],
        "trace":             [trace],
    }


# ── Specialist 4: Pediatric ───────────────────────────────────
async def pediatric_specialist(state: TriageState) -> dict:
    """
    Evaluates pediatric risk from collected symptoms.
    Invoked when patient age is under 12 or child-specific symptoms mentioned.
    """
    system_prompt = SystemMessage(content=f"""
You are a specialist AI assistant with expertise in pediatric triage.

Patient information:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'not specified')}
- Severity (1-10): {state.get('severity', 'not specified')}
- Age: {state.get('age', 'not specified')}
- Existing conditions: {state.get('existing_conditions', [])}

Assess this case from a pediatric perspective.
Consider: febrile seizure risk, dehydration, respiratory distress in children,
meningitis signs, safeguarding concerns.

Respond in this EXACT format:

URGENCY: <low|moderate|high|emergency>

CONFIDENCE: <0-100>

ASSESSMENT:
<2-3 sentences on pediatric risk assessment and recommended action.>

RED FLAGS:
<List any pediatric red flags present, one per line. Write 'none' if none.>
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw      = response.content
    parsed   = parse_specialist_response(raw)

    trace: AgentTrace = {
        "step":       state.get("step_count", 0),
        "agent":      "pediatric_specialist",
        "action":     "specialist_assessment",
        "reasoning":  "Pediatric specialist invoked due to patient age or child symptoms",
        "output":     parsed["assessment"],
        "confidence": parsed["confidence"],
    }

    return {
        "specialist_called": "pediatric",
        "confidence":        parsed["confidence"],
        "urgency":           parsed["urgency"],
        "trace":             [trace],
    }


# ── Specialist 5: General ─────────────────────────────────────
async def general_specialist(state: TriageState) -> dict:
    """
    General triage assessment when no specific specialist is identified.
    Fallback for symptoms that don't match a specialist domain.
    """
    system_prompt = SystemMessage(content=f"""
You are a general medical triage specialist AI assistant.

Patient information:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'not specified')}
- Severity (1-10): {state.get('severity', 'not specified')}
- Age: {state.get('age', 'not specified')}
- Existing conditions: {state.get('existing_conditions', [])}

Provide a general triage assessment covering all relevant systems.

Respond in this EXACT format:

URGENCY: <low|moderate|high|emergency>

CONFIDENCE: <0-100>

ASSESSMENT:
<2-3 sentences on general risk assessment and recommended action.>

RED FLAGS:
<List any red flags present, one per line. Write 'none' if none.>
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw      = response.content
    parsed   = parse_specialist_response(raw)

    trace: AgentTrace = {
        "step":       state.get("step_count", 0),
        "agent":      "general_specialist",
        "action":     "specialist_assessment",
        "reasoning":  "General specialist invoked — no domain-specific match found",
        "output":     parsed["assessment"],
        "confidence": parsed["confidence"],
    }

    return {
        "specialist_called": "general",
        "confidence":        parsed["confidence"],
        "urgency":           parsed["urgency"],
        "trace":             [trace],
    }


# ── Specialist dispatcher ─────────────────────────────────────
SPECIALIST_MAP = {
    "cardiac":      cardiac_specialist,
    "respiratory":  respiratory_specialist,
    "mental":       mental_health_specialist,
    "pediatric":    pediatric_specialist,
    "general":      general_specialist,
}


async def call_specialist(specialist_name: str, state: TriageState) -> dict:
    """
    Dispatches to the correct specialist by name.
    Called by the orchestrator — not hardcoded in the graph.
    """
    fn = SPECIALIST_MAP.get(specialist_name, general_specialist)
    return await fn(state)