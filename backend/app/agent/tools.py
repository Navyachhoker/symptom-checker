"""
Clinical tools available to the orchestrator agent.
Each tool is a plain async function — the orchestrator decides
whether to call them based on case state, not a fixed pipeline.
"""

from app.agent.state import TriageState


# ── Tool 1: Clinical Risk Scorer ──────────────────────────────
async def calculate_clinical_score(state: TriageState) -> dict:
    """
    Computes a rule-based clinical risk score from structured state fields.
    This deliberately does NOT use an LLM — calculators should not guess.
    Based on a simplified NEWS2-style scoring system.
    Returns a numeric score and risk band.
    """
    score = 0
    reasons = []

    severity = state.get("severity")
    if severity:
        try:
            sev_num = int(str(severity).split("/")[0].strip())
            if sev_num >= 8:
                score += 3
                reasons.append(f"High severity ({sev_num}/10)")
            elif sev_num >= 5:
                score += 2
                reasons.append(f"Moderate severity ({sev_num}/10)")
            elif sev_num >= 3:
                score += 1
                reasons.append(f"Mild severity ({sev_num}/10)")
        except (ValueError, AttributeError):
            pass

    age = state.get("age")
    if age:
        try:
            age_num = int(str(age).strip())
            if age_num >= 65:
                score += 2
                reasons.append(f"Age {age_num} (elderly, higher risk)")
            elif age_num >= 50:
                score += 1
                reasons.append(f"Age {age_num} (middle-aged, moderate risk)")
        except (ValueError, AttributeError):
            pass

    conditions = state.get("existing_conditions", [])
    high_risk_conditions = {
        "diabetes", "hypertension", "heart disease", "asthma",
        "copd", "cancer", "immunocompromised", "kidney disease"
    }
    for c in conditions:
        if any(h in c.lower() for h in high_risk_conditions):
            score += 2
            reasons.append(f"High-risk condition: {c}")
            break

    symptoms = [s.lower() for s in state.get("symptoms", [])]
    red_flag_symptoms = {
        "chest pain": 3,
        "difficulty breathing": 3,
        "shortness of breath": 3,
        "loss of consciousness": 3,
        "seizure": 3,
        "stroke": 3,
        "severe headache": 2,
        "vomiting blood": 3,
        "high fever": 2,
    }
    for symptom, points in red_flag_symptoms.items():
        if any(symptom in s for s in symptoms):
            score += points
            reasons.append(f"Red flag symptom: {symptom}")

    # Map score to risk band
    if score >= 7:
        band = "emergency"
    elif score >= 5:
        band = "high"
    elif score >= 3:
        band = "moderate"
    else:
        band = "low"

    return {
        "score":   score,
        "band":    band,
        "reasons": reasons,
        "summary": f"Clinical score {score} → {band.upper()} risk. "
                   f"Factors: {', '.join(reasons) if reasons else 'none identified'}",
    }


# ── Tool 2: Symptom Red Flag Checker ─────────────────────────
async def check_red_flags(state: TriageState) -> dict:
    """
    Hard-coded safety net — certain symptom combinations
    bypass LLM reasoning entirely and force emergency urgency.
    LLMs should NOT be the sole safety net in a medical context.
    """
    symptoms = " ".join(state.get("symptoms", [])).lower()
    messages_text = " ".join([
        m.content for m in state.get("messages", [])
        if hasattr(m, "content")
    ]).lower()
    combined = symptoms + " " + messages_text

    emergency_patterns = [
        (["chest pain", "left arm"],       "Chest pain with left arm involvement — possible cardiac event"),
        (["chest pain", "shortness"],      "Chest pain with breathing difficulty — possible cardiac/pulmonary emergency"),
        (["difficulty breathing", "blue"], "Breathing difficulty with cyanosis — call emergency services"),
        (["unconscious"],                  "Loss of consciousness reported"),
        (["seizure"],                      "Seizure reported"),
        (["stroke", "face"],               "Possible stroke symptoms"),
        (["face drooping"],                "Possible stroke — face drooping reported"),
        (["vomiting blood"],               "Haematemesis — emergency"),
        (["severe chest"],                 "Severe chest symptoms reported"),
    ]

    for keywords, reason in emergency_patterns:
        if all(kw in combined for kw in keywords):
            return {
                "emergency": True,
                "reason":    reason,
                "action":    "BYPASS_LLM_ESCALATE_IMMEDIATELY",
            }

    return {"emergency": False, "reason": None, "action": "continue"}


# ── Tool 3: Specialist Router ─────────────────────────────────
def identify_specialist(state: TriageState) -> str:
    """
    Determines which specialist agent the orchestrator should call
    based on current symptoms. Returns specialist name as a string.
    The orchestrator decides WHETHER to call — this just identifies WHO.
    """
    symptoms = " ".join(state.get("symptoms", [])).lower()
    messages_text = " ".join([
        m.content for m in state.get("messages", [])
        if hasattr(m, "content")
    ]).lower()
    combined = symptoms + " " + messages_text

    cardiac_keywords    = ["chest", "heart", "palpitation", "cardiac", "arm pain", "jaw pain"]
    respiratory_keywords = ["breath", "breathing", "lung", "asthma", "copd", "wheez", "cough"]
    mental_keywords     = ["anxiety", "panic", "depression", "mental", "suicid", "self-harm"]
    pediatric_keywords  = ["child", "baby", "infant", "toddler", "years old" ]

    age = state.get("age")
    if age:
        try:
            if int(str(age).strip()) < 12:
                return "pediatric"
        except (ValueError, AttributeError):
            pass

    scores = {
        "cardiac":     sum(1 for k in cardiac_keywords    if k in combined),
        "respiratory": sum(1 for k in respiratory_keywords if k in combined),
        "mental":      sum(1 for k in mental_keywords     if k in combined),
        "pediatric":   sum(1 for k in pediatric_keywords  if k in combined),
    }

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── Tool 4: Escalation Handler ────────────────────────────────
async def escalate_to_human(state: TriageState) -> dict:
    """
    Called when the orchestrator's confidence is below threshold
    after MAX_STEPS or when safety agent flags a concern.
    In production this would page a nurse or schedule a callback.
    For now returns a structured escalation message.
    """
    return {
        "escalated": True,
        "message": (
            "Based on the information provided, this case requires "
            "human clinical review. Please contact a medical professional "
            "directly or call your local healthcare helpline. "
            "If symptoms are worsening, call emergency services immediately."
        ),
        "reason": f"Confidence below threshold after {state.get('step_count', 0)} steps",
    }