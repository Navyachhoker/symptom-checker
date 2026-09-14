"""
Safety agent — second-pass reviewer.

Every triage decision produced by the orchestrator passes through
this agent before being shown to the user. It can:
  - Approve the decision as-is
  - Upgrade the urgency if it spots something the orchestrator missed
  - Flag the case for human escalation

This is the "catches false negatives" story for interviews:
the safety agent exists because LLMs can be confidently wrong,
and in a medical context a missed emergency is the worst failure mode.
"""

import re
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from app.agent.state import TriageState, AgentTrace
from app.config import settings

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=256,
)

# Urgency levels in order — used to enforce upgrades only, never downgrades
URGENCY_ORDER = ["low", "moderate", "high", "emergency"]


def urgency_index(level) -> int:
    try:
        return URGENCY_ORDER.index(str(level).lower())
    except (ValueError, AttributeError):
        return 1  # default to moderate


def higher_urgency(a: str, b: str) -> str:
    """Returns whichever urgency level is more severe."""
    return a if urgency_index(a) >= urgency_index(b) else b


async def safety_review(state: TriageState) -> dict:
    """
    Reviews the orchestrator's triage decision.

    Key design decisions:
    - Safety agent can ONLY upgrade urgency, never downgrade
    - Hard-coded patterns override LLM review entirely
    - Disagreement between orchestrator and safety agent is logged
      as a trace entry — this is the interview-worthy failure case
    - Returns updated state with safety_approved = True always
      (even if it overrides, the safety agent has approved the
      final output, not the original)
    """
    original_urgency   = state.get("urgency") or "moderate"
    original_advice    = state.get("advice") or ""
    original_confidence = state.get("confidence", 70)
    step               = state.get("step_count", 0)

    # ── Hard-coded override patterns ──────────────────────────
    # These bypass LLM review — certain combinations are always
    # emergency regardless of what the orchestrator decided.
    symptoms    = " ".join(state.get("symptoms", [])).lower()
    messages_text = " ".join([
        m.content for m in state.get("messages", [])
        if hasattr(m, "content")
    ]).lower()
    combined = symptoms + " " + messages_text

    hard_overrides = [
        (
            ["chest pain", "left arm"],
            "Safety override: chest pain + left arm involvement — mandatory emergency"
        ),
        (
            ["chest pain", "jaw"],
            "Safety override: chest pain + jaw pain — possible cardiac event"
        ),
        (
            ["difficulty breathing", "blue"],
            "Safety override: breathing difficulty + cyanosis — mandatory emergency"
        ),
        (
            ["unconscious"],
            "Safety override: loss of consciousness reported"
        ),
        (
            ["seizure"],
            "Safety override: seizure reported — mandatory emergency"
        ),
        (
            ["vomiting blood"],
            "Safety override: haematemesis — mandatory emergency"
        ),
        (
            ["chest pain", "sweat"],
            "Safety override: chest pain + sweating — possible ACS"
        ),
    ]

    for keywords, override_reason in hard_overrides:
        if all(kw in combined for kw in keywords):
            if original_urgency != "emergency":
                # Orchestrator missed this — log it as a caught false negative
                trace: AgentTrace = {
                    "step":       step,
                    "agent":      "safety_agent",
                    "action":     "urgency_override",
                    "reasoning":  override_reason,
                    "output":     f"Upgraded {original_urgency} → emergency (hard override)",
                    "confidence": 99,
                }

                upgraded_advice = (
                    f"{original_advice} "
                    f"IMPORTANT: The safety review system has identified additional "
                    f"risk factors and upgraded this assessment to emergency. "
                    f"Please call emergency services immediately."
                )

                return {
                    "urgency":        "emergency",
                    "confidence":     99,
                    "advice":         upgraded_advice,
                    "safety_approved": True,
                    "trace":          [trace],
                }
            else:
                # Orchestrator already said emergency — safety agrees
                trace: AgentTrace = {
                    "step":       step,
                    "agent":      "safety_agent",
                    "action":     "approve",
                    "reasoning":  "Emergency urgency confirmed by safety pattern match",
                    "output":     "Approved — emergency classification correct",
                    "confidence": 99,
                }
                return {
                    "safety_approved": True,
                    "trace":           [trace],
                }

    # ── LLM-based safety review ───────────────────────────────
    system_prompt = SystemMessage(content=f"""
You are a medical safety reviewer performing a second-pass check
on an AI triage decision before it is shown to a patient.

Original triage decision:
- Urgency: {original_urgency}
- Confidence: {original_confidence}%
- Advice given: {original_advice or 'not available'}

Patient information:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'not specified')}
- Severity (1-10): {state.get('severity', 'not specified')}
- Age: {state.get('age', 'not specified')}
- Existing conditions: {state.get('existing_conditions', [])}
- Specialist consulted: {state.get('specialist_called', 'none')}

Your role:
- You may APPROVE the decision (agree it is correct)
- You may UPGRADE the urgency if you believe it is too low
- You may NEVER downgrade urgency (erring on the side of caution)
- Flag any red flags the original assessment may have missed

Respond in this EXACT format:

DECISION: <APPROVE|UPGRADE>

UPGRADED_URGENCY: <low|moderate|high|emergency>
(only required if DECISION is UPGRADE, otherwise write 'same')

REASONING:
<One or two sentences explaining your decision.>

FLAG:
<Any specific concern not addressed in the original advice, or 'none'.>
""")

    response = await llm.ainvoke([system_prompt])
    raw      = response.content

    decision_match  = re.search(r"DECISION:\s*(\w+)", raw, re.IGNORECASE)
    upgraded_match  = re.search(r"UPGRADED_URGENCY:\s*(\w+)", raw, re.IGNORECASE)
    reasoning_match = re.search(
        r"REASONING:\s*(.*?)(?=FLAG:|$)", raw, re.DOTALL | re.IGNORECASE
    )
    flag_match      = re.search(r"FLAG:\s*(.*?)$", raw, re.DOTALL | re.IGNORECASE)

    decision  = decision_match.group(1).strip().upper()  if decision_match  else "APPROVE"
    upgraded  = upgraded_match.group(1).strip().lower()  if upgraded_match  else original_urgency
    reasoning = reasoning_match.group(1).strip()         if reasoning_match else ""
    flag      = flag_match.group(1).strip()              if flag_match      else "none"

    # ── APPROVE path ──────────────────────────────────────────
    if decision == "APPROVE":
        trace: AgentTrace = {
            "step":       step,
            "agent":      "safety_agent",
            "action":     "approve",
            "reasoning":  reasoning or "Original triage decision approved",
            "output":     f"Approved {original_urgency} urgency classification",
            "confidence": original_confidence,
        }
        return {
            "safety_approved": True,
            "trace":           [trace],
        }

    # ── UPGRADE path ──────────────────────────────────────────
    # Safety agent can only upgrade, never downgrade
    final_urgency = higher_urgency(upgraded, original_urgency)

    if final_urgency != original_urgency:
        # Genuine upgrade — orchestrator missed something
        upgrade_note = ""
        if flag and flag.lower() != "none":
            upgrade_note = f" Note from safety review: {flag}"

        upgraded_advice = (
            f"{original_advice} "
            f"The safety review system has upgraded this assessment "
            f"from {original_urgency} to {final_urgency}.{upgrade_note}"
        )

        trace: AgentTrace = {
            "step":       step,
            "agent":      "safety_agent",
            "action":     "urgency_override",
            "reasoning":  reasoning,
            "output":     f"Upgraded {original_urgency} → {final_urgency}. Flag: {flag}",
            "confidence": original_confidence,
        }

        return {
            "urgency":         final_urgency,
            "advice":          upgraded_advice,
            "safety_approved": True,
            "trace":           [trace],
        }

    else:
        # Safety said UPGRADE but the urgency was already higher —
        # log it but don't change anything
        trace: AgentTrace = {
            "step":       step,
            "agent":      "safety_agent",
            "action":     "approve",
            "reasoning":  f"Upgrade requested to {upgraded} but original {original_urgency} is already higher",
            "output":     f"Approved {original_urgency} — no change needed",
            "confidence": original_confidence,
        }
        return {
            "safety_approved": True,
            "trace":           [trace],
        }