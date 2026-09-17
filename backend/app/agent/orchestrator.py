"""
Orchestrator agent — the core of the multi-agent system.

Replaces the fixed intake → followup → triage pipeline with a
planning loop that decides at each step:
  1. Do I need more information? → ask a clarifying question
  2. Do I need a specialist? → call the right specialist agent
  3. Do I need a tool? → run clinical scorer or red flag checker
  4. Do I have enough confidence? → conclude with triage decision
  5. Am I stuck after MAX_STEPS? → escalate to human

This loop is what makes the system genuinely multi-agent rather
than a fixed pipeline with different prompts.
"""

import re
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from app.agent.state import TriageState, AgentTrace
from app.agent.tools import (
    calculate_clinical_score,
    check_red_flags,
    identify_specialist,
    escalate_to_human,
)
from app.agent.specialists import call_specialist
from app.agent.safety import higher_urgency
from app.config import settings

logger = logging.getLogger(__name__)

# ── LLM ───────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=settings.groq_api_key,
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_tokens=1024,
)

MAX_STEPS          = 3     # max planning iterations before escalation
CONFIDENCE_THRESHOLD = 75  # minimum confidence to conclude without escalation

# ── Safety post-processing constants ───────────────────────────
AI_DISCLAIMER = "This is AI-generated guidance, not a medical diagnosis. Always consult a qualified healthcare professional."

DIAGNOSIS_PATTERNS = [
    r"\byou have (a|an)\s+\w+(itis|osis|emia|pathy|attack|infection|disease|stroke)\b",
    r"\byou are having a\b",
]


# ── Step 1: Extract symptoms from user message ────────────────
async def extract_symptoms(state: TriageState) -> dict:
    """
    Parses the latest user message and extracts structured
    clinical fields into state. Runs on every user turn.
    """
    system_prompt = SystemMessage(content="""
You are a clinical data extraction assistant.
Extract structured information from the patient message.

Respond ONLY with a JSON object — no prose, no explanation:
{
  "symptoms": ["list", "of", "symptoms"],
  "duration": "how long or null",
  "severity": "X/10 or null",
  "age": "number or null",
  "existing_conditions": ["list or empty"]
}
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw = response.content

    symptoms       = state.get("symptoms", [])
    duration       = state.get("duration")
    severity       = state.get("severity")
    age            = state.get("age")
    existing_conds = state.get("existing_conditions", [])

    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        data  = json.loads(clean)
        symptoms       = data.get("symptoms", symptoms) or symptoms
        duration       = data.get("duration")  or duration
        severity       = data.get("severity")  or severity
        age            = data.get("age")        or age
        existing_conds = data.get("existing_conditions", existing_conds) or existing_conds
    except Exception as exc:
        # Deliberately not logging `raw` here — it is the model's attempt to
        # structure the patient's actual symptoms/age/conditions, and this
        # path can fire on ordinary malformed output. Length is still useful
        # for spotting truncation vs. garbage output without leaking content.
        logger.warning(
            "extract_symptoms: failed to parse LLM JSON output (%s). Response length: %d chars",
            exc, len(raw or ""),
        )

    return {
        "symptoms":            symptoms,
        "duration":            duration,
        "severity":            severity,
        "age":                 age,
        "existing_conditions": existing_conds,
    }


# ── Step 2: Orchestrator decides next action ──────────────────
async def orchestrator_decide(state: TriageState) -> dict:
    """
    The planning brain. Looks at current case state and decides
    what to do next. Returns an action dict the loop executes.

    Possible actions:
      - ask_question: need more info from the user
      - call_specialist: invoke a domain specialist
      - run_tool: run clinical scorer or red flag checker
      - conclude: enough confidence to give final triage
      - escalate: confidence too low after too many steps
    """
    step      = state.get("step_count", 0)
    confidence = state.get("confidence", 0)

    # Hard exits
    if step >= MAX_STEPS and confidence < CONFIDENCE_THRESHOLD:
        return {"action": "escalate", "reasoning": f"Max steps ({MAX_STEPS}) reached with confidence {confidence}%"}

    if state.get("triage_complete"):
        return {"action": "conclude", "reasoning": "Triage already complete"}

    # Check what we have
    has_symptoms  = len(state.get("symptoms", [])) > 0
    has_severity  = state.get("severity") is not None
    has_duration  = state.get("duration") is not None
    specialist_done = state.get("specialist_called") is not None
    tool_done     = "clinical_score" in state.get("tool_calls", [])
    questions_asked = state.get("questions_asked", 0)
    # Hard rule (not just prompt guidance): cardiac and respiratory cases
    # must get a specialist opinion before concluding, regardless of how
    # confident the generic rule-based clinical scorer alone is. These
    # domains carry a domain-specific differential (e.g. PE, asthma
    # exacerbation, ACS) that a fixed point-based score has no concept
    # of -- a tool-confidence shortcut is acceptable for general/pediatric
    # cases, but not here. Gated on tool_done so this only intervenes at
    # the point the LLM would otherwise be tempted to conclude early; it
    # doesn't force specialist before the tool has even run once.
    if tool_done and not specialist_done and has_symptoms:
        identified_domain = identify_specialist(state)
        if identified_domain in ("cardiac", "respiratory"):
            return {
                "action": "call_specialist",
                "reasoning": f"{identified_domain} domain identified — specialist "
                             f"consultation required before concluding, regardless "
                             f"of tool-only confidence",
                "specialist": identified_domain,
            }


    system_prompt = SystemMessage(content=f"""
You are an orchestrator for a medical triage AI system.
Your job is to decide what to do next given the current case state.

Current case state:
- Symptoms collected: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'unknown')}
- Severity: {state.get('severity', 'unknown')}
- Age: {state.get('age', 'unknown')}
- Existing conditions: {state.get('existing_conditions', [])}
- Current confidence: {confidence}%
- Steps taken: {step}/{MAX_STEPS}
- Specialist already called: {state.get('specialist_called', 'none')}
- Tools already used: {state.get('tool_calls', [])}
- Questions asked so far: {questions_asked}

Rules:
1. If symptoms are missing and questions_asked < 2, ask a question
2. If you have symptoms but no clinical score yet, run the clinical scorer tool
3. If confidence < {CONFIDENCE_THRESHOLD} and no specialist called yet, call a specialist
4. If confidence >= {CONFIDENCE_THRESHOLD} or specialist is done, conclude
5. Never ask more than 2 clarifying questions
6. If stuck after step {MAX_STEPS - 1}, escalate

Respond with ONLY a JSON object:
{{
  "action": "ask_question|call_specialist|run_tool|conclude|escalate",
  "reasoning": "one sentence explaining why",
  "question": "the question to ask (only if action is ask_question)",
  "specialist": "cardiac|respiratory|mental|pediatric|general (only if action is call_specialist)",
  "tool": "clinical_score|red_flag_check (only if action is run_tool)"
}}
""")

    response = await llm.ainvoke([system_prompt])
    raw = response.content

    try:
        clean  = re.sub(r"```json|```", "", raw).strip()
        action = json.loads(clean)
        return action
    except Exception as exc:
        # Fallback — if parse fails, conclude to avoid infinite loop
        logger.error(
            "orchestrator_decide: failed to parse LLM action JSON (%s). Raw response: %r",
            exc, raw,
        )
        return {
            "action":    "conclude",
            "reasoning": "Could not parse orchestrator decision — defaulting to conclude",
        }


# ── Step 3: Execute the chosen action ────────────────────────
async def execute_action(action: dict, state: TriageState) -> dict:
    """
    Executes whatever action the orchestrator decided.
    Returns state updates to merge into the main state.
    """
    action_type = action.get("action", "conclude")
    reasoning   = action.get("reasoning", "")
    step        = state.get("step_count", 0)

    # ── Ask a clarifying question ─────────────────────────────
    if action_type == "ask_question":
        question = action.get("question", "Can you tell me more about your symptoms?")

        trace: AgentTrace = {
            "step":       step,
            "agent":      "orchestrator",
            "action":     "ask_question",
            "reasoning":  reasoning,
            "output":     question,
            "confidence": state.get("confidence", 0),
        }

        return {
            "messages":            [AIMessage(content=question)],
            "awaiting_user_input": True,
            "questions_asked":     state.get("questions_asked", 0) + 1,
            "step_count":          step + 1,
            "trace":               [trace],
        }

    # ── Call a specialist ─────────────────────────────────────
    elif action_type == "call_specialist":
        specialist_name = action.get("specialist") or identify_specialist(state)
        specialist_result = await call_specialist(specialist_name, state)

        trace: AgentTrace = {
            "step":       step,
            "agent":      "orchestrator",
            "action":     "call_specialist",
            "reasoning":  reasoning,
            "output":     f"Called {specialist_name} specialist",
            "confidence": specialist_result.get("confidence", 0),
        }

        # A specialist assesses one domain and can legitimately be less
        # alarmed than an earlier tool result (e.g. a respiratory specialist
        # correctly finding no respiratory red flags on a case a general
        # clinical-score tool flagged as moderate for other reasons). That
        # narrower view should be able to raise the estimate, but must not
        # silently erase a broader/systemic finding already in state --
        # same upgrade-only principle safety.py applies at the final pass,
        # just applied one step earlier so it isn't lost before then.
        specialist_urgency = specialist_result.get("urgency")
        existing_urgency    = state.get("urgency")
        if specialist_urgency and existing_urgency:
            final_urgency = higher_urgency(specialist_urgency, existing_urgency)
        else:
            final_urgency = specialist_urgency or existing_urgency

        return {
            "specialist_called": specialist_name,
            "confidence":        specialist_result.get("confidence", state.get("confidence", 0)),
            "urgency":           final_urgency,
            "step_count":        step + 1,
            "trace":             [trace] + specialist_result.get("trace", []),
            "awaiting_user_input": False,
        }

    # ── Run a tool ────────────────────────────────────────────
    elif action_type == "run_tool":
        tool_name   = action.get("tool", "clinical_score")
        tool_calls  = state.get("tool_calls", [])
        tool_result = {}

        if tool_name == "clinical_score":
            tool_result = await calculate_clinical_score(state)
            tool_calls  = tool_calls + ["clinical_score"]

            # Map clinical score band to confidence boost
            band_confidence = {
                "low":       65,
                "moderate":  70,
                "high":      80,
                "emergency": 90,
            }
            new_confidence = band_confidence.get(
                tool_result.get("band", "moderate"),
                state.get("confidence", 50)
            )

        elif tool_name == "red_flag_check":
            tool_result = await check_red_flags(state)
            tool_calls  = tool_calls + ["red_flag_check"]

            if tool_result.get("emergency"):
                new_confidence = 95
            else:
                new_confidence = state.get("confidence", 50)
        else:
            new_confidence = state.get("confidence", 50)

        trace: AgentTrace = {
            "step":       step,
            "agent":      "orchestrator",
            "action":     "run_tool",
            "reasoning":  reasoning,
            "output":     tool_result.get("summary", str(tool_result)),
            "confidence": new_confidence,
        }

        update = {
            "tool_calls":  tool_calls,
            "confidence":  new_confidence,
            "step_count":  step + 1,
            "trace":       [trace],
            "awaiting_user_input": False,
        }
        # Surface the clinical scorer's band so generate_final_triage's
        # prompt shows an actual estimate instead of "not yet determined" --
        # without this, the tool's finding was computed but never visible
        # to the step that decides final urgency. Still advisory, not a
        # hard floor: the final LLM can and does override it, same as it
        # can override a specialist's assessment.
        if tool_name == "clinical_score" and not state.get("urgency"):
            update["urgency"] = tool_result.get("band")

        return update

    # ── Escalate to human ─────────────────────────────────────
    elif action_type == "escalate":
        escalation = await escalate_to_human(state)

        trace: AgentTrace = {
            "step":       step,
            "agent":      "orchestrator",
            "action":     "escalate",
            "reasoning":  reasoning,
            "output":     escalation["message"],
            "confidence": state.get("confidence", 0),
        }

        return {
            "messages":          [AIMessage(content=escalation["message"])],
            "needs_escalation":  True,
            "triage_complete":   True,
            "awaiting_user_input": False,
            "urgency":           state.get("urgency") or "moderate",
            "advice":            escalation.get("message", "") or "Your case has been flagged for human review.",
            "step_count":        step + 1,
            "trace":             [trace],
        }

    # ── Conclude with triage decision ─────────────────────────
    else:
        return await generate_final_triage(state, reasoning, step)


# ── Step 4: Generate final triage output ─────────────────────
async def generate_final_triage(
    state: TriageState,
    reasoning: str,
    step: int,
) -> dict:
    """
    Generates the final triage decision incorporating all
    specialist assessments and tool results collected so far.
    """
    specialist = state.get("specialist_called", "none")
    tool_calls = state.get("tool_calls", [])

    system_prompt = SystemMessage(content=f"""
You are an AI medical triage assistant producing a final triage decision.

Patient information:
- Symptoms: {state.get('symptoms', [])}
- Duration: {state.get('duration', 'not specified')}
- Severity (1-10): {state.get('severity', 'not specified')}
- Age: {state.get('age', 'not specified')}
- Existing conditions: {state.get('existing_conditions', [])}

Work already done:
- Specialist consulted: {specialist}
- Tools used: {tool_calls}
- Current confidence: {state.get('confidence', 70)}%
- Current urgency estimate: {state.get('urgency', 'not yet determined')}

Produce the final triage assessment. Use all available context.

Respond in this EXACT format:

URGENCY: <low|moderate|high|emergency>

CONFIDENCE: <0-100>

ADVICE:
<3-4 sentences of clear, compassionate, actionable advice.
Reference what was considered (e.g. 'Given your chest symptoms and age').
Tell the user exactly what to do next.
Remind them this is AI guidance, not a medical diagnosis.>

SUMMARY:
<One sentence summarising the main symptoms and urgency for the medical record.>

SAFETY RULE: Emergency symptoms (chest pain + arm, breathing + blue lips,
unconsciousness, seizure) MUST result in emergency urgency.
""")

    response = await llm.ainvoke([system_prompt] + state["messages"])
    raw      = response.content

    urgency    = state.get("urgency") or "moderate"
    confidence = state.get("confidence", 70)
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

    # ── Safety post-processing ─────────────────────────────────
    # Guarantee the AI disclaimer is present regardless of whether
    # the LLM remembered to include it.
    if AI_DISCLAIMER.lower() not in advice.lower():
        advice = f"{advice} {AI_DISCLAIMER}"

    # Flag (don't auto-rewrite) any diagnosis-style phrasing so it
    # can be reviewed — auto-rewriting risks mangling urgent advice.
    diagnosis_flag = None
    for pattern in DIAGNOSIS_PATTERNS:
        if re.search(pattern, advice, re.IGNORECASE):
            diagnosis_flag = f"Diagnosis-language pattern matched: {pattern}"
            break

    urgency_labels = {
        "low":       "Low urgency",
        "moderate":  "Moderate urgency",
        "high":      "High urgency",
        "emergency": "Emergency",
    }
    label         = urgency_labels.get(urgency, "Moderate urgency")
    final_message = f"Triage Assessment — {label}\n\n{advice}"

    trace: AgentTrace = {
        "step":       step,
        "agent":      "orchestrator",
        "action":     "conclude",
        "reasoning":  reasoning,
        "output":     f"Final triage: {urgency} ({confidence}% confidence)",
        "confidence": confidence,
    }

    traces = [trace]
    if diagnosis_flag:
        traces.append({
            "step":       step,
            "agent":      "orchestrator",
            "action":     "diagnosis_language_flag",
            "reasoning":  diagnosis_flag,
            "output":     advice,
            "confidence": confidence,
        })

    return {
        "messages":            [AIMessage(content=final_message)],
        "urgency":             urgency,
        "confidence":          confidence,
        "advice":              advice,
        "symptoms_summary":    summary,
        "triage_complete":     True,
        "awaiting_user_input": False,
        "step_count":          step + 1,
        "trace":               traces,
    }


# ── Main orchestrator node (called by graph) ──────────────────
async def orchestrator_node(state: TriageState) -> dict:
    """
    The single entry point called by LangGraph on each turn.

    Flow:
    1. Extract symptoms from latest message
    2. Check for hard-coded red flags (bypass LLM if found)
    3. Run the planning loop until action requires user input or completion
    4. Return updated state
    """
    # Step 1 — extract symptoms
    extraction = await extract_symptoms(state)
    state = {**state, **extraction}

    # Step 2 — red flag bypass (hard safety net, no LLM involved)
    red_flag_result = await check_red_flags(state)
    if red_flag_result.get("emergency"):
        bypass_message = (
            "Based on the symptoms you have described, this is a medical emergency. "
            "Please call emergency services (999 / 112 / 911) immediately. "
            "Do not wait. This is not a situation where AI triage is appropriate — "
            "you need immediate human medical intervention. "
            f"{AI_DISCLAIMER}"
        )
        trace: AgentTrace = {
            "step":       0,
            "agent":      "red_flag_checker",
            "action":     "emergency_bypass",
            "reasoning":  red_flag_result.get("reason", "Red flag pattern detected"),
            "output":     bypass_message,
            "confidence": 99,
        }
        return {
            "messages":            [AIMessage(content=bypass_message)],
            "urgency":             "emergency",
            "confidence":          99,
            "advice":              bypass_message,
            "symptoms_summary":    red_flag_result.get("reason", "Emergency red flag detected"),
            "triage_complete":     True,
            "awaiting_user_input": False,
            "tool_calls":          state.get("tool_calls", []) + ["red_flag_check"],
            "trace":               [trace],
            "needs_escalation":    False,
            "safety_approved":     True,
            "safety_review_ran":   False,
        }

    # Step 3 — planning loop
    max_inner_steps = 3
    inner_step      = 0

    while inner_step < max_inner_steps:
        action = await orchestrator_decide(state)
        result = await execute_action(action, state)

        # Merge trace explicitly BEFORE spreading `result` over `state`.
        # `{**state, **result}` overwrites state["trace"] with just this
        # iteration's entries — grabbing "existing" afterward would only
        # be re-reading that same overwritten value, which is a no-op.
        # We build the combined list first, then apply it after the merge.
        combined_trace = state.get("trace", []) + result.get("trace", [])

        state = {**state, **result}
        state["trace"] = combined_trace

        action_type = action.get("action", "conclude")

        # Stop inner loop if we need user input or are done
        if action_type in ("ask_question", "conclude", "escalate"):
            break

        inner_step += 1

    return state