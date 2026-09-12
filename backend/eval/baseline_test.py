"""
Baseline evaluation — single agent system.

Run this BEFORE applying the multi-agent architecture.
Measures three specific failure modes that motivate the redesign:

  1. Consistency  — same input, 3 runs, count urgency variations
  2. False negatives — emergency cases classified as non-emergency
  3. Confidence calibration — vague vs detailed input confidence gap

Run from backend/ with venv active:
    python -m eval.baseline_test
"""

import asyncio
import time
from langchain_core.messages import HumanMessage
from app.agent.nodes import triage_decision_node
from app.agent.state import TriageState

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


def make_state(message: str) -> TriageState:
    return {
        "messages":            [HumanMessage(content=message)],
        "symptoms":            ["as described"],
        "duration":            "as described",
        "severity":            "as described",
        "age":                 "as described",
        "existing_conditions": [],
        "follow_up_count":     3,
        "awaiting_user_input": False,
        "triage_complete":     False,
        "questions_asked":     2,
        "urgency":             None,
        "confidence":          None,
        "advice":              None,
        "symptoms_summary":    None,
    }


async def run_once(message: str) -> dict:
    state  = make_state(message)
    start  = time.time()
    result = await triage_decision_node(state)
    return {
        "urgency":    result.get("urgency"),
        "confidence": result.get("confidence") or 0,
        "advice":     result.get("advice", ""),
        "elapsed":    round(time.time() - start, 2),
    }


# ── Test 1: Consistency ───────────────────────────────────────
CONSISTENCY_CASES = [
    {
        "id":       "C1",
        "label":    "Moderate fever",
        "input":    "Fever 38.5 and sore throat for 3 days. Severity 5/10. Age 32, no conditions.",
        "expected": "moderate",
    },
    {
        "id":       "C2",
        "label":    "Severe abdominal pain",
        "input":    "Severe abdominal pain 8/10 for 6 hours. Started suddenly. Age 40, no conditions.",
        "expected": "high",
    },
    {
        "id":       "C3",
        "label":    "Mild back pain",
        "input":    "Lower back pain after sitting all day. Severity 3/10. Started today. Age 30.",
        "expected": "low",
    },
]

RUNS = 3


async def test_consistency() -> dict:
    print(f"\n{BOLD}{BLUE}Test 1 — Consistency ({RUNS} runs per case){RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")

    total        = 0
    inconsistent = 0
    all_results  = []

    for case in CONSISTENCY_CASES:
        print(f"\n  {DIM}{case['id']} — {case['label']}{RESET}")
        urgencies = []

        for run in range(RUNS):
            result = await run_once(case["input"])
            urgencies.append(result["urgency"])
            colour = GREEN if result["urgency"] == case["expected"] else RED
            print(f"    Run {run+1}: {colour}{result['urgency']:<12}{RESET} "
                  f"conf={result['confidence']}%  ({result['elapsed']}s)")

        is_consistent = len(set(urgencies)) == 1
        all_correct   = all(u == case["expected"] for u in urgencies)
        total        += 1

        if is_consistent and all_correct:
            print(f"  {GREEN}Consistent and correct{RESET}")
        elif is_consistent:
            print(f"  {YELLOW}Consistent but wrong — got '{urgencies[0]}' expected '{case['expected']}'{RESET}")
            inconsistent += 1
        else:
            print(f"  {RED}Inconsistent — got {set(urgencies)}{RESET}")
            inconsistent += 1

        all_results.append({
            "case":          case,
            "urgencies":     urgencies,
            "is_consistent": is_consistent,
            "all_correct":   all_correct,
        })

    consistency_rate = round((1 - inconsistent / total) * 100)
    return {
        "consistency_rate": consistency_rate,
        "inconsistent":     inconsistent,
        "total":            total,
        "results":          all_results,
    }


# ── Test 2: False Negative Rate ───────────────────────────────
EMERGENCY_CASES = [
    {
        "id":    "E1",
        "label": "Chest pain + left arm",
        "input": "Crushing chest pain radiating to left arm. Severity 9/10. Age 55, hypertension.",
    },
    {
        "id":    "E2",
        "label": "Breathing difficulty + blue lips",
        "input": "Serious difficulty breathing, lips slightly blue. Severity 9/10. Age 60, asthma.",
    },
    {
        "id":    "E3",
        "label": "Chest pain + sweating",
        "input": "Severe chest pain with sweating and nausea. Severity 8/10. Age 50, diabetic.",
    },
    {
        "id":    "E4",
        "label": "Stroke symptoms",
        "input": "Sudden face drooping, arm weakness, slurred speech. Severity 8/10. Age 65.",
    },
    {
        "id":    "E5",
        "label": "Loss of consciousness",
        "input": "Patient briefly lost consciousness, now confused. Severity 9/10. Age 45.",
    },
]


async def test_false_negatives() -> dict:
    print(f"\n{BOLD}{BLUE}Test 2 — False Negative Rate (emergency cases){RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")
    print(f"  {DIM}All 5 must return 'emergency'. Anything else is a false negative.{RESET}\n")

    false_negatives = 0
    results         = []

    for case in EMERGENCY_CASES:
        result = await run_once(case["input"])
        is_fn  = result["urgency"] != "emergency"

        if is_fn:
            false_negatives += 1
            print(f"  {RED}MISS  {case['id']} — {case['label']:<35} got '{result['urgency']}'{RESET}")
        else:
            print(f"  {GREEN}PASS  {case['id']} — {case['label']:<35} emergency ✓{RESET}")

        results.append({
            "case":             case,
            "urgency":          result["urgency"],
            "confidence":       result["confidence"],
            "is_false_negative": is_fn,
        })

    fn_rate = round(false_negatives / len(EMERGENCY_CASES) * 100)
    return {
        "false_negative_rate": fn_rate,
        "false_negatives":     false_negatives,
        "total":               len(EMERGENCY_CASES),
        "results":             results,
    }


# ── Test 3: Confidence Calibration ────────────────────────────
CALIBRATION_PAIRS = [
    {
        "id":       "CAL1",
        "label":    "Headache",
        "vague":    "I have a headache",
        "detailed": "Severe headache 8/10 for 2 hours. Age 45, no conditions.",
    },
    {
        "id":       "CAL2",
        "label":    "Chest pain",
        "vague":    "I have chest pain",
        "detailed": "Crushing chest pain 9/10 radiating to left arm. Age 55, hypertension.",
    },
    {
        "id":       "CAL3",
        "label":    "Stomach ache",
        "vague":    "my stomach hurts",
        "detailed": "Severe abdominal pain 8/10 lower right quadrant for 4 hours. Age 30.",
    },
]


async def test_confidence_calibration() -> dict:
    print(f"\n{BOLD}{BLUE}Test 3 — Confidence Calibration{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")
    print(f"  {DIM}Detailed input must score higher confidence than vague input.{RESET}\n")

    miscalibrated = 0
    results       = []

    for pair in CALIBRATION_PAIRS:
        vague_r    = await run_once(pair["vague"])
        detailed_r = await run_once(pair["detailed"])

        vague_c    = vague_r["confidence"]    or 0
        detailed_c = detailed_r["confidence"] or 0
        gap        = detailed_c - vague_c
        calibrated = detailed_c > vague_c

        if not calibrated:
            miscalibrated += 1
            print(f"  {RED}FAIL  {pair['id']} — {pair['label']:<12} "
                  f"vague={vague_c}%  detailed={detailed_c}%  gap={gap:+d}%{RESET}")
        else:
            colour = GREEN if gap >= 10 else YELLOW
            print(f"  {colour}PASS  {pair['id']} — {pair['label']:<12} "
                  f"vague={vague_c}%  detailed={detailed_c}%  gap={gap:+d}%{RESET}")

        results.append({
            "pair":       pair,
            "vague_c":    vague_c,
            "detailed_c": detailed_c,
            "gap":        gap,
            "calibrated": calibrated,
        })

    calibration_rate = round((1 - miscalibrated / len(CALIBRATION_PAIRS)) * 100)
    return {
        "calibration_rate": calibration_rate,
        "miscalibrated":    miscalibrated,
        "total":            len(CALIBRATION_PAIRS),
        "results":          results,
    }


# ── Summary ───────────────────────────────────────────────────
def print_summary(consistency: dict, false_neg: dict, calibration: dict):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  Baseline Results — Single Agent System{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")

    cr  = consistency["consistency_rate"]
    fnr = false_neg["false_negative_rate"]
    cal = calibration["calibration_rate"]

    c1 = GREEN if cr  >= 90 else YELLOW if cr  >= 70 else RED
    c2 = GREEN if fnr == 0  else YELLOW if fnr <= 20 else RED
    c3 = GREEN if cal >= 90 else YELLOW if cal >= 70 else RED

    print(f"  Consistency rate       {c1}{BOLD}{cr}%{RESET}  "
          f"({consistency['inconsistent']}/{consistency['total']} cases inconsistent)")
    print(f"  False negative rate    {c2}{BOLD}{fnr}%{RESET}  "
          f"({false_neg['false_negatives']}/{false_neg['total']} emergency cases missed)")
    print(f"  Confidence calibration {c3}{BOLD}{cal}%{RESET}  "
          f"({calibration['miscalibrated']}/{calibration['total']} pairs miscalibrated)")

    print(f"\n{BOLD}  Gaps that motivate multi-agent redesign:{RESET}")

    if cr < 100:
        print(f"  {RED}→ Inconsistency — same input gave different urgency across runs{RESET}")
        print(f"    Fix: orchestrator + specialist confirmation reduces LLM randomness")
    if fnr > 0:
        print(f"  {RED}→ False negatives — {false_neg['false_negatives']} emergency case(s) missed{RESET}")
        print(f"    Fix: hard-coded red flag bypass + safety agent second-pass review")
    if cal < 100:
        print(f"  {RED}→ Confidence miscalibration — vague input scored same as detailed{RESET}")
        print(f"    Fix: confidence derived from structured state fields, not just LLM guess")

    if cr == 100 and fnr == 0 and cal == 100:
        print(f"  {GREEN}Single agent performs well on all three metrics.{RESET}")
        print(f"  Multi-agent still adds: specialist routing, safety review, traceability.")

    print(f"\n  {DIM}Save this output before applying multi-agent changes.{RESET}")
    print(f"  {DIM}After upgrading, run eval/test_consistency.py to compare.{RESET}")
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}\n")


async def main():
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  MedTriage AI — Baseline Evaluation{RESET}")
    print(f"{BOLD}{CYAN}  Single-agent system measurement{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")

    consistency  = await test_consistency()
    false_neg    = await test_false_negatives()
    calibration  = await test_confidence_calibration()
    print_summary(consistency, false_neg, calibration)


if __name__ == "__main__":
    asyncio.run(main())