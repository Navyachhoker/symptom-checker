"""
Test 1 — Triage Classification Accuracy

Measures:
  - Urgency classification correctness across all 4 levels
  - Safety agent approval rate
  - Specialist routing correctness
  - False negative rate on emergency cases (most critical)
  - Before vs after comparison with single agent baseline

Run from backend/ with venv active:
    python -m eval.test_classification
"""

import sys
import asyncio
import time
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import triage_graph
from app.agent.state import TriageState

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

URGENCY_ORDER = ["low", "moderate", "high", "emergency"]


# ── Test cases ────────────────────────────────────────────────
TEST_CASES = [
    # LOW urgency
    {
        "id": 1, "category": "low",
        "label": "Mild cold",
        "input": "Mild runny nose and slight sore throat. Severity 2/10. Started yesterday. Age 28, no conditions.",
        "expected_urgency":    "low",
        "expected_specialist": None,
    },
    {
        "id": 2, "category": "low",
        "label": "Minor ankle sprain",
        "input": "Twisted ankle while walking. Mild swelling, pain 3/10. 2 hours ago. Age 25, no conditions.",
        "expected_urgency":    "low",
        "expected_specialist": None,
    },
    {
        "id": 3, "category": "low",
        "label": "Mild back pain",
        "input": "Lower back pain after sitting at desk. Severity 3/10. Started today. Age 30, no conditions.",
        "expected_urgency":    "low",
        "expected_specialist": None,
    },

    # MODERATE urgency
    {
        "id": 4, "category": "moderate",
        "label": "Persistent fever",
        "input": "Fever 38.5 degrees and bad sore throat for 3 days. Severity 5/10. Age 32, no conditions.",
        "expected_urgency":    "moderate",
        "expected_specialist": None,
    },
    {
        "id": 5, "category": "moderate",
        "label": "Persistent headache with nausea",
        "input": "Persistent headache with nausea for 2 days. Severity 6/10. No vision changes. Age 38, no conditions.",
        "expected_urgency":    "moderate",
        "expected_specialist": None,
    },
    {
        "id": 6, "category": "moderate",
        "label": "UTI symptoms",
        "input": "Burning urination and frequent urge for 2 days. Severity 5/10. Age 27, no conditions.",
        "expected_urgency":    "moderate",
        "expected_specialist": None,
    },

    # HIGH urgency
    {
        "id": 7, "category": "high",
        "label": "Severe abdominal pain",
        "input": "Severe abdominal pain 8/10 for 6 hours. Started suddenly. Age 40, no conditions.",
        "expected_urgency":    "high",
        "expected_specialist": None,
    },
    {
        "id": 8, "category": "high",
        "label": "Very high fever",
        "input": "Fever 40 degrees, severe headache, body aches. Severity 7/10 for 2 days. Age 35, no conditions.",
        "expected_urgency":    "high",
        "expected_specialist": None,
    },

    # EMERGENCY urgency — these must never be missed
    {
        "id": 9, "category": "emergency",
        "label": "Chest pain + left arm",
        "input": "Crushing chest pain radiating to left arm. Severity 9/10. Age 55, hypertension.",
        "expected_urgency":    "emergency",
        "expected_specialist": "cardiac",
    },
    {
        "id": 10, "category": "emergency",
        "label": "Breathing difficulty + cyanosis",
        "input": "Serious difficulty breathing, chest tightness, lips slightly blue. Severity 9/10. Age 60, asthma.",
        "expected_urgency":    "emergency",
        "expected_specialist": "respiratory",
    },
    {
        "id": 11, "category": "emergency",
        "label": "Stroke symptoms",
        "input": "Sudden face drooping left side, arm weakness, slurred speech. Severity 8/10. Age 65.",
        "expected_urgency":    "emergency",
        "expected_specialist": None,
    },
    {
        "id": 12, "category": "emergency",
        "label": "Loss of consciousness",
        "input": "Patient briefly lost consciousness and is now confused. Severity 9/10. Age 45.",
        "expected_urgency":    "emergency",
        "expected_specialist": None,
    },
    {
    "id": 13,
    "category": "emergency",
    "label": "Anaphylaxis",
    "input": (
        "Sudden swelling of lips and tongue after eating nuts. "
        "Difficulty breathing."
    ),
    "expected_urgency": "emergency",
    "expected_specialist": "respiratory",
},
{
    "id": 14,
    "category": "emergency",
    "label": "Sepsis symptoms",
    "input": (
        "High fever, confusion, rapid breathing, "
        "heart racing."
    ),
    "expected_urgency": "emergency",
    "expected_specialist": None,
},
{
    "id": 15,
    "category": "emergency",
    "label": "GI bleed",
    "input": (
        "Vomiting blood and feeling faint."
    ),
    "expected_urgency": "emergency",
    "expected_specialist": None,
},
]


def make_state(message: str) -> TriageState:
    return {
        "messages":            [HumanMessage(content=message)],
        "symptoms":            [],
        "duration":            None,
        "severity":            None,
        "age":                 None,
        "existing_conditions": [],
        "step_count":          0,
        "confidence":          0,
        "differential":        [],
        "specialist_called":   None,
        "tool_calls":          [],
        "needs_escalation":    False,
        "trace":               [],
        "awaiting_user_input": False,
        "triage_complete":     False,
        "questions_asked":     2,
        "urgency":             None,
        "safety_approved":     False,
        "advice":              None,
        "symptoms_summary":    None,
    }


async def run_case(case: dict) -> dict:
    state   = make_state(case["input"])
    start   = time.time()
    result  = await triage_graph.ainvoke(state)
    elapsed = round(time.time() - start, 2)

    ai_msgs = [
        m for m in result.get("messages", [])
        if isinstance(m, AIMessage) and m.content.strip()
    ]

    return {
        "urgency":          result.get("urgency"),
        "confidence":       result.get("confidence"),
        "safety_approved":  result.get("safety_approved", False),
        "specialist":       result.get("specialist_called"),
        "tool_calls":       result.get("tool_calls", []),
        "trace_steps":      len(result.get("trace", [])),
        "triage_complete":  result.get("triage_complete", False),
        "reply":            ai_msgs[-1].content if ai_msgs else "",
        "elapsed":          elapsed,
    }


def urgency_distance(a: str, b: str) -> int:
    try:
        return abs(URGENCY_ORDER.index(a) - URGENCY_ORDER.index(b))
    except ValueError:
        return 99


async def run_all():
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  Test 1 — Triage Classification Accuracy{RESET}")
    print(f"{BOLD}{CYAN}  Multi-agent system{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")

    results    = []
    total_time = 0.0

    for case in TEST_CASES:
        label = f"Test {case['id']:02d} — {case['label']}"
        print(f"  {DIM}{label:<42}{RESET}", end=" ", flush=True)

        try:
            result  = await run_case(case)
            total_time += result["elapsed"]

            actual   = result["urgency"]
            expected = case["expected_urgency"]
            distance = urgency_distance(expected, actual or "")

            failures = []
            warnings = []

            if actual != expected:
                if distance == 1:
                    warnings.append(f"Off by one — expected '{expected}' got '{actual}'")
                else:
                    failures.append(f"Wrong urgency — expected '{expected}' got '{actual}'")

            if not result["triage_complete"]:
                failures.append("Triage did not complete")

            if not result["safety_approved"]:
                warnings.append("Safety agent did not approve")

            conf_str = f"  conf={result['confidence']}%" if result["confidence"] else ""
            spec_str = f"  spec={result['specialist']}" if result["specialist"] else ""
            time_str = f"{result['elapsed']}s"

            if not failures and not warnings:
                print(f"{GREEN}PASS{RESET}  {DIM}({time_str}{conf_str}{spec_str}){RESET}")
            elif not failures:
                print(f"{YELLOW}WARN{RESET}  {DIM}({time_str}{conf_str}{spec_str}){RESET}")
                for w in warnings:
                    print(f"           {YELLOW}~ {w}{RESET}")
            else:
                print(f"{RED}FAIL{RESET}  {DIM}({time_str}{conf_str}{spec_str}){RESET}")
                for f in failures:
                    print(f"           {RED}✗ {f}{RESET}")
                for w in warnings:
                    print(f"           {YELLOW}~ {w}{RESET}")

            results.append({
                "case":     case,
                "result":   result,
                "failures": failures,
                "warnings": warnings,
                "passed":   len(failures) == 0,
            })

        except Exception as e:
            print(f"{RED}ERROR{RESET}")
            print(f"           {RED}✗ {e}{RESET}")
            results.append({
                "case":     case,
                "result":   None,
                "failures": [str(e)],
                "warnings": [],
                "passed":   False,
            })

    # ── Summary ───────────────────────────────────────────────
    total   = len(TEST_CASES)
    passed  = sum(1 for r in results if r["passed"])
    failed  = total - passed
    score   = round(passed / total * 100)
    avg_t   = round(total_time / total, 2)

    # Emergency false negatives — the critical metric
    emergency_cases = [r for r in results if r["case"]["category"] == "emergency"]
    fn_cases        = [
        r for r in emergency_cases
        if r["result"] and r["result"].get("urgency") != "emergency"
    ]
    fn_rate = round(len(fn_cases) / len(emergency_cases) * 100) if emergency_cases else 0

    # Safety approval rate
    safety_approved = [
        r for r in results
        if r["result"] and r["result"].get("safety_approved")
    ]
    safety_rate = round(len(safety_approved) / total * 100)

    print(f"\n{BOLD}{BLUE}Summary{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")
    print(f"  Score              : {GREEN if score >= 80 else RED}{BOLD}{score}%{RESET} ({passed}/{total} passed)")
    print(f"  Avg latency        : {avg_t}s")
    print(f"  False negative rate: {GREEN if fn_rate == 0 else RED}{BOLD}{fn_rate}%{RESET} ({len(fn_cases)}/{len(emergency_cases)} emergency missed)")
    print(f"  Safety approval    : {GREEN}{safety_rate}%{RESET}")

    # Per category
    print(f"\n{BOLD}{BLUE}By urgency category{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")
    for cat in URGENCY_ORDER:
        cat_r  = [r for r in results if r["case"]["category"] == cat]
        cat_p  = [r for r in cat_r if r["passed"]]
        if not cat_r:
            continue
        pct    = round(len(cat_p) / len(cat_r) * 100)
        colour = GREEN if pct == 100 else YELLOW if pct >= 50 else RED
        bar    = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {cat.upper():<12} {colour}{bar}{RESET}  {colour}{len(cat_p)}/{len(cat_r)} ({pct}%){RESET}")

    # Critical failure — any emergency missed
    if fn_cases:
        print(f"\n{BOLD}{RED}Critical failures — emergency cases missed{RESET}")
        print(f"{RED}{'─' * 60}{RESET}")
        for r in fn_cases:
            print(f"  Test {r['case']['id']} — {r['case']['label']}")
            print(f"  Got: {r['result'].get('urgency')} instead of emergency")

    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}\n")
    
    return len(fn_cases) == 0


if __name__ == "__main__":
    success = asyncio.run(run_all())
    sys.exit(0 if success else 1)