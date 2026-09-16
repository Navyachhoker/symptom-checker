"""
Latency evaluation for the medical triage pipeline.

Run as a script (not via pytest — see NOTE below):
    python -m eval.test_latency
    PIPELINE_RUNS=10 python -m eval.test_latency
    python -m eval.test_latency --quiet --json

NOTE on naming: functions here are prefixed `check_*` rather than `test_*`
on purpose. If pytest ever collects this file, `test_*` async functions
with no `assert`s would be reported as "passed" regardless of whether
latency actually regressed. Real pass/fail is enforced explicitly below
via `sys.exit(1)`, independent of any test runner.

Fixes vs. previous version:
  - Tool-latency loop rebuilds `state` every iteration (previously reused
    one mutable dict across 1000 calls; if a tool appends to state lists
    as a side effect, later iterations silently got slower/faster).
  - One untimed warm-up call before each timed loop.
  - Real failure signal: raises the process exit code when any check fails.
  - Percentiles are computed per-category, not blended across categories
    that plausibly take different code paths.
  - Each pipeline call is wrapped in try/except so one transient API/
    network failure doesn't kill the whole run; failures are recorded
    and counted as failures, not silently dropped.
  - Output trimmed to one line per result (quiet mode drops color/labels
    entirely) to cut console/token volume when this is piped into logs
    or read by another LLM.
"""

import os
import sys
import json as jsonlib
import asyncio
import argparse
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.agent.graph import triage_graph
from app.agent.tools import calculate_clinical_score, check_red_flags
from app.agent.state import TriageState

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

C = {"g": "\033[92m", "r": "\033[91m", "y": "\033[93m", "b": "\033[94m",
     "c": "\033[96m", "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m"}


def paint(s, color, enabled):
    return f"{C[color]}{s}{C['reset']}" if enabled else s


THRESHOLDS = {
    "full_pipeline": 5.0,   # seconds, avg per category
    "tool_ms": 100.0,       # milliseconds, avg
    "p90": 4.0,             # seconds
    "p99": 6.0,             # seconds
}

MIN_N_FOR_TAIL = 10  # samples needed before trusting p90/p99

TOOL_CASES = [
    {"label": "clinical_score", "input": "Chest pain 9/10. Age 55, hypertension.", "runs": 1000},
    {"label": "red_flag", "input": "Chest pain radiating to left arm.", "runs": 1000},
]

PIPELINE_CASES = [
    {"label": "low", "input": "Mild runny nose. Severity 2/10. Age 28."},
    {"label": "high", "input": "Severe abdominal pain 8/10. 6 hours. Age 40."},
    {"label": "emergency", "input": "Crushing chest pain, left arm. Severity 9/10. Age 55, hypertension."},
]

PIPELINE_RUNS = int(os.environ.get("PIPELINE_RUNS", "1"))


def make_state(message: str) -> TriageState:
    return {
        "messages": [HumanMessage(content=message)],
        "symptoms": ["as described"], "duration": "as described",
        "severity": "as described", "age": "as described",
        "existing_conditions": [], "step_count": 3, "confidence": 70,
        "differential": [], "specialist_called": None, "tool_calls": [],
        "needs_escalation": False, "trace": [], "awaiting_user_input": False,
        "triage_complete": False, "questions_asked": 2, "urgency": "moderate",
        "safety_approved": False, "advice": "Test advice for safety review",
        "symptoms_summary": None,
    }


# ------------------------------------------------------------------
# Tool latency (rule-based, no LLM)
# ------------------------------------------------------------------

async def run_tool(label: str, message: str):
    state = make_state(message)
    start = time.perf_counter()
    if label == "clinical_score":
        await calculate_clinical_score(state)
    else:
        await check_red_flags(state)
    return (time.perf_counter() - start) * 1000


async def check_tool_latency(quiet: bool) -> list:
    if not quiet:
        print(f"\n{paint('Tool latency (rule-based)', 'bold', True)}")
    results = []
    for case in TOOL_CASES:
        await run_tool(case["label"], case["input"])  # warm-up, discarded
        times = [await run_tool(case["label"], case["input"]) for _ in range(case["runs"])]
        avg, mn, mx = round(statistics.mean(times), 3), round(min(times), 3), round(max(times), 3)
        passed = avg <= THRESHOLDS["tool_ms"]
        results.append({"label": case["label"], "avg_ms": avg, "min_ms": mn, "max_ms": mx, "passed": passed})
        if not quiet:
            status = paint("OK", "g", True) if passed else paint("SLOW", "r", True)
            print(f"  {case['label']:<16} avg={avg:>7.3f}ms min={mn:>7.3f}ms max={mx:>7.3f}ms {status}")
    return results


# ------------------------------------------------------------------
# Full pipeline latency
# ------------------------------------------------------------------

async def run_pipeline_once(message: str):
    state = make_state(message)
    start = time.perf_counter()
    try:
        result = await triage_graph.ainvoke(state)
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": time.perf_counter() - start}
    return {"ok": True, "elapsed": round(time.perf_counter() - start, 3), "result": result}


def percentiles(times: list):
    if len(times) < MIN_N_FOR_TAIL:
        return None, None
    qs = statistics.quantiles(times, n=100, method="inclusive")
    return round(qs[89], 3), round(qs[98], 3)


def extract_node_timings(trace: list) -> dict:
    """
    Best-effort extraction of per-node timing from state["trace"].

    KNOWN LIMITATION: the real AgentTrace schema (see app/agent/state.py)
    is {step, agent, action, reasoning, output, confidence} — it has no
    duration/elapsed field, and nothing in orchestrator.py or safety.py
    currently records per-node timing into the trace. This function will
    therefore always return an empty dict until timing instrumentation
    is added upstream (tracked separately — not a Phase 1 change).
    """
    node_times = {}
    if not trace:
        return node_times
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        node = entry.get("agent")
        duration = entry.get("duration") or entry.get("duration_ms") or entry.get("elapsed")
        if node and duration is not None:
            node_times.setdefault(node, []).append(duration)
    return node_times



async def check_pipeline_latency(quiet: bool) -> dict:
    if not quiet:
        print(f"\n{paint('Full pipeline latency', 'bold', True)} (runs/case={PIPELINE_RUNS})")

    results = []
    for case in PIPELINE_CASES:
        await run_pipeline_once(case["input"])  # warm-up, discarded

        times, errors = [], 0
        for _ in range(PIPELINE_RUNS):
            r = await run_pipeline_once(case["input"])
            if r["ok"]:
                times.append(r["elapsed"])
            else:
                errors += 1

        if not times:
            results.append({"label": case["label"], "passed": False, "errors": errors,
                             "avg": None, "min": None, "max": None, "p90": None, "p99": None})
            if not quiet:
                print(f"  {case['label']:<10} {paint(f'FAILED ({errors} errors, no successful runs)', 'r', True)}")
            continue

        avg, mn, mx = round(statistics.mean(times), 3), round(min(times), 3), round(max(times), 3)
        p90, p99 = percentiles(times)
        passed = avg <= THRESHOLDS["full_pipeline"] and errors == 0
        if p90 is not None:
            passed = passed and p90 <= THRESHOLDS["p90"] and p99 <= THRESHOLDS["p99"]

        results.append({"label": case["label"], "avg": avg, "min": mn, "max": mx,
                         "p90": p90, "p99": p99, "errors": errors, "passed": passed})

        if not quiet:
            status = paint("OK", "g", True) if passed else paint("FAIL", "r", True)
            tail = f" p90={p90:.3f}s p99={p99:.3f}s" if p90 is not None else " p90/p99=n/a(<10 samples)"
            err = f" errors={errors}" if errors else ""
            print(f"  {case['label']:<10} avg={avg:.3f}s min={mn:.3f}s max={mx:.3f}s{tail}{err} {status}")

    return {"results": results}


# ------------------------------------------------------------------
# Reporting / exit code
# ------------------------------------------------------------------

def summarize(tool_results: list, pipeline_results: dict, quiet: bool, as_json: bool) -> bool:
    all_passed = all(r["passed"] for r in tool_results) and all(r["passed"] for r in pipeline_results["results"])

    if as_json:
        print(jsonlib.dumps({
            "tool_results": tool_results,
            "pipeline_results": pipeline_results["results"],
            "passed": all_passed,
        }))
        return all_passed

    if not quiet:
        print(f"\n{paint('=' * 50, 'c', True)}")
    status = paint("PASS", "g", True) if all_passed else paint("FAIL", "r", True)
    print(f"{paint('LATENCY SUMMARY', 'bold', True)}: {status}")
    if not quiet:
        for r in tool_results:
            if not r["passed"]:
                print(f"  tool:{r['label']} avg={r['avg_ms']}ms > {THRESHOLDS['tool_ms']}ms")
        for r in pipeline_results["results"]:
            if not r["passed"]:
                reason = f"errors={r.get('errors', 0)}" if not r["avg"] else f"avg={r['avg']}s"
                print(f"  pipeline:{r['label']} {reason}")
        print(f"{paint('=' * 50, 'c', True)}")

    return all_passed


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="Only print the final pass/fail line.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary instead of text.")
    args = parser.parse_args()

    tool_results = await check_tool_latency(args.quiet)
    pipeline_results = await check_pipeline_latency(args.quiet)
    passed = summarize(tool_results, pipeline_results, args.quiet, args.json)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())