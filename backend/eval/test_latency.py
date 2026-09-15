"""
Test 3 — System Latency and Performance (token-optimized)

Measures:
  - Full pipeline latency (orchestrator -> specialist -> safety) end-to-end
  - P50/max latency across a small, representative case set
  - Per-node timing breakdown, extracted from state["trace"] if populated
  - Latency by urgency category
  - Slow request detection
  - Rule-based tool timing (free — no LLM calls)

DESIGN NOTE: earlier version separately timed the orchestrator, each
specialist, and the safety agent in isolation (COMPONENT_CASES), on top
of timing the full pipeline. That double-counted LLM calls: a single
pipeline run already exercises all of those nodes. This version times
components ONLY via the trace emitted from real pipeline runs, cutting
LLM calls from ~45+ down to ~9 (3 cases x 1 run x 3 LLM nodes).

Run from backend/ with venv active:
    python -m eval.test_latency
"""

import asyncio
import time
import statistics
from langchain_core.messages import HumanMessage
from app.agent.graph import triage_graph
from app.agent.tools import calculate_clinical_score, check_red_flags
from app.agent.state import TriageState

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

THRESHOLDS = {
    "full_pipeline": 5.0,   # end-to-end including all agents
    "tool":          0.1,   # rule-based tools (no LLM)
    "p90_target":    4.0,
    "p99_target":    6.0,
}


def make_state(message: str, urgency: str = None) -> TriageState:
    return {
        "messages":            [HumanMessage(content=message)],
        "symptoms":            ["as described"],
        "duration":            "as described",
        "severity":            "as described",
        "age":                 "as described",
        "existing_conditions": [],
        "step_count":          3,
        "confidence":          70,
        "differential":        [],
        "specialist_called":   None,
        "tool_calls":          [],
        "needs_escalation":    False,
        "trace":               [],
        "awaiting_user_input": False,
        "triage_complete":     False,
        "questions_asked":     2,
        "urgency":             urgency,
        "safety_approved":     False,
        "advice":              "Test advice for safety review" if urgency else None,
        "symptoms_summary":    None,
    }


# ── Rule-based tool timing (no LLM, run freely) ───────────────
TOOL_CASES = [
    {
        "label":     "Clinical score tool (rule-based)",
        "input":     "Chest pain 9/10. Age 55, hypertension.",
        "component": "clinical_score",
        "runs":      10,
    },
    {
        "label":     "Red flag checker (rule-based)",
        "input":     "Chest pain radiating to left arm.",
        "component": "red_flag",
        "runs":      10,
    },
]


async def time_tool(component: str, state: TriageState) -> float:
    start = time.perf_counter()
    if component == "clinical_score":
        await calculate_clinical_score(state)
    elif component == "red_flag":
        await check_red_flags(state)
    return round(time.perf_counter() - start, 6)


async def test_tool_latency() -> list:
    print(f"\n{BOLD}{BLUE}Test 3a — Rule-based Tool Latency (no LLM calls){RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}\n")

    results = []
    for case in TOOL_CASES:
        state = make_state(case["input"], urgency="moderate")
        times = [await time_tool(case["component"], state) for _ in range(case["runs"])]

        avg = round(statistics.mean(times), 3)
        mn, mx = round(min(times), 3), round(max(times), 3)
        ok = avg <= THRESHOLDS["tool"]
        colour = GREEN if ok else RED

        print(f"  {case['label']:<35} min={mn}s avg={avg}s max={mx}s "
              f"{colour}{'OK' if ok else 'SLOW'}{RESET}")

        results.append({
            "label": case["label"], "component": case["component"],
            "min": mn, "avg": avg, "max": mx,
            "threshold": THRESHOLDS["tool"], "passed": ok,
        })
    return results


# ── Full pipeline latency (this is where LLM calls happen) ───
PIPELINE_CASES = [
    {
        "label":    "Low urgency — cold symptoms",
        "input":    "Mild runny nose. Severity 2/10. Age 28.",
        "category": "low",
    },
    {
        "label":    "High urgency — abdominal pain",
        "input":    "Severe abdominal pain 8/10. 6 hours. Age 40.",
        "category": "high",
    },
    {
        "label":    "Emergency — cardiac",
        "input":    "Crushing chest pain, left arm. Severity 9/10. Age 55, hypertension.",
        "category": "emergency",
    },
]

PIPELINE_RUNS = 1  # bump this back up only once the harness is verified correct


def extract_node_timings(trace: list) -> dict:
    """
    Best-effort extraction of per-node timing from state["trace"].
    Defensive: if entries don't have the expected shape, skip quietly
    rather than crashing — we don't yet know the exact trace schema
    orchestrator.py emits.
    """
    node_times = {}
    if not trace:
        return node_times
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        node = entry.get("node") or entry.get("name")
        duration = entry.get("duration") or entry.get("duration_ms") or entry.get("elapsed")
        if node and duration is not None:
            node_times.setdefault(node, []).append(duration)
    return node_times


async def test_pipeline_latency() -> dict:
    print(f"\n{BOLD}{BLUE}Test 3b — Full Pipeline Latency ({PIPELINE_RUNS} run(s) each){RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")
    print(f"  {DIM}End-to-end: orchestrator -> specialist -> safety agent{RESET}\n")

    all_times = []
    results = []
    all_node_times = {}

    for case in PIPELINE_CASES:
        times = []
        print(f"  {case['label']}")

        for run in range(PIPELINE_RUNS):
            state = make_state(case["input"])
            start = time.time()
            result = await triage_graph.ainvoke(state)
            elapsed = round(time.time() - start, 3)
            times.append(elapsed)

            node_times = extract_node_timings(result.get("trace", []))
            for node, durations in node_times.items():
                all_node_times.setdefault(node, []).extend(durations)

            urgency = result.get("urgency", "?")
            print(f"    Run {run+1}: {elapsed}s  urgency={urgency}")

        avg = round(statistics.mean(times), 3)
        mx  = round(max(times), 3)
        passed = avg <= THRESHOLDS["full_pipeline"]
        all_times.extend(times)

        colour = GREEN if passed else RED
        print(f"  avg={avg}s  max={mx}s  {colour}{'OK' if passed else 'SLOW'}{RESET}\n")

        results.append({
            "label": case["label"], "category": case["category"],
            "times": times, "avg": avg, "max": mx, "passed": passed,
        })

    if all_node_times:
        print(f"  {BOLD}Per-node breakdown (from trace){RESET}")
        for node, durations in all_node_times.items():
            print(f"    {node:<25} avg={round(statistics.mean(durations), 3)}s "
                  f"(n={len(durations)})")
    else:
        print(f"  {DIM}No per-node trace data found — trace format not recognized "
              f"or not populated. Update extract_node_timings() to match your "
              f"orchestrator's trace schema for a node breakdown.{RESET}")

    all_times_sorted = sorted(all_times)
    n = len(all_times_sorted)
    p50 = round(all_times_sorted[int(n * 0.50)], 3)
    p90 = round(all_times_sorted[min(int(n * 0.90), n - 1)], 3)
    p99 = round(all_times_sorted[min(int(n * 0.99), n - 1)], 3)
    mean = round(statistics.mean(all_times), 3)

    print(f"\n{BOLD}{BLUE}Latency Summary ({n} pipeline runs total){RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")

    p50_ok = p50 <= THRESHOLDS["full_pipeline"]
    p90_ok = p90 <= THRESHOLDS["p90_target"]
    p99_ok = p99 <= THRESHOLDS["p99_target"]

    print(f"  Mean : {mean}s")
    print(f"  P50  : {GREEN if p50_ok else RED}{p50}s{RESET}  (target <{THRESHOLDS['full_pipeline']}s)")
    print(f"  P90  : {GREEN if p90_ok else YELLOW}{p90}s{RESET}  (target <{THRESHOLDS['p90_target']}s)")
    print(f"  P99  : {GREEN if p99_ok else RED}{p99}s{RESET}  (target <{THRESHOLDS['p99_target']}s)")

    if n < 10:
        print(f"  {DIM}Note: small sample size (n={n}) — percentiles are illustrative, "
              f"not statistically robust. Increase PIPELINE_RUNS for a real distribution "
              f"once you have LLM budget to spare.{RESET}")

    return {
        "results": results, "all_times": all_times,
        "p50": p50, "p90": p90, "p99": p99, "mean": mean,
        "p90_passed": p90_ok, "p99_passed": p99_ok,
    }


async def test_latency_by_category(pipeline_results: dict) -> None:
    print(f"\n{BOLD}{BLUE}Test 3c — Latency by Urgency Category{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")

    by_cat = {}
    for r in pipeline_results["results"]:
        by_cat.setdefault(r["category"], []).extend(r["times"])

    for cat in ["low", "moderate", "high", "emergency"]:
        if cat not in by_cat:
            continue
        times = by_cat[cat]
        avg = round(statistics.mean(times), 3)
        mx  = round(max(times), 3)
        ok  = avg <= THRESHOLDS["full_pipeline"]
        colour = GREEN if ok else RED
        bar_pct = min(int(avg / THRESHOLDS["full_pipeline"] * 10), 10)
        bar = "█" * bar_pct + "░" * (10 - bar_pct)
        print(f"  {cat.upper():<12} {colour}{bar}{RESET}  avg={avg}s  max={mx}s")


async def test_slow_requests(pipeline_results: dict) -> None:
    print(f"\n{BOLD}{BLUE}Test 3d — Slow Request Detection{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")

    all_times = pipeline_results["all_times"]
    threshold = THRESHOLDS["full_pipeline"]
    slow = [t for t in all_times if t > threshold]

    if not slow:
        print(f"  {GREEN}No slow requests detected (all under {threshold}s){RESET}")
    else:
        print(f"  {RED}{len(slow)}/{len(all_times)} requests exceeded {threshold}s threshold{RESET}")
        for t in sorted(slow, reverse=True):
            print(f"  {RED}-> {t}s{RESET}")


def print_summary(tool_results: list, pipeline_results: dict):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  Test 3 — Performance Summary{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")

    tool_passed = sum(1 for r in tool_results if r["passed"])
    pipe_passed = sum(1 for r in pipeline_results["results"] if r["passed"])

    print(f"  Tool tests     : {tool_passed}/{len(tool_results)} within threshold")
    print(f"  Pipeline tests : {pipe_passed}/{len(pipeline_results['results'])} within threshold")
    print(f"  P50 latency    : {pipeline_results['p50']}s")
    print(f"  P90 latency    : {pipeline_results['p90']}s  (target <{THRESHOLDS['p90_target']}s)")
    print(f"  P99 latency    : {pipeline_results['p99']}s  (target <{THRESHOLDS['p99_target']}s)")

    if tool_results:
        tool_avg = round(statistics.mean(r["avg"] for r in tool_results), 3)
        pipe_avg = round(statistics.mean(r["avg"] for r in pipeline_results["results"]), 3)
        speedup = round(pipe_avg / tool_avg) if tool_avg > 0 else "∞"
        print(f"\n  Rule-based avg : {GREEN}{tool_avg}s{RESET}")
        print(f"  Pipeline avg   : {pipe_avg}s")
        print(f"  Speedup        : {GREEN}{speedup}x faster{RESET} for rule-based tools")

    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}\n")


async def main():
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  Test 3 — System Latency and Performance{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")

    tool_results = await test_tool_latency()
    pipeline_results = await test_pipeline_latency()

    await test_latency_by_category(pipeline_results)
    await test_slow_requests(pipeline_results)
    print_summary(tool_results, pipeline_results)


if __name__ == "__main__":
    asyncio.run(main())