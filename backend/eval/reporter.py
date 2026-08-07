"""
Terminal output and summary reporting.
Completely decoupled from agent logic and validation.
Swap this file to change output format (e.g. JSON, HTML, CI-friendly).
"""

from eval.test_cases import URGENCY_ORDER

# ── Colours ───────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


def print_header(total: int):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  MedTriage AI — Evaluation Suite{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{DIM}  Running {total} test cases...\n{RESET}")


def print_test_result(case: dict, result: dict, failures: list, warnings: list):
    label    = f"Test {case['id']:02d} — {case['description']}"
    elapsed  = result.get("elapsed_seconds", 0)
    conf     = result.get("confidence")
    conf_str = f"  confidence={conf}%" if conf is not None else ""
    time_str = f"{elapsed}s"

    print(f"  {DIM}{label:<45}{RESET}", end=" ", flush=True)

    if not failures and not warnings:
        print(f"{GREEN}PASS{RESET}  {DIM}({time_str}{conf_str}){RESET}")

    elif not failures:
        print(f"{YELLOW}WARN{RESET}  {DIM}({time_str}{conf_str}){RESET}")
        for w in warnings:
            print(f"           {YELLOW}~ {w}{RESET}")

    else:
        print(f"{RED}FAIL{RESET}  {DIM}({time_str}{conf_str}){RESET}")
        for f in failures:
            print(f"           {RED}✗ {f}{RESET}")
        for w in warnings:
            print(f"           {YELLOW}~ {w}{RESET}")


def print_error(case: dict, error: str):
    label = f"Test {case['id']:02d} — {case['description']}"
    print(f"  {DIM}{label:<45}{RESET}", end=" ")
    print(f"{RED}ERROR{RESET}")
    print(f"           {RED}✗ {error}{RESET}")


def _section(title: str):
    print(f"\n{BOLD}{BLUE}{title}{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")


def print_summary(results: list, total_time: float):
    total    = len(results)
    passed   = sum(1 for r in results if r["passed"])
    warned   = sum(1 for r in results if r["passed"] and r["warnings"])
    failed   = total - passed
    avg_time = round(total_time / total, 2) if total else 0
    score    = round((passed / total) * 100) if total else 0

    score_colour = GREEN if score >= 80 else YELLOW if score >= 60 else RED

    _section("Summary")
    print(f"  Total tests  : {total}")
    print(f"  Passed       : {GREEN}{passed}{RESET}")
    print(f"  Failed       : {RED}{failed}{RESET}")
    print(f"  With warnings: {YELLOW}{warned}{RESET}")
    print(f"  Avg response : {avg_time}s")
    print(f"  Score        : {score_colour}{BOLD}{score}%{RESET}")
    print()

    if score == 100:
        print(f"  {GREEN}All tests passed. Agent classification is reliable.{RESET}")
    elif score >= 80:
        print(f"  {GREEN}Agent is performing well. Acceptable for demo and portfolio.{RESET}")
    elif score >= 60:
        print(f"  {YELLOW}Agent needs improvement. Review failed cases and refine prompts.{RESET}")
    else:
        print(f"  {RED}Agent is unreliable. Significant prompt engineering required.{RESET}")


def print_category_breakdown(results: list):
    _section("Accuracy by Urgency Category")

    for cat in URGENCY_ORDER:
        cat_cases  = [r for r in results if r["case"]["category"] == cat]
        cat_passed = [r for r in cat_cases if r["passed"]]
        if not cat_cases:
            continue
        pct    = round(len(cat_passed) / len(cat_cases) * 100)
        colour = GREEN if pct == 100 else YELLOW if pct >= 50 else RED
        bar    = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(
            f"  {cat.upper():<12} {colour}{bar}{RESET}  "
            f"{colour}{len(cat_passed)}/{len(cat_cases)} ({pct}%){RESET}"
        )


def print_failed_details(results: list):
    failed = [r for r in results if not r["passed"]]
    if not failed:
        return

    _section("Failed Test Details")
    for r in failed:
        print(f"\n  {BOLD}Test {r['case']['id']} — {r['case']['description']}{RESET}")
        print(f"  {DIM}Input    : {r['case']['input'][:90]}...{RESET}")
        if r["result"]:
            print(f"  Expected : {r['case']['expected_urgency']}")
            print(f"  Got      : {RED}{r['result']['actual_urgency']}{RESET}")
            if r["result"].get("advice"):
                print(f"  Advice   : {r['result']['advice'][:100]}")
        for f in r["failures"]:
            print(f"  {RED}Failure  : {f}{RESET}")


def print_recommendations(results: list):
    failed = [r for r in results if not r["passed"]]
    if not failed:
        return

    _section("Recommendations")

    urgency_fails = [
        r for r in failed
        if any("urgency" in f.lower() for f in r["failures"])
    ]
    if urgency_fails:
        print(f"  {YELLOW}Urgency mismatches detected.{RESET}")
        print(f"  → Review triage_decision_node prompt in nodes.py")
        print(f"  → Add concrete examples for each urgency level")
        print(f"  → Lower temperature to 0.1 for more consistent output\n")

    slow = [
        r for r in results
        if r["result"] and r["result"].get("elapsed_seconds", 0) > 5
    ]
    if slow:
        print(f"  {YELLOW}Slow response times detected.{RESET}")
        print(f"  → Check Groq API key rate limits")
        print(f"  → Consider reducing max_tokens in nodes.py\n")

    leak_fails = [
        r for r in failed
        if any("leaked" in f.lower() for f in r["failures"])
    ]
    if leak_fails:
        print(f"  {YELLOW}JSON leaking into replies detected.{RESET}")
        print(f"  → Review the visible_reply stripping logic in nodes.py\n")


def print_footer():
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}\n")