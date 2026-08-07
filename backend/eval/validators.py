"""
Validation logic.
Each function takes a case and result and returns (failures, warnings).
Failures count against the score. Warnings are noted but do not fail the test.

To add a new check, define a function following the same signature
and add it to CHECKS at the bottom.
"""

from eval.test_cases import VALID_URGENCY_LEVELS, URGENCY_ORDER


def urgency_distance(expected: str, actual: str) -> int:
    """How many urgency levels apart are two classifications."""
    if expected not in URGENCY_ORDER or actual not in URGENCY_ORDER:
        return 99
    return abs(URGENCY_ORDER.index(expected) - URGENCY_ORDER.index(actual))


# ── Individual checks ─────────────────────────────────────────

def check_urgency(case: dict, result: dict) -> tuple[list, list]:
    failures, warnings = [], []
    actual   = result.get("actual_urgency")
    expected = case["expected_urgency"]

    if actual != expected:
        distance = urgency_distance(expected, actual)
        if distance == 1:
            warnings.append(
                f"Urgency off by one level — expected '{expected}' got '{actual}'"
            )
        else:
            failures.append(
                f"Urgency mismatch — expected '{expected}' got '{actual}'"
            )
    return failures, warnings


def check_valid_urgency_value(case: dict, result: dict) -> tuple[list, list]:
    actual = result.get("actual_urgency")
    if actual not in VALID_URGENCY_LEVELS:
        return [f"Invalid urgency value returned: '{actual}'"], []
    return [], []


def check_advice_quality(case: dict, result: dict) -> tuple[list, list]:
    advice = result.get("advice", "")
    if not advice or len(advice.strip()) < 20:
        return ["Advice text is empty or too short (under 20 chars)"], []
    return [], []


def check_no_json_leak(case: dict, result: dict) -> tuple[list, list]:
    reply = result.get("reply", "")
    if "<extract>" in reply or '"symptoms":' in reply:
        return ["Internal extraction block or JSON leaked into reply"], []
    return [], []


def check_triage_completion(case: dict, result: dict) -> tuple[list, list]:
    if not result.get("is_complete"):
        return ["Triage did not reach completion after 3 turns"], []
    return [], []


def check_response_time(case: dict, result: dict) -> tuple[list, list]:
    elapsed = result.get("elapsed_seconds", 0)
    if elapsed > 10:
        return [f"Response too slow — {elapsed}s (limit 10s)"], []
    return [], []


def check_confidence_present(case: dict, result: dict) -> tuple[list, list]:
    if result.get("confidence") is None:
        return [], ["Confidence score was not returned"]
    if result.get("confidence", 100) < 50:
        return [], [
            f"Confidence is very low ({result['confidence']}%) — "
            "agent may have insufficient information"
        ]
    return [], []


def check_symptoms_summary(case: dict, result: dict) -> tuple[list, list]:
    if not result.get("symptoms_summary"):
        return [], ["Symptoms summary was not returned"]
    return [], []


# ── Registry — add new checks here ───────────────────────────
CHECKS = [
    check_urgency,
    check_valid_urgency_value,
    check_advice_quality,
    check_no_json_leak,
    check_triage_completion,
    check_response_time,
    check_confidence_present,
    check_symptoms_summary,
]


def validate(case: dict, result: dict) -> tuple[list, list]:
    """
    Run all registered checks against a result.
    Returns (all_failures, all_warnings).
    """
    all_failures, all_warnings = [], []
    for check in CHECKS:
        failures, warnings = check(case, result)
        all_failures.extend(failures)
        all_warnings.extend(warnings)
    return all_failures, all_warnings