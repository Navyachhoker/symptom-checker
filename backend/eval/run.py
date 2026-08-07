"""
Entry point for the evaluation suite.

Run from backend/ with venv active:
    python -m eval.run
"""

import asyncio
from eval.test_cases import TEST_CASES
from eval.runner     import run_single
from eval.validators import validate
from eval.reporter   import (
    print_header,
    print_test_result,
    print_error,
    print_summary,
    print_category_breakdown,
    print_failed_details,
    print_recommendations,
    print_footer,
)


async def main():
    print_header(len(TEST_CASES))

    results    = []
    total_time = 0.0

    for case in TEST_CASES:
        try:
            result             = await run_single(case)
            failures, warnings = validate(case, result)
            total_time        += result["elapsed_seconds"]

            print_test_result(case, result, failures, warnings)

            results.append({
                "case":     case,
                "result":   result,
                "failures": failures,
                "warnings": warnings,
                "passed":   len(failures) == 0,
            })

        except Exception as e:
            print_error(case, str(e))
            results.append({
                "case":     case,
                "result":   None,
                "failures": [str(e)],
                "warnings": [],
                "passed":   False,
            })

    print_summary(results, total_time)
    print_category_breakdown(results)
    print_failed_details(results)
    print_recommendations(results)
    print_footer()


if __name__ == "__main__":
    asyncio.run(main())