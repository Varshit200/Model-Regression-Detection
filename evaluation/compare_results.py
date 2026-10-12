
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

BASELINE_PATH = REPORTS_DIR / "baseline_evaluation.json"
CANDIDATE_PATH = REPORTS_DIR / "latest_evaluation.json"
COMPARISON_PATH = REPORTS_DIR / "latest_comparison.json"


REGRESSION_THRESHOLD = 5.0


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compare_reports(baseline: dict, candidate: dict) -> dict:
    baseline_results = {
        item["id"]: item for item in baseline["results"]
    }
    candidate_results = {
        item["id"]: item for item in candidate["results"]
    }

    common_ids = baseline_results.keys() & candidate_results.keys()
    regressions = []
    improvements = []

    for case_id in sorted(common_ids):
        old_passed = baseline_results[case_id]["passed"]
        new_passed = candidate_results[case_id]["passed"]

        if old_passed and not new_passed:
            regressions.append(case_id)
        elif not old_passed and new_passed:
            improvements.append(case_id)

    old_accuracy = baseline["accuracy_percent"]
    new_accuracy = candidate["accuracy_percent"]

    accuracy_change = round(new_accuracy - old_accuracy, 2)
    threshold_breached = accuracy_change < -REGRESSION_THRESHOLD

    return {
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "baseline_accuracy": old_accuracy,
        "candidate_accuracy": new_accuracy,
        "accuracy_change": accuracy_change,
        "regression_threshold": REGRESSION_THRESHOLD,
        "threshold_breached": threshold_breached,
        "regressions": regressions,
        "improvements": improvements,
        "regression_detected": (
            bool(regressions) or threshold_breached
        ),
    }


def main():
    if not BASELINE_PATH.exists() or not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            "Missing baseline or candidate report. "
            "Run the evaluation and save a baseline first."
        )

    baseline = load_report(BASELINE_PATH)
    candidate = load_report(CANDIDATE_PATH)

    comparison = compare_reports(baseline, candidate)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with COMPARISON_PATH.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    print("--- Regression Comparison ---")
    print(f"Baseline accuracy: {comparison['baseline_accuracy']:.2f}%")
    print(f"Candidate accuracy: {comparison['candidate_accuracy']:.2f}%")
    print(f"Accuracy change: {comparison['accuracy_change']:+.2f}%")
    print(
        f"Regression threshold: "
        f"{comparison['regression_threshold']:.2f} percentage points"
    )
    print(f"Threshold breached: {comparison['threshold_breached']}")
    print(f"Regressions: {comparison['regressions']}")
    print(f"Improvements: {comparison['improvements']}")
    print(f"Regression detected: {comparison['regression_detected']}")
    print(f"Comparison report saved to: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
