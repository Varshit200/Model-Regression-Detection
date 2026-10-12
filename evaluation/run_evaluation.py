
import json
from datetime import datetime, timezone
from pathlib import Path

from app.classifier import classify_email

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "golden_dataset.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "latest_evaluation.json"


def main():
    with DATASET_PATH.open("r", encoding="utf-8-sig") as file:
        dataset = json.load(file)

    test_cases = dataset["test_cases"]
    total = len(test_cases)
    correct = 0
    results = []

    print("Starting Model Regression Evaluation...")
    print(f"Test cases loaded: {total}\n")

    for case in test_cases:
        expected = case["expected"]["category"]

        try:
            prediction = classify_email(case["input"])
            predicted = prediction.category
            passed = predicted == expected

            if passed:
                correct += 1

            results.append({
                "id": case["id"],
                "input": case["input"],
                "expected_category": expected,
                "predicted_category": predicted,
                "passed": passed,
                "summary": prediction.summary,
            })

            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] {case['id']} | "
                f"Expected: {expected} | Predicted: {predicted}"
            )

        except Exception as error:
            results.append({
                "id": case["id"],
                "expected_category": expected,
                "predicted_category": None,
                "passed": False,
                "error": str(error),
            })
            print(f"[ERROR] {case['id']}: {error}")

    accuracy = (correct / total * 100) if total else 0

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "mode": "mock",
        "total_cases": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "accuracy_percent": round(accuracy, 2),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n--- Evaluation Summary ---")
    print(f"Total cases: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
