import json
from pathlib import Path

report_path = Path("reports/latest_evaluation.json")

with report_path.open("r", encoding="utf-8") as file:
    report = json.load(file)


case = next(item for item in report["results"] if item["id"] == "BILL-001")
case["predicted_category"] = "technical"
case["passed"] = False


correct = sum(item["passed"] for item in report["results"])
total = len(report["results"])

report["correct_predictions"] = correct
report["incorrect_predictions"] = total - correct
report["accuracy_percent"] = round(correct / total * 100, 2)

with report_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("Simulated one regression in the candidate report.")
print(f"Candidate accuracy: {report['accuracy_percent']:.2f}%")
