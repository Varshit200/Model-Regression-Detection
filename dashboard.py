
import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Model Regression Detection",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Model Regression Detection System")
st.caption("Evaluate model quality and identify performance regressions.")

evaluation_path = REPORTS / "latest_evaluation.json"
comparison_path = REPORTS / "latest_comparison.json"


def load_json(path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


evaluation = load_json(evaluation_path)
comparison = load_json(comparison_path)

if evaluation is None or comparison is None:
    st.error("Reports are missing or invalid. Run the evaluation pipeline first.")
    st.code("python run_pipeline.py")
    st.stop()

baseline = comparison.get("baseline_accuracy", 0)
candidate = comparison.get(
    "candidate_accuracy",
    evaluation.get("accuracy_percent", 0),
)
change = comparison.get("accuracy_change", candidate - baseline)
threshold = comparison.get("regression_threshold", 5)
regressions = comparison.get("regressions", [])
detected = comparison.get("regression_detected", False)

if detected:
    st.error("🚨 REGRESSION DETECTED — Review the failing test cases.")
else:
    st.success("✅ No regression detected against the saved baseline.")

st.subheader("Performance Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline Accuracy", f"{baseline:.2f}%")
col2.metric("Candidate Accuracy", f"{candidate:.2f}%")
col3.metric("Accuracy Change", f"{change:+.2f} pp")
col4.metric("Regression Threshold", f"{threshold:.2f} pp")

st.subheader("Evaluation Summary")

total = evaluation.get("total_cases", 0)
correct = evaluation.get("correct_predictions", 0)
incorrect = evaluation.get("incorrect_predictions", 0)

col1, col2, col3 = st.columns(3)
col1.metric("Total Test Cases", total)
col2.metric("Correct Predictions", correct)
col3.metric("Incorrect Predictions", incorrect)

if total:
    st.progress(max(0.0, min(1.0, correct / total)))
    st.caption(f"Accuracy: {evaluation.get('accuracy_percent', 0):.2f}%")

st.subheader("Failed Regression Cases")

results = evaluation.get("results", [])
failed = [item for item in results if not item.get("passed", False)]

if failed:
    st.dataframe(
        [
            {
                "Test ID": item.get("id"),
                "Input": item.get("input"),
                "Expected": item.get("expected_category"),
                "Predicted": item.get("predicted_category"),
            }
            for item in failed
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No failed cases in the latest evaluation report.")

st.subheader("All Evaluation Cases")

if results:
    st.dataframe(
        [
            {
                "Test ID": item.get("id"),
                "Expected": item.get("expected_category"),
                "Predicted": item.get("predicted_category"),
                "Result": "PASS" if item.get("passed") else "FAIL",
                "Summary": item.get("summary", ""),
            }
            for item in results
        ],
        use_container_width=True,
        hide_index=True,
    )

with st.expander("View raw regression comparison"):
    st.json(comparison)

with st.expander("View raw evaluation report"):
    st.json(evaluation)

st.caption(
    "Reports are read from the reports/ folder. "
    "Rerun your evaluation pipeline to refresh the results."
)
