
import unittest

from evaluation.compare_results import compare_reports


def make_report(accuracy, results):
    return {
        "accuracy_percent": accuracy,
        "results": results,
    }


def make_result(case_id, passed):
    return {
        "id": case_id,
        "passed": passed,
    }


class TestRegressionDetection(unittest.TestCase):

    def test_no_regression_when_accuracy_is_unchanged(self):
        baseline = make_report(
            100.0,
            [make_result("BILL-001", True)],
        )
        candidate = make_report(
            100.0,
            [make_result("BILL-001", True)],
        )

        result = compare_reports(baseline, candidate)

        self.assertFalse(result["regression_detected"])
        self.assertEqual(result["accuracy_change"], 0.0)

    def test_detects_accuracy_drop_beyond_threshold(self):
        baseline = make_report(
            100.0,
            [make_result("BILL-001", True)],
        )
        candidate = make_report(
            94.0,
            [make_result("BILL-001", True)],
        )

        result = compare_reports(baseline, candidate)

        self.assertTrue(result["threshold_breached"])
        self.assertTrue(result["regression_detected"])

    def test_detects_individual_test_regression(self):
        baseline = make_report(
            100.0,
            [make_result("BILL-001", True)],
        )
        candidate = make_report(
            98.0,
            [make_result("BILL-001", False)],
        )

        result = compare_reports(baseline, candidate)

        self.assertIn("BILL-001", result["regressions"])
        self.assertTrue(result["regression_detected"])


if __name__ == "__main__":
    unittest.main()
