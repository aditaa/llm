import unittest

from llm.eval_policy import compare_summaries, evaluate_promotion_policy, summary_from_report


class EvalPolicyTests(unittest.TestCase):
    def test_summary_from_report(self) -> None:
        report = {
            "summary": {
                "pass_rate": 0.5,
                "check_pass_rate": 0.8,
                "avg_case_score": 0.7,
                "cases_total": 10,
                "cases_passed": 5,
                "checks_total": 20,
                "checks_passed": 16,
            }
        }
        summary = summary_from_report(report)
        self.assertAlmostEqual(summary.pass_rate, 0.5)
        self.assertEqual(summary.cases_total, 10)

    def test_compare_summaries(self) -> None:
        current = summary_from_report(
            {
                "summary": {
                    "pass_rate": 0.6,
                    "check_pass_rate": 0.7,
                    "avg_case_score": 0.8,
                    "cases_total": 10,
                    "cases_passed": 6,
                    "checks_total": 20,
                    "checks_passed": 14,
                }
            }
        )
        baseline = summary_from_report(
            {
                "summary": {
                    "pass_rate": 0.5,
                    "check_pass_rate": 0.65,
                    "avg_case_score": 0.78,
                    "cases_total": 10,
                    "cases_passed": 5,
                    "checks_total": 20,
                    "checks_passed": 13,
                }
            }
        )
        delta = compare_summaries(current, baseline)
        self.assertAlmostEqual(delta["pass_rate_delta"], 0.1)
        self.assertAlmostEqual(delta["check_pass_rate_delta"], 0.05)

    def test_promotion_absolute_policy_passes(self) -> None:
        current = summary_from_report(
            {
                "summary": {
                    "pass_rate": 0.7,
                    "check_pass_rate": 0.85,
                    "avg_case_score": 0.82,
                    "cases_total": 10,
                    "cases_passed": 7,
                    "checks_total": 20,
                    "checks_passed": 17,
                }
            }
        )
        policy = {
            "name": "p1",
            "absolute": {
                "min_pass_rate": 0.6,
                "min_check_pass_rate": 0.8,
                "min_avg_case_score": 0.8,
            },
        }
        result = evaluate_promotion_policy(current=current, baseline=None, policy=policy)
        self.assertTrue(result["promoted"])
        self.assertEqual(result["failed_checks"], [])

    def test_promotion_fails_on_regression_drop(self) -> None:
        current = summary_from_report(
            {
                "summary": {
                    "pass_rate": 0.6,
                    "check_pass_rate": 0.7,
                    "avg_case_score": 0.72,
                    "cases_total": 10,
                    "cases_passed": 6,
                    "checks_total": 20,
                    "checks_passed": 14,
                }
            }
        )
        baseline = summary_from_report(
            {
                "summary": {
                    "pass_rate": 0.7,
                    "check_pass_rate": 0.8,
                    "avg_case_score": 0.8,
                    "cases_total": 10,
                    "cases_passed": 7,
                    "checks_total": 20,
                    "checks_passed": 16,
                }
            }
        )
        policy = {
            "name": "p2",
            "regression": {"max_pass_rate_drop": 0.05},
        }
        result = evaluate_promotion_policy(current=current, baseline=baseline, policy=policy)
        self.assertFalse(result["promoted"])
        self.assertIn("max_pass_rate_drop", result["failed_checks"])

    def test_regression_policy_requires_baseline(self) -> None:
        current = summary_from_report(
            {
                "summary": {
                    "pass_rate": 0.7,
                    "check_pass_rate": 0.8,
                    "avg_case_score": 0.8,
                    "cases_total": 10,
                    "cases_passed": 7,
                    "checks_total": 20,
                    "checks_passed": 16,
                }
            }
        )
        policy = {"name": "p3", "regression": {"max_pass_rate_drop": 0.05}}
        result = evaluate_promotion_policy(current=current, baseline=None, policy=policy)
        self.assertFalse(result["promoted"])
        self.assertIn("max_pass_rate_drop", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()

