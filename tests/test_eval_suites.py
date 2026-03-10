import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "configs" / "eval"
ALLOWED_CHECK_KEYS = {
    "min_completion_chars",
    "expect_any",
    "reject_any",
    "max_url_count",
    "max_repeated_char_run",
    "max_non_ascii_ratio",
}


class EvalSuiteConfigTests(unittest.TestCase):
    def test_eval_suites_use_supported_check_keys(self) -> None:
        for suite_path in sorted(EVAL_DIR.glob("*suite*.json")):
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            self.assertIsInstance(suite.get("name"), str, msg=str(suite_path))
            cases = suite.get("cases")
            self.assertIsInstance(cases, list, msg=str(suite_path))
            self.assertGreater(len(cases), 0, msg=str(suite_path))
            for case in cases:
                self.assertIsInstance(case.get("id"), str, msg=str(suite_path))
                self.assertIsInstance(case.get("prompt"), str, msg=str(suite_path))
                checks = case.get("checks")
                self.assertIsInstance(checks, dict, msg=str(suite_path))
                unknown = set(checks.keys()) - ALLOWED_CHECK_KEYS
                self.assertEqual(set(), unknown, msg=f"{suite_path}: {case.get('id')}")

    def test_phase1_talk_suite_is_non_coding(self) -> None:
        suite_path = EVAL_DIR / "english_talk_suite_v1.json"
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        categories = {str(case.get("category", "")).lower() for case in suite.get("cases", [])}
        self.assertNotIn("coding", categories)

    def test_phase1_talk_wrapper_uses_talk_configs(self) -> None:
        wrapper = (REPO_ROOT / "scripts" / "train_supervisor_phase1_english_talk.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("configs/eval/english_talk_suite_v1.json", wrapper)
        self.assertIn("configs/eval/generation_talk_smoke_v1.json", wrapper)
        self.assertIn("configs/eval/promotion_policy_talk_v1.json", wrapper)

    def test_talk_promotion_policy_schema(self) -> None:
        policy_path = EVAL_DIR / "promotion_policy_talk_v1.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        for section in ("absolute", "regression", "improvement"):
            self.assertIn(section, policy, msg=str(policy_path))
            self.assertIsInstance(policy[section], dict, msg=str(policy_path))

