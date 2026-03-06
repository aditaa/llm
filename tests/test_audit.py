import tempfile
import unittest
from pathlib import Path

from llm.audit import DatasetRiskConfig, analyze_dataset_risk


class DatasetRiskAuditTests(unittest.TestCase):
    def test_detects_heuristic_risk_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text(
                "\n".join(
                    [
                        "All women are lazy and men are superior.",
                        "I cannot help with that request.",
                        "The democrats and republican candidates had an election debate.",
                        "You are a stupid idiot.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            report = analyze_dataset_risk([path], DatasetRiskConfig(top_k=10))
            summary = report["summary"]

            self.assertEqual(report["files_seen"], 1)
            self.assertEqual(report["lines_nonempty"], 4)
            self.assertEqual(summary["lines_with_stereotype"], 1)
            self.assertEqual(summary["lines_with_refusal"], 1)
            self.assertEqual(summary["lines_with_political"], 1)
            self.assertEqual(summary["lines_with_toxicity"], 1)

    def test_respects_global_line_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text("line one\nline two\nline three\n", encoding="utf-8")

            report = analyze_dataset_risk(
                [path],
                DatasetRiskConfig(
                    top_k=5,
                    max_total_lines=2,
                ),
            )

            self.assertTrue(report["truncated"])
            self.assertEqual(report["lines_seen"], 2)


if __name__ == "__main__":
    unittest.main()
