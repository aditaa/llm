from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm.corpus import (
    CleanCorpusConfig,
    CorpusQualityConfig,
    analyze_corpora,
    clean_corpora_batch,
    load_boilerplate_lines_from_report,
    save_quality_report,
)


class CorpusQualityTests(unittest.TestCase):
    def test_analyze_corpora_detects_boilerplate_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            f1 = root / "one.txt"
            f2 = root / "two.txt"

            boiler = "Shared boilerplate line for all pages and templates"
            f1.write_text(
                "\n".join(
                    [
                        "Home | About | Contact",
                        "This is an article about survival skills and water purification.",
                        boiler,
                        boiler,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            f2.write_text(
                "\n".join(
                    [
                        boiler,
                        "Another useful technical paragraph with enough letters for training text.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = analyze_corpora(
                [f1, f2],
                CorpusQualityConfig(
                    top_k=10,
                    boilerplate_min_occurrences=3,
                    boilerplate_min_files=2,
                    boilerplate_min_chars=10,
                    boilerplate_max_chars=200,
                ),
            )

            self.assertEqual(report["files_seen"], 2)
            self.assertGreater(report["duplicate_nonempty_lines"], 0)
            self.assertGreaterEqual(len(report["boilerplate_candidates"]), 1)
            self.assertIn(
                boiler,
                {row["line"] for row in report["boilerplate_candidates"]},
            )

            report_path = root / "quality.json"
            save_quality_report(report, report_path)
            loaded_lines = load_boilerplate_lines_from_report(report_path)
            self.assertIn(boiler, loaded_lines)


class CorpusCleaningTests(unittest.TestCase):
    def test_clean_corpora_batch_filters_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True, exist_ok=True)

            boiler = "Shared boilerplate line for all pages and templates"
            valid = "Valid technical content line about Linux kernel networking and routing"
            valid2 = "Another useful technical paragraph with enough letters for training text"

            file_a = input_dir / "a.txt"
            file_b = input_dir / "b.txt"
            file_a.write_text(
                "\n".join(
                    [
                        "short",
                        "1234567890123456789012345",
                        boiler,
                        valid,
                        valid,
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            file_b.write_text(
                "\n".join([valid, valid2]) + "\n",
                encoding="utf-8",
            )

            report = clean_corpora_batch(
                input_files=[file_a, file_b],
                output_dir=output_dir,
                config=CleanCorpusConfig(
                    min_chars=20,
                    max_chars=0,
                    min_alpha_ratio=0.20,
                    max_digit_ratio=0.35,
                    dedupe_within_file=True,
                    dedupe_global=True,
                    max_lines_per_file=0,
                    skip_existing=True,
                    output_suffix=".clean.txt",
                ),
                boilerplate_lines={boiler},
            )

            out_a = (output_dir / "a.clean.txt").read_text(encoding="utf-8").splitlines()
            out_b = (output_dir / "b.clean.txt").read_text(encoding="utf-8").splitlines()

            self.assertEqual(out_a, [valid])
            self.assertEqual(out_b, [valid2])

            totals = report["totals"]
            self.assertEqual(totals["kept_lines"], 2)
            self.assertEqual(totals["removed_too_short"], 1)
            self.assertEqual(totals["removed_high_digit"], 1)
            self.assertEqual(totals["removed_boilerplate"], 1)
            self.assertEqual(totals["removed_duplicate_within"], 1)
            self.assertEqual(totals["removed_duplicate_global"], 1)

            # Ensure report payload is JSON serializable.
            json.dumps(report)

    def test_clean_corpora_batch_strips_web_shell_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True, exist_ok=True)

            noisy = (
                "How to parse XML in Python? - Stack Overflow "
                "Stack Exchange Stack Overflow Questions Tags Users About "
                "Stack Overflow Public Questions Tags Users About "
                "Public Asked Apr 23 '21 at 12:09 Active Apr 23 '21 at 12:09 Viewed 53 times "
                "<div>How to parse XML in Python? How to parse XML in Python using lxml and "
                "ElementTree?</div>"
            )
            (input_dir / "a.txt").write_text(noisy + "\n", encoding="utf-8")

            report = clean_corpora_batch(
                input_files=[input_dir / "a.txt"],
                output_dir=output_dir,
                config=CleanCorpusConfig(
                    min_chars=20,
                    max_chars=0,
                    min_alpha_ratio=0.20,
                    max_digit_ratio=0.35,
                    dedupe_within_file=True,
                    dedupe_global=False,
                    max_lines_per_file=0,
                    skip_existing=True,
                    output_suffix=".clean.txt",
                    decode_html_entities=True,
                    strip_html_tags=True,
                    strip_site_suffixes=True,
                    strip_nav_phrases=True,
                    strip_stack_metadata=True,
                    collapse_repeated_prefix=True,
                    strip_inline_score_tokens=True,
                ),
                boilerplate_lines=set(),
            )

            out = (output_dir / "a.clean.txt").read_text(encoding="utf-8").strip()
            self.assertIn("How to parse XML in Python", out)
            self.assertNotIn("Stack Exchange", out)
            self.assertNotIn("<div>", out)
            self.assertNotIn("Asked", out)
            self.assertNotIn("Viewed", out)
            self.assertNotIn("Public", out)
            self.assertNotIn("? 0 ", out)
            self.assertEqual(report["totals"]["kept_lines"], 1)

    def test_clean_corpora_batch_en_only_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True, exist_ok=True)

            english = "This is a practical guide for building a reliable backup strategy on Linux."
            spanish = "Este es un ejemplo de texto en espanol para comprobar el filtro de idioma."
            mostly_code = "SELECT * FROM users WHERE id = 42; INSERT INTO logs VALUES ('x');"
            (input_dir / "a.txt").write_text(
                "\n".join([english, spanish, mostly_code]) + "\n",
                encoding="utf-8",
            )

            report = clean_corpora_batch(
                input_files=[input_dir / "a.txt"],
                output_dir=output_dir,
                config=CleanCorpusConfig(
                    min_chars=20,
                    max_chars=0,
                    min_alpha_ratio=0.20,
                    max_digit_ratio=0.35,
                    dedupe_within_file=True,
                    dedupe_global=False,
                    max_lines_per_file=0,
                    skip_existing=True,
                    output_suffix=".clean.txt",
                    english_only=True,
                    english_min_words=6,
                    english_min_stopword_ratio=0.05,
                    english_min_stopword_count=1,
                    english_min_latin_ratio=0.90,
                ),
                boilerplate_lines=set(),
            )

            out_lines = (output_dir / "a.clean.txt").read_text(encoding="utf-8").splitlines()
            self.assertIn(english, out_lines)
            self.assertNotIn(spanish, out_lines)
            self.assertNotIn(mostly_code, out_lines)
            self.assertEqual(report["totals"]["removed_non_english"], 2)


if __name__ == "__main__":
    unittest.main()
