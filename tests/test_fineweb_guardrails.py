import json
import tempfile
import unittest
from pathlib import Path

from llm.fineweb_guardrails import validate_job_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class FineWebGuardrailsTests(unittest.TestCase):
    def test_validate_job_artifacts_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            files_list = root / "files.txt"
            files_list.write_text("a.parquet\n", encoding="utf-8")

            shard_file = output_dir / "train_000000.bin"
            shard_file.write_bytes(b"\x01\x00\x02\x00")
            val_file = output_dir / "val_000000.bin"
            val_file.write_bytes(b"\x03\x00")

            manifest = {
                "input_files": ["/tmp/a.parquet"],
                "token_dtype": "uint16",
                "train": {"total_tokens": 2, "shards": [{"path": "train_000000.bin", "tokens": 2}]},
                "val": {"total_tokens": 1, "shards": [{"path": "val_000000.bin", "tokens": 1}]},
            }
            _write_json(output_dir / "manifest.json", manifest)

            report = {"manifest": str(output_dir / "manifest.json"), "rows_sharded": 3}
            _write_json(root / "report.json", report)

            rows, tokens, shards = validate_job_artifacts(
                job_id="job1",
                report_path=root / "report.json",
                output_dir=output_dir,
                files_list=files_list,
            )
            self.assertEqual(rows, 3)
            self.assertEqual(tokens, 3)
            self.assertEqual(shards, 2)

    def test_validate_job_artifacts_fails_when_manifest_missing_expected_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            files_list = root / "files.txt"
            files_list.write_text("a.parquet\nb.parquet\n", encoding="utf-8")

            shard_file = output_dir / "train_000000.bin"
            shard_file.write_bytes(b"\x01\x00")

            manifest = {
                "input_files": ["/tmp/a.parquet"],
                "token_dtype": "uint16",
                "train": {"total_tokens": 1, "shards": [{"path": "train_000000.bin", "tokens": 1}]},
                "val": {"total_tokens": 0, "shards": []},
            }
            _write_json(output_dir / "manifest.json", manifest)

            report = {"manifest": str(output_dir / "manifest.json"), "rows_sharded": 1}
            _write_json(root / "report.json", report)

            with self.assertRaises(ValueError):
                validate_job_artifacts(
                    job_id="job1",
                    report_path=root / "report.json",
                    output_dir=output_dir,
                    files_list=files_list,
                )


if __name__ == "__main__":
    unittest.main()

