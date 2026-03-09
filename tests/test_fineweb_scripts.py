import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

from llm.tokenizer import BPETokenizer

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_parquet(path: Path, rows: list[str]) -> None:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required to write parquet fixtures")
    table = pa.table({"text": rows})
    pq.write_table(table, path)


def _read_tokens(manifest: dict[str, object], output_dir: Path, split: str) -> list[int]:
    token_dtype = str(manifest["token_dtype"])
    array_type = "H" if token_dtype == "uint16" else "I"
    split_meta = manifest[split]
    assert isinstance(split_meta, dict)
    shards = split_meta["shards"]
    assert isinstance(shards, list)

    all_tokens: list[int] = []
    for shard in shards:
        assert isinstance(shard, dict)
        shard_path = output_dir / str(shard["path"])
        token_count = int(shard["tokens"])
        values = array(array_type)
        with shard_path.open("rb") as handle:
            values.fromfile(handle, token_count)
        all_tokens.extend(int(v) for v in values)
    return all_tokens


@unittest.skipIf(pa is None or pq is None, "pyarrow is required for parquet fixture tests")
class FineWebParquetToShardsTests(unittest.TestCase):
    def test_encode_batch_size_does_not_change_tokens(self) -> None:
        repo_root = _repo_root()
        script = repo_root / "scripts" / "fineweb_parquet_to_shards.py"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir(parents=True, exist_ok=True)

            rows_a = [f"alpha line {i}" for i in range(24)]
            rows_b = [f"beta line {i}" for i in range(24)]
            _write_parquet(input_dir / "000.parquet", rows_a)
            _write_parquet(input_dir / "001.parquet", rows_b)

            files_list = tmp_path / "files.txt"
            files_list.write_text("000.parquet\n001.parquet\n", encoding="utf-8")

            tokenizer_path = tmp_path / "tok.json"
            tokenizer = BPETokenizer.train_from_iterator(
                [*(line + "\n" for line in rows_a), *(line + "\n" for line in rows_b)],
                vocab_size=512,
                min_frequency=1,
            )
            tokenizer.save(tokenizer_path)

            output_a = tmp_path / "out_a"
            output_b = tmp_path / "out_b"
            report_a = tmp_path / "report_a.json"
            report_b = tmp_path / "report_b.json"

            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root / "src")

            base_args = [
                str(script),
                "--input-dir",
                str(input_dir),
                "--files-list",
                str(files_list),
                "--tokenizer-in",
                str(tokenizer_path),
                "--field",
                "text",
                "--batch-size",
                "8",
                "--shard-size-tokens",
                "64",
                "--val-ratio",
                "0.2",
                "--seed",
                "17",
                "--min-chars",
                "1",
            ]
            run_a = [
                sys.executable,
                *base_args,
                "--output-dir",
                str(output_a),
                "--encode-batch-size",
                "1",
                "--report-output",
                str(report_a),
            ]

            proc_a = subprocess.run(
                run_a,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_a.returncode, 0, msg=proc_a.stderr)
            run_b = [
                sys.executable,
                *base_args,
                "--output-dir",
                str(output_b),
                "--encode-batch-size",
                "16",
                "--report-output",
                str(report_b),
            ]

            proc_b = subprocess.run(
                run_b,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_b.returncode, 0, msg=proc_b.stderr)

            manifest_a = json.loads((output_a / "manifest.json").read_text(encoding="utf-8"))
            manifest_b = json.loads((output_b / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest_a["line_count"], manifest_b["line_count"])
            self.assertEqual(
                manifest_a["train"]["total_tokens"], manifest_b["train"]["total_tokens"]
            )
            self.assertEqual(manifest_a["val"]["total_tokens"], manifest_b["val"]["total_tokens"])
            self.assertEqual(
                len(manifest_a["train"]["shards"]), len(manifest_b["train"]["shards"])
            )
            self.assertEqual(len(manifest_a["val"]["shards"]), len(manifest_b["val"]["shards"]))

            self.assertEqual(
                _read_tokens(manifest_a, output_a, "train"),
                _read_tokens(manifest_b, output_b, "train"),
            )
            self.assertEqual(
                _read_tokens(manifest_a, output_a, "val"),
                _read_tokens(manifest_b, output_b, "val"),
            )


@unittest.skipIf(shutil.which("rsync") is None, "rsync is required for stage script")
class StageFineWebFromWarmTests(unittest.TestCase):
    def test_skip_list_excludes_blocked_parquet(self) -> None:
        repo_root = _repo_root()
        script = repo_root / "scripts" / "stage_fineweb_from_warm.sh"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_dir = tmp_path / "src"
            dest_dir = tmp_path / "dest"
            src_dir.mkdir(parents=True, exist_ok=True)
            dest_dir.mkdir(parents=True, exist_ok=True)

            (src_dir / "a.parquet").write_bytes(b"A")
            (src_dir / "b.parquet").write_bytes(b"B")
            (src_dir / "c.parquet").write_bytes(b"C")

            skip_list = tmp_path / "skip.txt"
            skip_list.write_text("b.parquet\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    "bash",
                    str(script),
                    "--src-dir",
                    str(src_dir),
                    "--dest-dir",
                    str(dest_dir),
                    "--max-files",
                    "10",
                    "--max-gib",
                    "0",
                    "--min-age-seconds",
                    "0",
                    "--skip-list",
                    str(skip_list),
                ],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            copied = sorted(p.name for p in dest_dir.glob("*.parquet"))
            self.assertEqual(copied, ["a.parquet", "c.parquet"])


class FineWebManifestDedupeTests(unittest.TestCase):
    def test_dedupe_disables_older_overlapping_manifest(self) -> None:
        repo_root = _repo_root()
        script = repo_root / "scripts" / "fineweb_manifest_dedupe.py"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shards_root = tmp_path / "shards"
            shards_root.mkdir(parents=True, exist_ok=True)

            old_dir = shards_root / "batch_old"
            new_dir = shards_root / "batch_new"
            unique_dir = shards_root / "batch_unique"
            old_dir.mkdir()
            new_dir.mkdir()
            unique_dir.mkdir()

            old_manifest = old_dir / "manifest.json"
            old_manifest.write_text(
                json.dumps({"input_files": ["000_00001.parquet", "000_00002.parquet"]}),
                encoding="utf-8",
            )
            new_manifest = new_dir / "manifest.json"
            new_manifest.write_text(
                json.dumps({"input_files": ["000_00001.parquet", "000_00002.parquet"]}),
                encoding="utf-8",
            )
            unique_manifest = unique_dir / "manifest.json"
            unique_manifest.write_text(
                json.dumps({"input_files": ["000_00003.parquet"]}),
                encoding="utf-8",
            )

            now = os.path.getmtime(new_manifest)
            os.utime(old_manifest, (now - 60, now - 60))
            os.utime(new_manifest, (now, now))

            report = tmp_path / "dedupe_report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--shards-root",
                    str(shards_root),
                    "--report-output",
                    str(report),
                    "--keep",
                    "newest",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            self.assertFalse(old_manifest.exists())
            self.assertTrue((old_dir / "manifest.duplicate.disabled.json").exists())
            self.assertTrue(new_manifest.exists())
            self.assertTrue(unique_manifest.exists())

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["manifest_total"], 3)
            self.assertEqual(payload["manifest_kept"], 2)
            self.assertEqual(payload["manifest_overlap"], 1)
            self.assertEqual(payload["unique_input_files"], 3)
            self.assertEqual(len(payload["disabled"]), 1)


if __name__ == "__main__":
    unittest.main()
