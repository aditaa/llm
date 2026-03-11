import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch

    from llm.model import GPTModel, ModelConfig
    from llm.tokenizer import BPETokenizer
except ModuleNotFoundError:
    torch = None
    GPTModel = None
    ModelConfig = None
    BPETokenizer = None


class ScriptTests(unittest.TestCase):
    def test_hf_download_watchdog_requires_completion_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_dir = root / "hf"
            proc = subprocess.run(
                [
                    "bash",
                    "scripts/hf_download_watchdog.sh",
                    "--dataset",
                    "HuggingFaceFW/fineweb",
                    "--repo-type",
                    "dataset",
                    "--include",
                    "sample/350BT/*.parquet",
                    "--local-dir",
                    str(local_dir),
                    "--exit-on-complete",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn(
                "--exit-on-complete requires --expected-parquet-files and/or --expected-bytes",
                proc.stderr,
            )

    def test_train_supervisor_help_lists_unique_input_gate(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/train_supervisor_rtx5070_350bt.sh", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--min-unique-input-files", proc.stdout)
        self.assertIn("--min-train-tokens", proc.stdout)
        self.assertIn("--train-stall-kill-seconds", proc.stdout)
        self.assertIn("--dedupe-report-keep", proc.stdout)

    def test_stage_loop_help_lists_stage_copy_jobs(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/fineweb_stage_shard_loop.sh", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--stage-copy-jobs", proc.stdout)
        self.assertIn("--stage-min-free-gib", proc.stdout)
        self.assertIn("--auto-tune-shard-jobs", proc.stdout)
        self.assertIn("--no-auto-tune-stage-copy-jobs", proc.stdout)
        self.assertIn("--expected-unique-input-files", proc.stdout)
        self.assertIn("--sync-background", proc.stdout)

    def test_benchmark_ctx_profiles_help_lists_profile_option(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/benchmark_rtx5070_context_profiles.sh", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--profiles", proc.stdout)
        self.assertIn("--compile-model", proc.stdout)
        self.assertIn("--sample-seconds", proc.stdout)

    def test_install_user_systemd_services_help(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/install_user_systemd_services.sh", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--user-systemd-dir", proc.stdout)
        self.assertIn("--install-watchdog", proc.stdout)

    def test_stage_watchdog_help_lists_cleanup_option(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/fineweb_stage_shard_watchdog.sh", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--no-cleanup-stale-workers", proc.stdout)
        self.assertIn("--expected-unique-input-files", proc.stdout)
        self.assertIn("--lock-file", proc.stdout)
        self.assertIn("--no-adopt-existing-loop", proc.stdout)

    def test_stage_watchdog_lock_is_independent_of_log_file(self) -> None:
        if (
            subprocess.run(
                ["pgrep", "-af", r"scripts/fineweb_stage_shard_loop.sh"],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        ):
            self.skipTest("stage loop already running on host")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hot = root / "hot"
            shards = root / "shards"
            state = root / "state"
            hot.mkdir(parents=True, exist_ok=True)
            shards.mkdir(parents=True, exist_ok=True)
            state.mkdir(parents=True, exist_ok=True)
            processed_file = state / "processed_parquet_files.txt"
            processed_file.write_text("", encoding="utf-8")
            lock_file = state / "watchdog.lock"
            log_a = state / "watchdog_a.log"
            log_b = state / "watchdog_b.log"

            # Provide an adoptable fake stage-loop process so watchdog stays alive.
            dummy = subprocess.Popen(
                ["bash", "-lc", "exec -a scripts/fineweb_stage_shard_loop.sh sleep 30"],
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            watchdog = subprocess.Popen(
                [
                    "bash",
                    "scripts/fineweb_stage_shard_watchdog.sh",
                    "--watchdog-log-file",
                    str(log_a),
                    "--lock-file",
                    str(lock_file),
                    "--check-interval-seconds",
                    "60",
                    "--stall-seconds",
                    "600",
                    "--hot-parquet-dir",
                    str(hot),
                    "--shards-root",
                    str(shards),
                    "--processed-file",
                    str(processed_file),
                    "--no-cleanup-stale-workers",
                ],
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                time.sleep(1.5)
                self.assertIsNone(
                    watchdog.poll(), "primary watchdog should still be running with lock held"
                )
                proc = subprocess.run(
                    [
                        "bash",
                        "scripts/fineweb_stage_shard_watchdog.sh",
                        "--watchdog-log-file",
                        str(log_b),
                        "--lock-file",
                        str(lock_file),
                        "--check-interval-seconds",
                        "60",
                        "--stall-seconds",
                        "600",
                        "--hot-parquet-dir",
                        str(hot),
                        "--shards-root",
                        str(shards),
                        "--processed-file",
                        str(processed_file),
                        "--no-cleanup-stale-workers",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 3, msg=proc.stderr)
                self.assertIn(
                    "another fineweb_stage_shard_watchdog instance is already running",
                    proc.stderr,
                )
            finally:
                watchdog.terminate()
                try:
                    watchdog.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    watchdog.kill()
                    watchdog.wait(timeout=5)
                dummy.terminate()
                try:
                    dummy.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    dummy.kill()
                    dummy.wait(timeout=5)

    def test_stage_from_warm_help_lists_lock_and_timeout_options(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/stage_fineweb_from_warm.sh", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--lock-wait-seconds", proc.stdout)
        self.assertIn("--rsync-retries", proc.stdout)
        self.assertIn("--rsync-timeout-seconds", proc.stdout)

    def test_checkpoint_offload_prune_help(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/checkpoint_offload_prune.sh", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--keep-local-runs", proc.stdout)
        self.assertIn("--sync-only", proc.stdout)

    def test_revalidate_bad_parquet_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/revalidate_bad_parquet.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--restage-valid", proc.stdout)
        self.assertIn("--no-rewrite-bad-list", proc.stdout)

    def test_offload_shard_bins_skip_if_missing_trained_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards_root = root / "shards"
            warm_root = root / "warm"
            shards_root.mkdir(parents=True, exist_ok=True)
            warm_root.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/offload_shard_bins_to_warm.py",
                    "--shards-root",
                    str(shards_root),
                    "--warm-shards-root",
                    str(warm_root),
                    "--require-trained-batches-file",
                    str(root / "missing_trained.txt"),
                    "--skip-if-trained-file-missing",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("shard_offload_skip", proc.stdout)

    def test_set_swappiness_help(self) -> None:
        proc = subprocess.run(
            ["bash", "scripts/set_swappiness.sh", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--value", proc.stdout)
        self.assertIn("--persist", proc.stdout)

    def test_render_eval_trend_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tsv = root / "eval_trend.tsv"
            tsv.write_text(
                "\t".join(
                    [
                        "run_tag",
                        "step",
                        "eval_rc",
                        "pass_rate",
                        "check_pass_rate",
                        "avg_case_score",
                        "cases_passed",
                        "cases_total",
                        "regression_pass",
                        "promotion_pass",
                        "failed_checks",
                        "baseline_report",
                        "report_json",
                    ]
                )
                + "\n"
                + "\n".join(
                    [
                        "run1\t1000\t0\t0.20\t0.70\t0.65\t3\t15\tNA\tNA\tnone\tNA\treport1.json",
                        "run2\t2000\t0\t0.30\t0.72\t0.68\t4\t15\tTrue\tFalse\tnone\treport1.json\treport2.json",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            html = root / "dashboard.html"
            summary = root / "summary.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/render_eval_trend_dashboard.py",
                    "--input-tsv",
                    str(tsv),
                    "--output-html",
                    str(html),
                    "--output-json",
                    str(summary),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(html.exists())
            self.assertTrue(summary.exists())
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["latest_step"], 2000)

    def test_pipeline_live_view_reports_supervisor_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            hot = root / "hot"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, hot, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            (sup_dir / "supervisor_20260309_100303.log").write_text(
                (
                    "[2026-03-09T10:08:04-05:00] "
                    "waiting_for_unique_inputs have=27 need=510 "
                    "overlap_inputs=0 overlap_manifests=0 sleep=60s\n"
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_live_view.py",
                    "--once",
                    "--no-alt-screen",
                    "--refresh-seconds",
                    "0.1",
                    "--warm-dir",
                    str(warm),
                    "--hot-dir",
                    str(hot),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--expected-parquet-files",
                    "510",
                    "--expected-bytes",
                    "1061360917731",
                    "--train-target-step",
                    "100000",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("Supervisor: gate=waiting_unique_inputs 27/510", proc.stdout)

    def test_pipeline_live_view_coverage_eta_from_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            hot = root / "hot"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, hot, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            (stage_dir / "processed_parquet_files.txt").write_text(
                "000_00001.parquet\n000_00002.parquet\n",
                encoding="utf-8",
            )
            (shards / "batch_0001").mkdir(parents=True, exist_ok=True)
            (shards / "batch_0001" / "manifest.json").write_text(
                json.dumps({"input_files": ["000_00001.parquet", "000_00002.parquet"]}),
                encoding="utf-8",
            )
            (stage_dir / "loop_20260309_100000.log").write_text(
                "\n".join(
                    [
                        (
                            "[2026-03-09T10:00:00-05:00] batch_start id=fw_a files=4 "
                            "tokenizer_arg=--tokenizer-in shard_jobs=2"
                        ),
                        "[2026-03-09T10:02:00-05:00] batch_done id=fw_a",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_live_view.py",
                    "--once",
                    "--no-alt-screen",
                    "--refresh-seconds",
                    "0.1",
                    "--warm-dir",
                    str(warm),
                    "--hot-dir",
                    str(hot),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--expected-parquet-files",
                    "10",
                    "--expected-bytes",
                    "100",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("Coverage: manifest_unique=2/10", proc.stdout)
            self.assertIn("from history", proc.stdout)
            coverage_line = next(
                line for line in proc.stdout.splitlines() if line.strip().startswith("Coverage:")
            )
            self.assertIn("eta=", coverage_line)
            self.assertNotIn("eta=unknown", coverage_line)

    def test_pipeline_live_view_reports_train_token_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            hot = root / "hot"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, hot, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            (sup_dir / "supervisor_20260309_120000.log").write_text(
                (
                    "[2026-03-09T12:00:01-05:00] "
                    "waiting_for_train_tokens have_tokens=123456 need_tokens=999999 "
                    "unique_inputs=51 overlap_inputs=0 overlap_manifests=0 sleep=60s\n"
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_live_view.py",
                    "--once",
                    "--no-alt-screen",
                    "--refresh-seconds",
                    "0.1",
                    "--warm-dir",
                    str(warm),
                    "--hot-dir",
                    str(hot),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--expected-parquet-files",
                    "510",
                    "--expected-bytes",
                    "1061360917731",
                    "--train-target-step",
                    "100000",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("Supervisor: gate=waiting_train_tokens 123456/999999", proc.stdout)

    def test_pipeline_live_view_reports_quality_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            hot = root / "hot"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, hot, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            (sup_dir / "eval_trend.tsv").write_text(
                "\n".join(
                    [
                        (
                            "run_tag\tstep\teval_rc\tpass_rate\tcheck_pass_rate\tavg_case_score\t"
                            "cases_passed\tcases_total\treport_json"
                        ),
                        "run1\t1000\t0\t0.20\t0.70\t0.65\t3\t15\treport1.json",
                        "run2\t2000\t0\t0.30\t0.75\t0.70\t5\t15\treport2.json",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sup_dir / "generation_trend.tsv").write_text(
                "\n".join(
                    [
                        (
                            "run_tag\tstep\tgeneration_rc\tpass_rate\tcheck_pass_rate\t"
                            "avg_case_score\tcases_passed\tcases_total\tregression_pass\t"
                            "baseline_report\treport_json"
                        ),
                        "run1\t1000\t0\t0.80\t0.85\t0.80\t4\t5\tTrue\tbase.json\tgen1.json",
                        "run2\t2000\t0\t1.00\t1.00\t1.00\t5\t5\tTrue\tgen1.json\tgen2.json",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_live_view.py",
                    "--once",
                    "--no-alt-screen",
                    "--refresh-seconds",
                    "0.1",
                    "--warm-dir",
                    str(warm),
                    "--hot-dir",
                    str(hot),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--expected-parquet-files",
                    "510",
                    "--expected-bytes",
                    "1061360917731",
                    "--train-target-step",
                    "100000",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("Quality:  heartbeat=improving", proc.stdout)
            self.assertIn("eval=improving", proc.stdout)
            self.assertIn("gen=improving", proc.stdout)

    def test_pipeline_live_view_auto_detects_train_target_step(self) -> None:
        if (
            subprocess.run(
                ["pgrep", "-af", r"llm\.cli train"],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        ):
            self.skipTest("trainer already running on host")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            hot = root / "hot"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, hot, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            (sup_dir / "supervisor_20260309_130904.log").write_text(
                (
                    "[2026-03-09T13:09:09-05:00] "
                    "train_launch manifests=11 step_now=138000 target_step=140000 "
                    "batch_size=12 grad_accum=2\n"
                ),
                encoding="utf-8",
            )
            (sup_dir / "train_138000_to_140000_20260309_130909.log").write_text(
                "step=139900 train_loss=3.12 lr=0.00003 tokens_seen=123 toks_per_sec=28000\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_live_view.py",
                    "--once",
                    "--no-alt-screen",
                    "--refresh-seconds",
                    "0.1",
                    "--warm-dir",
                    str(warm),
                    "--hot-dir",
                    str(hot),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("Training: step=139900/140000", proc.stdout)

    def test_pipeline_eta_report_auto_detects_train_target_step(self) -> None:
        if (
            subprocess.run(
                ["pgrep", "-af", r"llm\.cli train"],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        ):
            self.skipTest("trainer already running on host")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            (sup_dir / "supervisor_20260309_130904.log").write_text(
                (
                    "[2026-03-09T13:09:09-05:00] "
                    "train_launch manifests=11 step_now=138000 target_step=140000 "
                    "batch_size=12 grad_accum=2\n"
                ),
                encoding="utf-8",
            )
            (sup_dir / "train_138000_to_140000_20260309_130909.log").write_text(
                "step=139900 train_loss=3.12 lr=0.00003 tokens_seen=123 toks_per_sec=28000\n",
                encoding="utf-8",
            )

            out_json = root / "status.json"
            out_txt = root / "status.txt"
            state_json = root / "state.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_eta_report.py",
                    "--warm-dir",
                    str(warm),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--output-json",
                    str(out_json),
                    "--output-text",
                    str(out_txt),
                    "--state-file",
                    str(state_json),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["train_step"], 139900)
            self.assertEqual(payload["expected"]["train_target_step"], 140000)
            self.assertEqual(payload["remaining"]["train_steps"], 100)

    def test_pipeline_eta_report_accepts_once_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            out_json = root / "status.json"
            out_txt = root / "status.txt"
            state_json = root / "state.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_eta_report.py",
                    "--once",
                    "--warm-dir",
                    str(warm),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--output-json",
                    str(out_json),
                    "--output-text",
                    str(out_txt),
                    "--state-file",
                    str(state_json),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(out_json.exists())
            self.assertIn("status_written", proc.stdout)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("trainer_stall_seconds", payload["metrics"])
            self.assertIn("offload_eligible_batches", payload["metrics"])

    def test_pipeline_eta_report_uses_coverage_fallback_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            (shards / "batch_0001").mkdir(parents=True, exist_ok=True)
            (shards / "batch_0001" / "manifest.json").write_text(
                json.dumps(
                    {
                        "input_files": [
                            "000_00001.parquet",
                            "000_00002.parquet",
                        ]
                    }
                ),
                encoding="utf-8",
            )

            processed = stage_dir / "processed_parquet_files.txt"
            processed.write_text(
                "\n".join(
                    [
                        "000_00001.parquet",
                        "000_00002.parquet",
                        "000_00003.parquet",
                        "000_00004.parquet",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            out_json = root / "status.json"
            out_txt = root / "status.txt"
            state_json = root / "state.json"
            now = time.time()
            state_json.write_text(
                json.dumps(
                    {
                        "ts": now - 10.0,
                        "warm_bytes": 0,
                        "warm_parquet_count": 0,
                        "sharded_parquet_count": 1,
                        "manifest_count": 1,
                        "manifest_unique_input_files": 2,
                        "train_step": 0,
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_eta_report.py",
                    "--warm-dir",
                    str(warm),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--expected-parquet-files",
                    "10",
                    "--expected-bytes",
                    "100",
                    "--output-json",
                    str(out_json),
                    "--output-text",
                    str(out_txt),
                    "--state-file",
                    str(state_json),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            rate = payload["rates"]["manifest_unique_inputs_per_sec"]
            self.assertIsNotNone(rate)
            self.assertGreater(rate, 0)
            self.assertEqual(
                payload["rates"]["manifest_unique_inputs_rate_source"],
                "sharding_fallback_no_overlap",
            )
            self.assertNotEqual(payload["eta_human"]["manifest_unique_inputs"], "unknown")

    def test_pipeline_eta_report_counts_offloaded_manifests_in_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            (shards / "batch_0001").mkdir(parents=True, exist_ok=True)
            (shards / "batch_0001" / "manifest.offloaded.json").write_text(
                json.dumps({"input_files": ["000_00077.parquet"]}),
                encoding="utf-8",
            )

            out_json = root / "status.json"
            out_txt = root / "status.txt"
            state_json = root / "state.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_eta_report.py",
                    "--warm-dir",
                    str(warm),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--expected-parquet-files",
                    "10",
                    "--expected-bytes",
                    "100",
                    "--output-json",
                    str(out_json),
                    "--output-text",
                    str(out_txt),
                    "--state-file",
                    str(state_json),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["manifest_unique_input_files"], 1)

    def test_pipeline_live_view_uses_eta_status_rate_fallback(self) -> None:
        if (
            subprocess.run(
                ["pgrep", "-af", r"llm\.cli train"],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        ):
            self.skipTest("trainer already running on host")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            hot = root / "hot"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, hot, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            (sup_dir / "supervisor_20260309_130904.log").write_text(
                (
                    "[2026-03-09T13:09:09-05:00] "
                    "train_launch manifests=11 step_now=138000 target_step=140000 "
                    "batch_size=12 grad_accum=2\n"
                ),
                encoding="utf-8",
            )
            (sup_dir / "train_138000_to_140000_20260309_130909.log").write_text(
                "step=139900 train_loss=3.12 lr=0.00003 tokens_seen=123 toks_per_sec=28000\n",
                encoding="utf-8",
            )

            eta_status = root / "pipeline_status.json"
            eta_status.write_text(
                json.dumps(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "rates": {"train_steps_per_sec": 2.0},
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_live_view.py",
                    "--once",
                    "--no-alt-screen",
                    "--refresh-seconds",
                    "0.1",
                    "--warm-dir",
                    str(warm),
                    "--hot-dir",
                    str(hot),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--train-target-step",
                    "140000",
                    "--eta-status-file",
                    str(eta_status),
                    "--eta-status-max-age-seconds",
                    "300",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn(
                "Training: step=139900/140000 rate=2.000 step/s (from eta-report)",
                proc.stdout,
            )

    def test_pipeline_eta_report_pgrep_root_count_dedupes_children(self) -> None:
        marker = f"eta_root_count_{int(time.time() * 1_000_000)}"
        cmd = (
            f"exec -a {marker} bash -lc "
            f"'exec -a {marker} sleep 15 & wait'"
        )
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.8)
            module_path = Path(__file__).resolve().parents[1] / "scripts" / "pipeline_eta_report.py"
            spec = importlib.util.spec_from_file_location("pipeline_eta_report", module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            count = module._pgrep_root_count(marker)
            self.assertEqual(count, 1)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    def test_pipeline_live_view_alerts_when_stage_controller_missing(self) -> None:
        if (
            subprocess.run(
                ["pgrep", "-af", "fineweb_stage_shard_loop.sh|fineweb_stage_shard_watchdog.sh"],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        ):
            self.skipTest("stage controller already running on host")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            hot = root / "hot"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, hot, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/pipeline_live_view.py",
                    "--once",
                    "--no-alt-screen",
                    "--refresh-seconds",
                    "0.1",
                    "--warm-dir",
                    str(warm),
                    "--hot-dir",
                    str(hot),
                    "--shards-root",
                    str(shards),
                    "--stage-state-dir",
                    str(stage_dir),
                    "--supervisor-state-dir",
                    str(sup_dir),
                    "--manifest-stall-seconds",
                    "1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("ALERT: stage pipeline controller is not running", proc.stdout)

    def test_pipeline_live_view_alerts_when_manifest_stalled(self) -> None:
        if (
            subprocess.run(
                ["pgrep", "-af", "fineweb_stage_shard_loop.sh|fineweb_stage_shard_watchdog.sh"],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        ):
            self.skipTest("stage controller already running on host")
        if (
            subprocess.run(
                ["pgrep", "-af", "scripts/fineweb_parquet_to_shards.py"],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        ):
            self.skipTest("shard builders already running on host")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            hot = root / "hot"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, hot, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            # Force one stale manifest while no sharder is active.
            batch_dir = shards / "batch_0001"
            batch_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = batch_dir / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            old_ts = time.time() - 300
            os.utime(manifest_path, (old_ts, old_ts))

            dummy = subprocess.Popen(
                ["bash", "-lc", "exec -a fineweb_stage_shard_loop.sh sleep 15"],
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "scripts/pipeline_live_view.py",
                        "--once",
                        "--no-alt-screen",
                        "--refresh-seconds",
                        "0.1",
                        "--warm-dir",
                        str(warm),
                        "--hot-dir",
                        str(hot),
                        "--shards-root",
                        str(shards),
                        "--stage-state-dir",
                        str(stage_dir),
                        "--supervisor-state-dir",
                        str(sup_dir),
                        "--manifest-stall-seconds",
                        "1",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertIn("ALERT: manifest count stalled", proc.stdout)
            finally:
                dummy.terminate()
                dummy.wait(timeout=5)

    def test_pipeline_live_view_alerts_when_stage_loop_unmanaged(self) -> None:
        if (
            subprocess.run(
                ["pgrep", "-af", "fineweb_stage_shard_loop.sh|fineweb_stage_shard_watchdog.sh"],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        ):
            self.skipTest("stage controller already running on host")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = root / "warm"
            hot = root / "hot"
            shards = root / "shards"
            stage_dir = root / "stage"
            sup_dir = root / "supervisor"
            for path in [warm, hot, shards, stage_dir, sup_dir]:
                path.mkdir(parents=True, exist_ok=True)

            dummy = subprocess.Popen(
                ["bash", "-lc", "exec -a fineweb_stage_shard_loop.sh sleep 15"],
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "scripts/pipeline_live_view.py",
                        "--once",
                        "--no-alt-screen",
                        "--refresh-seconds",
                        "0.1",
                        "--warm-dir",
                        str(warm),
                        "--hot-dir",
                        str(hot),
                        "--shards-root",
                        str(shards),
                        "--stage-state-dir",
                        str(stage_dir),
                        "--supervisor-state-dir",
                        str(sup_dir),
                        "--expected-parquet-files",
                        "10",
                        "--expected-bytes",
                        "100",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertIn(
                    "ALERT: stage-loop running without watchdog auto-restart",
                    proc.stdout,
                )
            finally:
                dummy.terminate()
                dummy.wait(timeout=5)

    @unittest.skipIf(shutil.which("timeout") is None, "timeout is required")
    def test_train_supervisor_waits_on_train_token_gate(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        if not (repo_root / ".venv" / "bin" / "python").exists():
            self.skipTest(".venv/bin/python not available in this test environment")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = root / "shards"
            batch = shards / "batch_0001"
            output_dir = root / "out"
            state_dir = root / "state"
            batch.mkdir(parents=True, exist_ok=True)

            (batch / "manifest.json").write_text(
                json.dumps(
                    {
                        "input_files": ["000_00001.parquet"],
                        "train": {"total_tokens": 123, "shards": []},
                        "val": {"total_tokens": 0, "shards": []},
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    "timeout",
                    "3",
                    "bash",
                    "scripts/train_supervisor_rtx5070_350bt.sh",
                    "--shards-path",
                    str(shards),
                    "--output-dir",
                    str(output_dir),
                    "--state-dir",
                    str(state_dir),
                    "--poll-seconds",
                    "1",
                    "--min-manifests",
                    "1",
                    "--min-unique-input-files",
                    "0",
                    "--min-train-tokens",
                    "999",
                    "--no-auto-tune",
                    "--no-eval-after-chunk",
                    "--no-generation-gate",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 124, msg=proc.stderr)
            self.assertIn("waiting_for_train_tokens", proc.stdout)

    @unittest.skipIf(shutil.which("timeout") is None, "timeout is required")
    def test_train_supervisor_singleton_scope_is_state_dir(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        if not (repo_root / ".venv" / "bin" / "python").exists():
            self.skipTest(".venv/bin/python not available in this test environment")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = root / "shards"
            batch = shards / "batch_0001"
            output_a = root / "out_a"
            output_b = root / "out_b"
            state_a = root / "state_a"
            state_b = root / "state_b"
            batch.mkdir(parents=True, exist_ok=True)

            (batch / "manifest.json").write_text(
                json.dumps(
                    {
                        "input_files": ["000_00001.parquet"],
                        "train": {"total_tokens": 123, "shards": []},
                        "val": {"total_tokens": 0, "shards": []},
                    }
                ),
                encoding="utf-8",
            )

            proc_a = subprocess.Popen(
                [
                    "timeout",
                    "12",
                    "bash",
                    "scripts/train_supervisor_rtx5070_350bt.sh",
                    "--shards-path",
                    str(shards),
                    "--output-dir",
                    str(output_a),
                    "--state-dir",
                    str(state_a),
                    "--poll-seconds",
                    "1",
                    "--min-manifests",
                    "1",
                    "--min-unique-input-files",
                    "0",
                    "--min-train-tokens",
                    "999",
                    "--no-auto-tune",
                    "--no-eval-after-chunk",
                    "--no-generation-gate",
                ],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                time.sleep(1.0)
                proc_b = subprocess.run(
                    [
                        "timeout",
                        "3",
                        "bash",
                        "scripts/train_supervisor_rtx5070_350bt.sh",
                        "--shards-path",
                        str(shards),
                        "--output-dir",
                        str(output_b),
                        "--state-dir",
                        str(state_b),
                        "--poll-seconds",
                        "1",
                        "--min-manifests",
                        "1",
                        "--min-unique-input-files",
                        "0",
                        "--min-train-tokens",
                        "999",
                        "--no-auto-tune",
                        "--no-eval-after-chunk",
                        "--no-generation-gate",
                    ],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc_b.returncode, 124, msg=proc_b.stderr)
                self.assertIn("waiting_for_train_tokens", proc_b.stdout)
            finally:
                proc_a.terminate()
                try:
                    proc_a.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc_a.kill()
                    proc_a.wait(timeout=5)
                if proc_a.stdout is not None:
                    proc_a.stdout.close()
                if proc_a.stderr is not None:
                    proc_a.stderr.close()


@unittest.skipIf(torch is None, "torch is not installed")
class PackagingScriptTests(unittest.TestCase):
    def test_package_inference_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_path = root / "tokenizer.json"
            tokenizer = BPETokenizer.train_from_iterator(
                ["hello world\n"],
                vocab_size=256,
                min_frequency=1,
            )
            tokenizer.save(tokenizer_path)

            model_config = ModelConfig(
                vocab_size=tokenizer.vocab_size,
                max_seq_len=32,
                n_layers=1,
                n_heads=1,
                d_model=32,
                dropout=0.0,
            )
            model = GPTModel(model_config)
            ckpt = root / "ckpt.pt"
            torch.save(
                {
                    "step": 12,
                    "model_state": model.state_dict(),
                    "optimizer_state": {},
                    "model_config": model_config.to_dict(),
                    "tokenizer_path": str(tokenizer_path),
                    "tokenizer_hash": "abc",
                    "tokenizer_contract_hash": "def",
                },
                ckpt,
            )

            out_dir = root / "bundle"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/package_inference_bundle.py",
                    "--checkpoint",
                    str(ckpt),
                    "--output-dir",
                    str(out_dir),
                    "--model-id",
                    "local/test-model",
                    "--create-tar",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            manifest = out_dir / "bundle_manifest.json"
            self.assertTrue(manifest.exists())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_id"], "local/test-model")
            self.assertGreaterEqual(len(payload["files"]), 2)
            self.assertTrue((root / "bundle.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
