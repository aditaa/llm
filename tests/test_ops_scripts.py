import json
import subprocess
import sys
import tempfile
import unittest
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
