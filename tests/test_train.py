import json
import tempfile
import unittest
from array import array
from pathlib import Path

try:
    import torch

    from llm.tokenizer import BPETokenizer, tokenizer_contract_fingerprint, tokenizer_fingerprint
    from llm.train import (
        ShardBatchSampler,
        TrainConfig,
        _apply_resume_context_policy,
        _compute_keep_steps,
        _init_ema_state,
        _lr_for_step,
        _prune_old_checkpoints,
        _resolve_amp_mode,
        _update_ema_state,
        collect_shard_training_info,
        run_training,
    )
except ModuleNotFoundError:
    torch = None
    BPETokenizer = None
    tokenizer_contract_fingerprint = None
    tokenizer_fingerprint = None
    TrainConfig = None
    ShardBatchSampler = None
    _apply_resume_context_policy = None
    _compute_keep_steps = None
    _init_ema_state = None
    _lr_for_step = None
    _prune_old_checkpoints = None
    _resolve_amp_mode = None
    _update_ema_state = None
    collect_shard_training_info = None
    run_training = None


def _write_tokenizer(path: Path, text: str, vocab_size: int = 256) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = BPETokenizer.train_from_iterator(
        [text],
        vocab_size=vocab_size,
        min_frequency=1,
    )
    tokenizer.save(path)
    return int(tokenizer.vocab_size)


def _write_shard(path: Path, token_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        array("H", token_ids).tofile(handle)


def _write_manifest(
    path: Path,
    *,
    tokenizer_path: Path,
    train_shard: str,
    val_shard: str,
    vocab_size: int = 8,
) -> None:
    manifest = {
        "input_path": "x.txt",
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_hash": tokenizer_fingerprint(tokenizer_path),
        "tokenizer_contract_hash": tokenizer_contract_fingerprint(tokenizer_path),
        "tokenizer_vocab_size": vocab_size,
        "token_dtype": "uint16",
        "shard_size_tokens": 1024,
        "val_ratio": 0.01,
        "seed": 42,
        "max_lines": 0,
        "line_count": 10,
        "train": {"total_tokens": 32, "shards": [{"path": train_shard, "tokens": 32}]},
        "val": {"total_tokens": 16, "shards": [{"path": val_shard, "tokens": 16}]},
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@unittest.skipIf(torch is None, "torch is not installed")
class TrainDataTests(unittest.TestCase):
    def test_resolve_amp_mode_cpu_defaults_to_fp32(self) -> None:
        enabled, dtype, use_scaler, effective = _resolve_amp_mode(torch.device("cpu"), "auto")
        self.assertFalse(enabled)
        self.assertIsNone(dtype)
        self.assertFalse(use_scaler)
        self.assertEqual(effective, "fp32")

    def test_resolve_amp_mode_rejects_invalid_precision(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_amp_mode(torch.device("cpu"), "bad")

    def test_resolve_amp_mode_rejects_half_precision_on_cpu(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_amp_mode(torch.device("cpu"), "fp16")

    def test_lr_schedule_constant(self) -> None:
        lr = _lr_for_step(
            step=10,
            max_steps=100,
            base_lr=1e-3,
            schedule="constant",
            warmup_steps=0,
            min_ratio=0.1,
        )
        self.assertAlmostEqual(lr, 1e-3, places=12)

    def test_lr_schedule_cosine_with_warmup(self) -> None:
        lr_warm = _lr_for_step(
            step=5,
            max_steps=100,
            base_lr=1e-3,
            schedule="cosine",
            warmup_steps=10,
            min_ratio=0.1,
        )
        self.assertAlmostEqual(lr_warm, 5e-4, places=12)

        lr_late = _lr_for_step(
            step=100,
            max_steps=100,
            base_lr=1e-3,
            schedule="cosine",
            warmup_steps=10,
            min_ratio=0.1,
        )
        self.assertAlmostEqual(lr_late, 1e-4, places=12)

    def test_collect_shard_training_info_rejects_mismatched_tokenizers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds1 = root / "d1"
            ds2 = root / "d2"
            tok1 = root / "tok1.json"
            tok2 = root / "tok2.json"

            vocab1 = _write_tokenizer(tok1, "aaaa bbbb cccc", vocab_size=260)
            vocab2 = _write_tokenizer(tok2, "zzzz yyyy xxxx", vocab_size=320)

            _write_shard(ds1 / "train_000000.bin", list(range(40)))
            _write_shard(ds1 / "val_000000.bin", list(range(20)))
            _write_shard(ds2 / "train_000000.bin", list(range(40)))
            _write_shard(ds2 / "val_000000.bin", list(range(20)))

            _write_manifest(
                ds1 / "manifest.json",
                tokenizer_path=tok1,
                train_shard="train_000000.bin",
                val_shard="val_000000.bin",
                vocab_size=vocab1,
            )
            _write_manifest(
                ds2 / "manifest.json",
                tokenizer_path=tok2,
                train_shard="train_000000.bin",
                val_shard="val_000000.bin",
                vocab_size=vocab2,
            )

            with self.assertRaises(ValueError):
                collect_shard_training_info(root)

    def test_collect_shard_training_info_and_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "dataset"
            tok = root / "tok.json"

            vocab_size = _write_tokenizer(tok, "a b c d e f g")
            _write_shard(ds / "train_000000.bin", list(range(100)))
            _write_shard(ds / "val_000000.bin", list(range(40)))
            _write_manifest(
                ds / "manifest.json",
                tokenizer_path=tok,
                train_shard="train_000000.bin",
                val_shard="val_000000.bin",
                vocab_size=vocab_size,
            )

            info = collect_shard_training_info(ds)
            self.assertEqual(info.vocab_size, vocab_size)
            self.assertEqual(len(info.train_shards), 1)
            self.assertEqual(len(info.val_shards), 1)

            sampler = ShardBatchSampler(
                shard_paths=info.train_shards,
                token_dtype=info.token_dtype,
                context_length=8,
                seed=7,
                device=torch.device("cpu"),
            )
            xb, yb = sampler.sample_batch(batch_size=4)
            self.assertEqual(tuple(xb.shape), (4, 8))
            self.assertEqual(tuple(yb.shape), (4, 8))
            self.assertEqual(xb.dtype, torch.long)
            self.assertTrue(torch.all(xb[:, 1:] == yb[:, :-1]))

    def test_sampler_limits_open_shard_cache_and_writes_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "dataset"
            tok = root / "tok.json"

            vocab_size = _write_tokenizer(tok, "a b c d e f g h i")
            for idx in range(6):
                _write_shard(ds / f"train_{idx:06d}.bin", list(range(80)))
            _write_shard(ds / "val_000000.bin", list(range(40)))
            manifest = {
                "input_path": "x.txt",
                "tokenizer_path": str(tok),
                "tokenizer_hash": tokenizer_fingerprint(tok),
                "tokenizer_contract_hash": tokenizer_contract_fingerprint(tok),
                "tokenizer_vocab_size": vocab_size,
                "token_dtype": "uint16",
                "shard_size_tokens": 1024,
                "val_ratio": 0.01,
                "seed": 42,
                "max_lines": 0,
                "line_count": 10,
                "train": {
                    "total_tokens": 480,
                    "shards": [
                        {"path": f"train_{idx:06d}.bin", "tokens": 80}
                        for idx in range(6)
                    ],
                },
                "val": {"total_tokens": 40, "shards": [{"path": "val_000000.bin", "tokens": 40}]},
            }
            (ds / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            info = collect_shard_training_info(ds)
            sampler = ShardBatchSampler(
                shard_paths=info.train_shards,
                token_dtype=info.token_dtype,
                context_length=8,
                seed=7,
                device=torch.device("cpu"),
                max_open_shards=2,
            )
            for _ in range(20):
                sampler.sample_batch(batch_size=8)
                self.assertLessEqual(len(sampler._array_cache), 2)

            trace_path = root / "sample_trace.json"
            count = sampler.write_sample_trace(trace_path)
            self.assertGreater(count, 0)
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertIn("sampled_shards", payload)
            self.assertTrue(payload["sampled_shards"])

    def test_ema_state_update(self) -> None:
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema_state = _init_ema_state(model)
        with torch.no_grad():
            model.weight.fill_(3.0)
        _update_ema_state(ema_state, model, decay=0.5)
        expected = torch.full_like(model.weight, 2.0)
        self.assertTrue(torch.allclose(ema_state["weight"], expected))

    def test_compute_keep_steps(self) -> None:
        keep = _compute_keep_steps(
            all_steps=[100, 200, 300, 400, 500],
            current_step=500,
            keep_last=2,
            keep_every=300,
        )
        self.assertEqual(keep, {300, 400, 500})

    def test_prune_old_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in [100, 200, 300, 400, 500]:
                (root / f"ckpt_step_{step:07d}.pt").write_bytes(b"x")
                (root / f"ckpt_step_{step:07d}.safetensors").write_bytes(b"x")
                (root / f"ckpt_step_{step:07d}_ema.safetensors").write_bytes(b"x")

            _prune_old_checkpoints(
                output_dir=root,
                current_step=500,
                keep_last=2,
                keep_every=300,
            )

            remaining = sorted(p.name for p in root.glob("ckpt_step_*.pt"))
            self.assertEqual(
                remaining,
                [
                    "ckpt_step_0000300.pt",
                    "ckpt_step_0000400.pt",
                    "ckpt_step_0000500.pt",
                ],
            )
            self.assertFalse((root / "ckpt_step_0000100.safetensors").exists())
            self.assertFalse((root / "ckpt_step_0000100_ema.safetensors").exists())

    def test_apply_resume_context_policy(self) -> None:
        from llm.model import ModelConfig

        cfg = ModelConfig(vocab_size=100, max_seq_len=512, n_layers=1, n_heads=1, d_model=64)
        extended = _apply_resume_context_policy(
            model_config=cfg,
            requested_context_length=1024,
            allow_extension=True,
        )
        self.assertTrue(extended)
        self.assertEqual(cfg.max_seq_len, 1024)

        with self.assertRaises(ValueError):
            _apply_resume_context_policy(
                model_config=cfg,
                requested_context_length=256,
                allow_extension=False,
            )

    def test_run_training_writes_sampled_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "dataset"
            out = root / "out"
            tok = root / "tok.json"
            trace = root / "sampled_trace.json"

            vocab_size = _write_tokenizer(tok, "simple tiny corpus for training")
            _write_shard(ds / "train_000000.bin", list(range(200)))
            _write_shard(ds / "val_000000.bin", list(range(120)))
            _write_manifest(
                ds / "manifest.json",
                tokenizer_path=tok,
                train_shard="train_000000.bin",
                val_shard="val_000000.bin",
                vocab_size=vocab_size,
            )

            result = run_training(
                TrainConfig(
                    shards_path=ds,
                    output_dir=out,
                    max_steps=1,
                    batch_size=2,
                    context_length=16,
                    grad_accum_steps=1,
                    eval_interval=1,
                    eval_steps=1,
                    log_interval=1,
                    device="cpu",
                    n_layers=1,
                    n_heads=1,
                    d_model=32,
                    sampled_shards_trace=trace,
                    sampled_shards_trace_min_rows=1,
                )
            )

            self.assertEqual(result["max_steps"], 1)
            self.assertTrue(trace.exists())
            payload = json.loads(trace.read_text(encoding="utf-8"))
            self.assertIn("sampled_shards", payload)
            self.assertTrue(payload["sampled_shards"])


if __name__ == "__main__":
    unittest.main()
