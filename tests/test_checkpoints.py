import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from llm.checkpoints import AverageCheckpointsConfig, run_checkpoint_average
except ModuleNotFoundError:
    torch = None
    AverageCheckpointsConfig = None
    run_checkpoint_average = None


@unittest.skipIf(torch is None, "torch is not installed")
class CheckpointAveragingTests(unittest.TestCase):
    def test_run_checkpoint_average_model_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_a = root / "a.pt"
            ckpt_b = root / "b.pt"
            out = root / "avg.pt"

            torch.save(
                {
                    "step": 10,
                    "model_state": {
                        "w": torch.tensor([1.0, 3.0], dtype=torch.float32),
                        "b": torch.tensor([2], dtype=torch.int64),
                    },
                    "optimizer_state": {"x": 1},
                    "model_config": {"dummy": 1},
                    "tokenizer_path": "tok.json",
                },
                ckpt_a,
            )
            torch.save(
                {
                    "step": 20,
                    "model_state": {
                        "w": torch.tensor([5.0, 7.0], dtype=torch.float32),
                        "b": torch.tensor([2], dtype=torch.int64),
                    },
                    "optimizer_state": {"x": 2},
                    "model_config": {"dummy": 1},
                    "tokenizer_path": "tok.json",
                },
                ckpt_b,
            )

            result = run_checkpoint_average(
                AverageCheckpointsConfig(
                    checkpoint_paths=[ckpt_a, ckpt_b],
                    output_path=out,
                    state_key="model_state",
                    export_safetensors=False,
                )
            )

            self.assertEqual(result["averaged_count"], 2)
            self.assertTrue(out.exists())

            merged = torch.load(out, map_location="cpu")
            self.assertEqual(merged["optimizer_state"], {})
            self.assertEqual(merged["averaged_state_key"], "model_state")
            self.assertTrue(torch.equal(merged["model_state"]["w"], torch.tensor([3.0, 5.0])))
            self.assertTrue(torch.equal(merged["model_state"]["b"], torch.tensor([2])))

    def test_run_checkpoint_average_ema_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_a = root / "a.pt"
            ckpt_b = root / "b.pt"
            out = root / "avg.pt"

            torch.save(
                {
                    "step": 10,
                    "model_state": {"w": torch.tensor([10.0], dtype=torch.float32)},
                    "ema_state": {"w": torch.tensor([2.0], dtype=torch.float32)},
                    "model_config": {"dummy": 1},
                    "tokenizer_path": "tok.json",
                },
                ckpt_a,
            )
            torch.save(
                {
                    "step": 20,
                    "model_state": {"w": torch.tensor([20.0], dtype=torch.float32)},
                    "ema_state": {"w": torch.tensor([6.0], dtype=torch.float32)},
                    "model_config": {"dummy": 1},
                    "tokenizer_path": "tok.json",
                },
                ckpt_b,
            )

            run_checkpoint_average(
                AverageCheckpointsConfig(
                    checkpoint_paths=[ckpt_a, ckpt_b],
                    output_path=out,
                    state_key="ema_state",
                    export_safetensors=False,
                )
            )

            merged = torch.load(out, map_location="cpu")
            self.assertTrue(torch.equal(merged["model_state"]["w"], torch.tensor([4.0])))
            self.assertEqual(merged["averaged_state_key"], "ema_state")


if __name__ == "__main__":
    unittest.main()
