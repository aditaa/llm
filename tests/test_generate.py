import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from llm.generate import GenerateConfig, run_generation
    from llm.model import GPTModel, ModelConfig
    from llm.tokenizer import BasicCharTokenizer
except ModuleNotFoundError:
    torch = None
    GenerateConfig = None
    run_generation = None
    GPTModel = None
    ModelConfig = None
    BasicCharTokenizer = None


@unittest.skipIf(torch is None, "torch is not installed")
class GenerateTests(unittest.TestCase):
    def test_run_generation_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokenizer_path = tmp_path / "vocab.json"
            tokenizer = BasicCharTokenizer.train("hello world\n")
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
            checkpoint_path = tmp_path / "ckpt.pt"
            torch.save(
                {
                    "step": 0,
                    "model_state": model.state_dict(),
                    "optimizer_state": {},
                    "model_config": model_config.to_dict(),
                    "tokenizer_path": str(tokenizer_path),
                },
                checkpoint_path,
            )

            result = run_generation(
                GenerateConfig(
                    checkpoint_path=checkpoint_path,
                    prompt="he",
                    max_new_tokens=8,
                    temperature=1.0,
                    top_k=0,
                    device="cpu",
                    seed=7,
                )
            )

        self.assertEqual(result["checkpoint_path"], str(checkpoint_path))
        self.assertEqual(result["tokenizer_path"], str(tokenizer_path))
        self.assertEqual(result["device"], "cpu")
        self.assertGreaterEqual(len(result["output_text"]), 2)
        self.assertGreaterEqual(result["token_count"], 2)

    def test_invalid_generate_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokenizer_path = tmp_path / "vocab.json"
            tokenizer = BasicCharTokenizer.train("abc")
            tokenizer.save(tokenizer_path)

            model_config = ModelConfig(
                vocab_size=tokenizer.vocab_size,
                max_seq_len=8,
                n_layers=1,
                n_heads=1,
                d_model=8,
                dropout=0.0,
            )
            model = GPTModel(model_config)
            checkpoint_path = tmp_path / "ckpt.pt"
            torch.save(
                {
                    "step": 0,
                    "model_state": model.state_dict(),
                    "optimizer_state": {},
                    "model_config": model_config.to_dict(),
                    "tokenizer_path": str(tokenizer_path),
                },
                checkpoint_path,
            )

            with self.assertRaises(ValueError):
                run_generation(
                    GenerateConfig(
                        checkpoint_path=checkpoint_path,
                        prompt="a",
                        max_new_tokens=1,
                        temperature=0.0,
                        top_k=0,
                        device="cpu",
                    )
                )


if __name__ == "__main__":
    unittest.main()
