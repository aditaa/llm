import json
import tempfile
import unittest
from array import array
from pathlib import Path

try:
    import torch

    from llm.train import ShardBatchSampler, collect_shard_training_info
except ModuleNotFoundError:
    torch = None
    ShardBatchSampler = None
    collect_shard_training_info = None


def _write_tokenizer(path: Path, stoi: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stoi": stoi}, indent=2), encoding="utf-8")


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
    def test_collect_shard_training_info_rejects_mismatched_tokenizers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds1 = root / "d1"
            ds2 = root / "d2"
            tok1 = root / "tok1.json"
            tok2 = root / "tok2.json"

            _write_tokenizer(tok1, {"<unk>": 0, "<bos>": 1, "<eos>": 2, "a": 3})
            _write_tokenizer(tok2, {"<unk>": 0, "<bos>": 1, "<eos>": 2, "b": 3})

            _write_shard(ds1 / "train_000000.bin", list(range(40)))
            _write_shard(ds1 / "val_000000.bin", list(range(20)))
            _write_shard(ds2 / "train_000000.bin", list(range(40)))
            _write_shard(ds2 / "val_000000.bin", list(range(20)))

            _write_manifest(
                ds1 / "manifest.json",
                tokenizer_path=tok1,
                train_shard="train_000000.bin",
                val_shard="val_000000.bin",
            )
            _write_manifest(
                ds2 / "manifest.json",
                tokenizer_path=tok2,
                train_shard="train_000000.bin",
                val_shard="val_000000.bin",
            )

            with self.assertRaises(ValueError):
                collect_shard_training_info(root)

    def test_collect_shard_training_info_and_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = root / "dataset"
            tok = root / "tok.json"

            _write_tokenizer(tok, {"<unk>": 0, "<bos>": 1, "<eos>": 2, "a": 3, "b": 4, "c": 5})
            _write_shard(ds / "train_000000.bin", list(range(100)))
            _write_shard(ds / "val_000000.bin", list(range(40)))
            _write_manifest(
                ds / "manifest.json",
                tokenizer_path=tok,
                train_shard="train_000000.bin",
                val_shard="val_000000.bin",
                vocab_size=6,
            )

            info = collect_shard_training_info(ds)
            self.assertEqual(info.vocab_size, 6)
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


if __name__ == "__main__":
    unittest.main()
