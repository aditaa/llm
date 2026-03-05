import json
import tempfile
import unittest
from array import array
from pathlib import Path

from llm.integrity import verify_shards
from llm.sharding import ShardConfig, shard_corpus
from llm.tokenizer import BasicCharTokenizer


def _build_sample_dataset(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "vocab.json"
    output_dir = tmp_path / "shards"

    corpus_text = "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\neta\ntheta\n"
    corpus_path.write_text(corpus_text, encoding="utf-8")
    tokenizer = BasicCharTokenizer.train(corpus_text)
    tokenizer.save(tokenizer_path)

    shard_corpus(
        ShardConfig(
            input_path=corpus_path,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            shard_size_tokens=8,
            val_ratio=0.1,
            seed=7,
        )
    )
    return output_dir / "manifest.json"


class IntegrityTests(unittest.TestCase):
    def test_verify_shards_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _build_sample_dataset(Path(tmp))
            results = verify_shards(manifest_path)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["ok"])
            self.assertGreater(results[0]["files_checked"], 0)

    def test_detect_shard_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _build_sample_dataset(Path(tmp))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shard_name = manifest["train"]["shards"][0]["path"]
            shard_path = manifest_path.parent / shard_name

            with shard_path.open("ab") as handle:
                handle.write(b"\x00")

            results = verify_shards(manifest_path)
            self.assertFalse(results[0]["ok"])
            self.assertTrue(any("size_mismatch" in err for err in results[0]["errors"]))

    def test_detect_token_out_of_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _build_sample_dataset(Path(tmp))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            split = "train" if manifest["train"]["shards"] else "val"
            shard_name = manifest[split]["shards"][0]["path"]
            shard_path = manifest_path.parent / shard_name

            array_type = "H" if manifest["token_dtype"] == "uint16" else "I"
            tokens = array(array_type)
            with shard_path.open("rb") as handle:
                tokens.fromfile(handle, int(manifest[split]["shards"][0]["tokens"]))

            tokens[0] = int(manifest["tokenizer_vocab_size"]) + 10
            with shard_path.open("wb") as handle:
                tokens.tofile(handle)

            results = verify_shards(manifest_path)
            self.assertFalse(results[0]["ok"])
            self.assertTrue(any("token_out_of_range" in err for err in results[0]["errors"]))

    def test_source_check_strict_missing_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = _build_sample_dataset(tmp_path)
            raw_zim_dir = tmp_path / "raw_zim"
            raw_zim_dir.mkdir(parents=True, exist_ok=True)

            results = verify_shards(
                manifest_path,
                check_token_ranges=False,
                raw_zim_dir=raw_zim_dir,
                strict_source=True,
            )
            self.assertFalse(results[0]["ok"])
            self.assertTrue(any("source_zim_missing" in err for err in results[0]["errors"]))

    def test_source_check_missing_warns_when_not_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = _build_sample_dataset(tmp_path)
            raw_zim_dir = tmp_path / "raw_zim"
            raw_zim_dir.mkdir(parents=True, exist_ok=True)

            results = verify_shards(
                manifest_path,
                check_token_ranges=False,
                raw_zim_dir=raw_zim_dir,
                strict_source=False,
            )
            self.assertTrue(results[0]["ok"])
            self.assertTrue(any("source_zim_missing" in w for w in results[0]["warnings"]))


if __name__ == "__main__":
    unittest.main()
