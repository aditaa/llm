import tempfile
import unittest
from pathlib import Path

from llm.sharding import ShardConfig, shard_corpus
from llm.tokenizer import BasicCharTokenizer


class ShardingTests(unittest.TestCase):
    def test_shard_corpus_writes_manifest_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = tmp_path / "corpus.txt"
            tokenizer_path = tmp_path / "vocab.json"
            output_dir = tmp_path / "shards"

            corpus_text = "alpha\nbeta\ngamma\ndelta\nepsilon\n"
            corpus_path.write_text(corpus_text, encoding="utf-8")
            tokenizer = BasicCharTokenizer.train(corpus_text)
            tokenizer.save(tokenizer_path)

            manifest = shard_corpus(
                ShardConfig(
                    input_path=corpus_path,
                    tokenizer_path=tokenizer_path,
                    output_dir=output_dir,
                    shard_size_tokens=8,
                    val_ratio=0.2,
                    seed=7,
                )
            )

            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertIn(manifest["token_dtype"], {"uint16", "uint32"})
            self.assertGreater(manifest["train"]["total_tokens"], 0)
            self.assertGreaterEqual(manifest["val"]["total_tokens"], 0)

            total_sharded_tokens = (
                manifest["train"]["total_tokens"] + manifest["val"]["total_tokens"]
            )
            expected_tokens = len(tokenizer.encode(corpus_text))
            self.assertEqual(total_sharded_tokens, expected_tokens)

            for shard in manifest["train"]["shards"] + manifest["val"]["shards"]:
                self.assertTrue((output_dir / shard["path"]).exists())

    def test_max_lines_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = tmp_path / "corpus.txt"
            tokenizer_path = tmp_path / "vocab.json"
            output_dir = tmp_path / "shards"

            corpus_text = "a\nb\nc\nd\ne\n"
            corpus_path.write_text(corpus_text, encoding="utf-8")
            tokenizer = BasicCharTokenizer.train(corpus_text)
            tokenizer.save(tokenizer_path)

            manifest = shard_corpus(
                ShardConfig(
                    input_path=corpus_path,
                    tokenizer_path=tokenizer_path,
                    output_dir=output_dir,
                    shard_size_tokens=100,
                    max_lines=2,
                )
            )
            self.assertEqual(manifest["line_count"], 2)
            sharded_tokens = manifest["train"]["total_tokens"] + manifest["val"]["total_tokens"]
            self.assertEqual(sharded_tokens, len(tokenizer.encode("a\nb\n")))


if __name__ == "__main__":
    unittest.main()
