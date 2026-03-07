import tempfile
import unittest
from pathlib import Path

from llm.sharding import ShardConfig, iter_corpus_files, shard_corpora_batch, shard_corpus
from llm.tokenizer import BPETokenizer


class ShardingTests(unittest.TestCase):
    def test_shard_corpus_writes_manifest_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = tmp_path / "corpus.txt"
            tokenizer_path = tmp_path / "vocab.json"
            output_dir = tmp_path / "shards"

            corpus_text = "alpha\nbeta\ngamma\ndelta\nepsilon\n"
            corpus_path.write_text(corpus_text, encoding="utf-8")
            tokenizer = BPETokenizer.train_from_iterator(
                [corpus_text],
                vocab_size=256,
                min_frequency=1,
            )
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
            self.assertIn("tokenizer_hash", manifest)
            self.assertIn("tokenizer_contract_hash", manifest)
            self.assertIn("tokenizer_contract", manifest)
            self.assertGreater(manifest["train"]["total_tokens"], 0)
            self.assertGreaterEqual(manifest["val"]["total_tokens"], 0)

            total_sharded_tokens = (
                manifest["train"]["total_tokens"] + manifest["val"]["total_tokens"]
            )
            expected_tokens = sum(
                len(tokenizer.encode(line)) for line in corpus_text.splitlines(keepends=True)
            )
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
            tokenizer = BPETokenizer.train_from_iterator(
                [corpus_text],
                vocab_size=256,
                min_frequency=1,
            )
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
            expected_tokens = len(tokenizer.encode("a\n")) + len(tokenizer.encode("b\n"))
            self.assertEqual(sharded_tokens, expected_tokens)

    def test_iter_corpus_files_filters_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "a.txt").write_text("a", encoding="utf-8")
            (tmp_path / "b.txt").write_text("b", encoding="utf-8")
            (tmp_path / "b.paths.txt").write_text("p", encoding="utf-8")
            (tmp_path / "c.md").write_text("x", encoding="utf-8")

            files = iter_corpus_files(
                input_dir=tmp_path,
                pattern="*.txt",
                exclude_patterns=["*.paths.txt"],
                include_stems={"a", "b"},
                limit_files=1,
            )
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "a.txt")

    def test_shard_corpora_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "in"
            output_root = tmp_path / "out"
            tokenizer_path = tmp_path / "vocab.json"
            input_dir.mkdir(parents=True, exist_ok=True)

            (input_dir / "first.txt").write_text("alpha\nbeta\n", encoding="utf-8")
            (input_dir / "second.txt").write_text("gamma\ndelta\n", encoding="utf-8")
            tokenizer = BPETokenizer.train_from_iterator(
                ["alpha\nbeta\ngamma\ndelta\n"],
                vocab_size=256,
                min_frequency=1,
            )
            tokenizer.save(tokenizer_path)

            files = iter_corpus_files(input_dir=input_dir)
            results = shard_corpora_batch(
                input_files=files,
                tokenizer_path=tokenizer_path,
                output_root=output_root,
                shard_size_tokens=16,
                val_ratio=0.1,
                seed=42,
                max_lines=0,
                skip_existing=True,
            )
            self.assertEqual(len(results), 2)
            self.assertTrue((output_root / "first" / "manifest.json").exists())
            self.assertTrue((output_root / "second" / "manifest.json").exists())
            self.assertTrue(all(row["status"] == "ok" for row in results))


if __name__ == "__main__":
    unittest.main()
