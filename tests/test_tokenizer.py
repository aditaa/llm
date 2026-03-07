import tempfile
import unittest
from pathlib import Path

from llm.tokenizer import BPETokenizer, load_tokenizer, tokenizer_fingerprint


class BPETokenizerTests(unittest.TestCase):
    def test_round_trip_encode_decode(self) -> None:
        tokenizer = BPETokenizer.train_from_iterator(
            ["hello world\nhello tokenizer\n"],
            vocab_size=256,
            min_frequency=1,
        )
        ids = tokenizer.encode("hello")
        decoded = tokenizer.decode(ids)
        self.assertIn("hello", decoded)

    def test_save_and_load_tokenizer(self) -> None:
        tokenizer = BPETokenizer.train_from_iterator(
            ["alpha beta gamma\n"],
            vocab_size=256,
            min_frequency=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            tokenizer.save(path)
            loaded = load_tokenizer(path)
        self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)
        self.assertIn("alpha", loaded.decode(loaded.encode("alpha")))

    def test_train_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p1 = d / "a.txt"
            p2 = d / "b.txt"
            p1.write_text("abca", encoding="utf-8")
            p2.write_text("zz\n", encoding="utf-8")
            tokenizer, stats = BPETokenizer.train_from_files(
                [p1, p2],
                vocab_size=256,
                min_frequency=1,
                chunk_size=2,
            )

        self.assertEqual(stats["files_seen"], 2)
        self.assertEqual(stats["chars_read"], 7)
        self.assertGreaterEqual(tokenizer.vocab_size, 256)
        self.assertIn("ab", tokenizer.decode(tokenizer.encode("abz")))

    def test_train_from_files_respects_max_chars_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.txt"
            p.write_text("abcdef", encoding="utf-8")
            tokenizer, stats = BPETokenizer.train_from_files(
                [p],
                vocab_size=256,
                min_frequency=1,
                max_chars_per_file=3,
                chunk_size=2,
            )
        self.assertEqual(stats["chars_read"], 3)
        self.assertGreaterEqual(tokenizer.vocab_size, 256)

    def test_tokenizer_fingerprint_changes_on_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "v1.json"
            p2 = Path(tmp) / "v2.json"
            BPETokenizer.train_from_iterator(["abc"], vocab_size=280, min_frequency=1).save(p1)
            BPETokenizer.train_from_iterator(["abcd"], vocab_size=320, min_frequency=1).save(p2)
            self.assertNotEqual(tokenizer_fingerprint(p1), tokenizer_fingerprint(p2))


if __name__ == "__main__":
    unittest.main()
