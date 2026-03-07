import tempfile
import unittest
from pathlib import Path

from llm.tokenizer import BasicCharTokenizer, load_tokenizer, tokenizer_fingerprint


class BasicCharTokenizerTests(unittest.TestCase):
    def test_round_trip_encode_decode(self) -> None:
        tokenizer = BasicCharTokenizer.train("hello world")
        ids = tokenizer.encode("hello")
        decoded = tokenizer.decode(ids)
        self.assertEqual(decoded, "hello")

    def test_unknown_token_maps_to_unk(self) -> None:
        tokenizer = BasicCharTokenizer.train("abc")
        ids = tokenizer.encode("abd")
        self.assertEqual(ids[-1], tokenizer.stoi["<unk>"])

    def test_save_and_load_vocab(self) -> None:
        tokenizer = BasicCharTokenizer.train("abc")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            tokenizer.save(path)
            loaded = BasicCharTokenizer.load(path)
        self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded.decode(loaded.encode("abc")), "abc")

    def test_load_tokenizer_detects_char_payload(self) -> None:
        tokenizer = BasicCharTokenizer.train("abc")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            tokenizer.save(path)
            loaded = load_tokenizer(path)
        self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded.decode(loaded.encode("abc")), "abc")

    def test_train_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p1 = d / "a.txt"
            p2 = d / "b.txt"
            p1.write_text("abca", encoding="utf-8")
            p2.write_text("zz\n", encoding="utf-8")
            tokenizer, stats = BasicCharTokenizer.train_from_files([p1, p2], chunk_size=2)

        self.assertEqual(stats["files_seen"], 2)
        self.assertEqual(stats["chars_read"], 7)
        self.assertEqual(stats["unique_chars"], 5)
        self.assertEqual(tokenizer.decode(tokenizer.encode("abz")), "abz")

    def test_train_from_files_respects_max_chars_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.txt"
            p.write_text("abcdef", encoding="utf-8")
            tokenizer, stats = BasicCharTokenizer.train_from_files(
                [p],
                max_chars_per_file=3,
                chunk_size=2,
            )
        self.assertEqual(stats["chars_read"], 3)
        self.assertEqual(stats["unique_chars"], 3)
        self.assertEqual(tokenizer.encode("d")[0], tokenizer.stoi["<unk>"])

    def test_tokenizer_fingerprint_changes_on_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "v1.json"
            p2 = Path(tmp) / "v2.json"
            BasicCharTokenizer.train("abc").save(p1)
            BasicCharTokenizer.train("abcd").save(p2)
            self.assertNotEqual(tokenizer_fingerprint(p1), tokenizer_fingerprint(p2))


if __name__ == "__main__":
    unittest.main()
