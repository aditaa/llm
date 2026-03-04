from pathlib import Path
import tempfile
import unittest

from llm.tokenizer import BasicCharTokenizer


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


if __name__ == "__main__":
    unittest.main()
