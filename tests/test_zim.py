import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm.zim import _blob_to_bytes, _iter_paths_from_search, html_to_text, normalize_whitespace


class ZimUtilsTests(unittest.TestCase):
    def test_html_to_text_strips_scripts_and_collapses_spaces(self) -> None:
        html = """
        <html><body>
          <h1>Title</h1>
          <script>var x = 1;</script>
          <p>Hello   world</p>
          <style>body { color: red; }</style>
        </body></html>
        """
        text = html_to_text(html)
        self.assertIn("Title", text)
        self.assertIn("Hello world", text)
        self.assertNotIn("var x = 1", text)
        self.assertNotIn("color: red", text)

    def test_normalize_whitespace(self) -> None:
        self.assertEqual(normalize_whitespace("a   b\n\nc\t d"), "a b c d")

    def test_blob_to_bytes(self) -> None:
        self.assertEqual(_blob_to_bytes(b"abc"), b"abc")
        self.assertEqual(_blob_to_bytes(bytearray(b"abc")), b"abc")
        with self.assertRaises(TypeError):
            _blob_to_bytes(object())

    def test_iter_paths_from_search_uses_fulltext_results(self) -> None:
        class FakeQuery:
            def set_query(self, _query: str) -> "FakeQuery":
                return self

        class FakeSearch:
            def getResults(self, offset: int, limit: int) -> list[str]:
                data = ["a", "b", "a"]
                return data[offset : offset + limit]

        class FakeSearcher:
            def __init__(self, _archive: object) -> None:
                pass

            def search(self, _query: FakeQuery) -> FakeSearch:
                return FakeSearch()

        class FakeSuggestionSearch:
            def getResults(self, _offset: int, _limit: int) -> list[str]:
                return []

        class FakeSuggestionSearcher:
            def __init__(self, _archive: object) -> None:
                pass

            def suggest(self, _query: str) -> FakeSuggestionSearch:
                return FakeSuggestionSearch()

        fake_libzim = SimpleNamespace(
            Query=FakeQuery,
            Searcher=FakeSearcher,
            SuggestionSearcher=FakeSuggestionSearcher,
        )
        with patch.dict("sys.modules", {"libzim": fake_libzim}):
            paths = list(_iter_paths_from_search(archive=object(), query="x", batch_size=2))
        self.assertEqual(paths, ["a", "b"])

    def test_iter_paths_from_search_falls_back_to_suggestions(self) -> None:
        class FakeQuery:
            def set_query(self, _query: str) -> "FakeQuery":
                return self

        class EmptySearch:
            def getResults(self, _offset: int, _limit: int) -> list[str]:
                return []

        class FakeSearcher:
            def __init__(self, _archive: object) -> None:
                pass

            def search(self, _query: FakeQuery) -> EmptySearch:
                return EmptySearch()

        class FakeSuggestionSearch:
            def getResults(self, offset: int, limit: int) -> list[str]:
                data = ["p1", "p2", "p2"]
                return data[offset : offset + limit]

        class FakeSuggestionSearcher:
            def __init__(self, _archive: object) -> None:
                pass

            def suggest(self, _query: str) -> FakeSuggestionSearch:
                return FakeSuggestionSearch()

        fake_libzim = SimpleNamespace(
            Query=FakeQuery,
            Searcher=FakeSearcher,
            SuggestionSearcher=FakeSuggestionSearcher,
        )
        with patch.dict("sys.modules", {"libzim": fake_libzim}):
            paths = list(_iter_paths_from_search(archive=object(), query="x", batch_size=2))
        self.assertEqual(paths, ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
