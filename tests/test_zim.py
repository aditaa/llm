import unittest

from llm.zim import _blob_to_bytes, html_to_text, normalize_whitespace


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


if __name__ == "__main__":
    unittest.main()
