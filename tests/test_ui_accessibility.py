import unittest
from html.parser import HTMLParser
from unittest.mock import patch

import app as media_app


class PageParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def by_id(self, element_id):
        return next(attrs for _, attrs in self.elements if attrs.get("id") == element_id)


class AccessiblePagesTest(unittest.TestCase):
    @patch("app._detect_gpu_encoder", return_value={})
    def test_home_controls_have_names_and_descriptions_in_each_mode(self, _gpu):
        for converter, downloader in ((True, True), (True, False), (False, True)):
            with self.subTest(converter=converter, downloader=downloader), patch.object(
                media_app, "ENABLE_CONVERTER", converter
            ), patch.object(media_app, "ENABLE_DOWNLOADER", downloader):
                response = media_app.app.test_client().get("/")
                page = PageParser(response.get_data(as_text=True))
                ids = {attrs["id"] for _, attrs in page.elements if "id" in attrs}
                self.assertEqual(sum(tag == "main" for tag, _ in page.elements), 1)
                for tag, attrs in page.elements:
                    if "aria-describedby" in attrs:
                        self.assertTrue(set(attrs["aria-describedby"].split()) <= ids)
                    if tag == "label":
                        self.assertIn(attrs["for"], ids)
                    if attrs.get("role") == "progressbar":
                        self.assertTrue(attrs.get("aria-label"))
                if converter:
                    self.assertEqual(page.by_id("file-input")["class"], "visually-hidden")
                    self.assertTrue(any(
                        tag == "label" and attrs.get("for") == "file-input"
                        for tag, attrs in page.elements
                    ))
                if downloader:
                    self.assertIn("required", page.by_id("youtube-url"))
                    self.assertTrue(any(
                        tag == "label" and attrs.get("for") == "youtube-url"
                        for tag, attrs in page.elements
                    ))
                    self.assertEqual(page.by_id("youtube-download-btn")["type"], "submit")
                    self.assertEqual(page.by_id("youtube-abort-btn")["type"], "button")
                response.close()

    @patch("app._list_files_recursive")
    def test_library_rows_and_scroll_regions_are_accessibly_labelled(self, list_files):
        list_files.return_value = [{
            "relative_path": "youtube/a-long-video-name.mp4",
            "modified": "2026-11-26 12:00",
            "size": "1 MB",
        }]
        response = media_app.app.test_client().get("/files")
        page = PageParser(response.get_data(as_text=True))
        self.assertEqual(sum(tag == "main" for tag, _ in page.elements), 1)
        self.assertEqual(sum(tag == "caption" for tag, _ in page.elements), 2)
        for tag, attrs in page.elements:
            if tag == "input":
                self.assertTrue(attrs.get("aria-label"))
            if tag == "th":
                self.assertEqual(attrs["scope"], "col")
            if attrs.get("class") == "table-wrap":
                self.assertEqual(attrs["tabindex"], "0")
                self.assertTrue(page.by_id(attrs["aria-labelledby"]))
        self.assertEqual(page.by_id("status")["role"], "status")
        response.close()


if __name__ == "__main__":
    unittest.main()
