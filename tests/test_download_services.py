import unittest
from html.parser import HTMLParser
from unittest.mock import MagicMock, patch

import app as media_app


NEW_SITE_URLS = {
    "reddit": ("https://www.reddit.com/r/videos/comments/example/", "https://v.redd.it/example"),
    "facebook": ("https://www.facebook.com/reel/123", "https://fb.watch/example/"),
    "twitch": ("https://www.twitch.tv/videos/123", "https://clips.twitch.tv/Example"),
    "vimeo": ("https://vimeo.com/123", "https://player.vimeo.com/video/123"),
    "bluesky": ("https://bsky.app/profile/example.test/post/example",),
    "dailymotion": ("https://www.dailymotion.com/video/example", "https://dai.ly/example"),
    "pinterest": ("https://www.pinterest.com/pin/123/", "https://www.pinterest.co.uk/pin/123/", "https://pin.it/example"),
    "streamable": ("https://streamable.com/example",),
}


class SupportedSitesParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_panel = False
        self.in_site = False
        self.sites = []
        self.url_attributes = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and attrs.get("class") == "supported-sites":
            self.in_panel = True
        if self.in_panel and tag == "li":
            self.in_site = True
        if tag == "input" and attrs.get("id") == "youtube-url":
            self.url_attributes = attrs

    def handle_endtag(self, tag):
        if tag == "section":
            self.in_panel = False
        if tag == "li":
            self.in_site = False

    def handle_data(self, data):
        if self.in_site:
            self.sites.append(data.strip())


class ImmediateThread:
    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


class DownloadServicesTest(unittest.TestCase):
    def test_new_sites_and_share_links_are_recognised(self):
        for service, urls in NEW_SITE_URLS.items():
            for url in urls:
                with self.subTest(url=url):
                    self.assertEqual(media_app._detect_download_service(url), service)

    def test_lookalike_domains_are_not_accepted(self):
        for domain in ("reddit.com.evil.test", "notfacebook.com", "twitch.tv@evil.test"):
            with self.subTest(domain=domain):
                self.assertIsNone(media_app._detect_download_service(f"https://{domain}/video"))

    @patch("app._ytdlp_available", return_value=True)
    @patch("app.threading.Thread", ImmediateThread)
    @patch("app.subprocess.Popen")
    def test_new_sites_start_jobs_in_their_own_folders(self, popen, _available):
        process = MagicMock()
        process.stdout = []
        process.returncode = 1
        popen.return_value = process
        client = media_app.app.test_client()
        for service, urls in NEW_SITE_URLS.items():
            for mode in ("video", "audio"):
                with self.subTest(service=service, mode=mode):
                    with self.assertLogs("media_converter"):
                        response = client.post("/media/download", json={"url": urls[0], "mode": mode})
                    self.assertEqual(response.status_code, 200)
                    job_id = response.get_json()["job_id"]
                    try:
                        job = media_app._youtube_jobs[job_id]
                        self.assertEqual(job["service"], service)
                        self.assertEqual(job["mode"], mode)
                        command = popen.call_args.args[0]
                        self.assertEqual(command[-1], urls[0])
                        self.assertEqual(
                            command[command.index("-o") + 1],
                            str(media_app.DOWNLOADS_FOLDER / service / f"{job_id}.%(ext)s"),
                        )
                    finally:
                        media_app._youtube_jobs.pop(job_id, None)

    @patch("app._detect_gpu_encoder", return_value={})
    def test_supported_sites_panel_in_full_and_downloader_only_modes(self, _gpu):
        client = media_app.app.test_client()
        for converter_enabled in (True, False):
            with self.subTest(converter_enabled=converter_enabled), patch.object(
                media_app, "ENABLE_CONVERTER", converter_enabled
            ):
                response = client.get("/")
                parser = SupportedSitesParser()
                parser.feed(response.get_data(as_text=True))
                self.assertEqual(
                    parser.sites,
                    ["YouTube", "Instagram", "TikTok", "X/Twitter", "Reddit", "Facebook",
                     "Twitch", "Vimeo", "Bluesky", "Dailymotion", "Pinterest", "Streamable", "Spotify"],
                )
                self.assertNotIn("PornHub", parser.sites)
                self.assertEqual(parser.url_attributes["placeholder"], "Paste a video or audio URL")
                self.assertEqual(
                    set(parser.url_attributes["aria-describedby"].split()),
                    {"supported-sites-note", "url-error"},
                )
                self.assertNotIn("saved permanently", response.get_data(as_text=True))
                response.close()


if __name__ == "__main__":
    unittest.main()
