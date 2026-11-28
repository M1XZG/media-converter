import unittest
from html.parser import HTMLParser
from unittest.mock import MagicMock, patch

from yt_dlp import YoutubeDL

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
        self.site_attributes = []
        self.site_icons = []
        self.url_attributes = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and attrs.get("class") == "supported-sites":
            self.in_panel = True
        if self.in_panel and tag == "li":
            self.in_site = True
            self.site_attributes.append(attrs)
        if self.in_site and tag == "span" and "supported-site-icon" in attrs.get("class", ""):
            self.site_icons.append(attrs)
        if tag == "input" and attrs.get("id") == "youtube-url":
            self.url_attributes = attrs

    def handle_endtag(self, tag):
        if tag == "section":
            self.in_panel = False
        if tag == "li":
            self.in_site = False

    def handle_data(self, data):
        if self.in_site and data.strip():
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
    def test_video_quality_beats_codec_preference_within_cap(self, popen, _available):
        popen.return_value.stdout = []
        popen.return_value.returncode = 1
        client = media_app.app.test_client()
        for quality, expected in (("1080", 1080), ("720", 720), ("480", 360), ("best", 1440)):
            with self.subTest(quality=quality), self.assertLogs("media_converter"):
                response = client.post("/media/download", json={
                    "url": "https://www.youtube.com/watch?v=example", "quality": quality,
                })
                self.addCleanup(media_app._youtube_jobs.pop, response.get_json()["job_id"])
                cmd = popen.call_args.args[0]
                with YoutubeDL({"format": cmd[cmd.index("-f") + 1], "quiet": True,
                                "format_sort": cmd[cmd.index("-S") + 1].split(",")}) as ydl:
                    result = ydl.process_ie_result({
                        "id": "example", "title": "example", "extractor": "test",
                        "formats": [
                            {"format_id": name, "url": f"https://example.test/{name}.mp4",
                             "height": h, "vcodec": vc, "acodec": ac}
                            for name, h, vc, ac in (
                                ("muxed", 360, "avc1", "mp4a.40.2"),
                                ("hd", 720, "avc1", "none"), ("fullhd", 1080, "avc1", "none"),
                                ("qhd", 1440, "vp9", "none"), ("audio", None, "none", "mp4a.40.2"),
                            )
                        ],
                    }, download=False)
                self.assertEqual(result["height"], expected)

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
                self.assertEqual(
                    [attrs["title"] for attrs in parser.site_attributes],
                    parser.sites,
                )
                self.assertEqual(len(parser.site_icons), len(parser.sites))
                self.assertTrue(all(attrs.get("aria-hidden") == "true" for attrs in parser.site_icons))
                for attrs in parser.site_icons:
                    icon_response = client.get(
                        f"/static/site-icons/{attrs['data-site']}.svg"
                    )
                    self.assertEqual(icon_response.status_code, 200)
                    self.assertIn("<svg", icon_response.get_data(as_text=True))
                    icon_response.close()
                self.assertEqual(parser.url_attributes["placeholder"], "Paste a video or audio URL")
                self.assertEqual(
                    set(parser.url_attributes["aria-describedby"].split()),
                    {"supported-sites-note", "url-error"},
                )
                self.assertIn(
                    'href="https://github.com/M1XZG/media-converter"',
                    response.get_data(as_text=True),
                )
                github_icon_response = client.get("/static/site-icons/github.svg")
                self.assertEqual(github_icon_response.status_code, 200)
                github_icon_response.close()
                self.assertNotIn("saved permanently", response.get_data(as_text=True))
                response.close()


if __name__ == "__main__":
    unittest.main()
