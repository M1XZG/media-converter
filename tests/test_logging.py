import unittest
from unittest.mock import MagicMock, patch

from app import _redact_url_for_log, _youtube_jobs, app


class ImmediateThread:
    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


class LoggingHelpersTest(unittest.TestCase):
    def test_preserves_normal_share_parameters(self):
        url = "https://youtu.be/Pcm2yDp8XxI?si=W6example"

        self.assertEqual(_redact_url_for_log(url), url)

    def test_redacts_credential_like_query_parameters(self):
        url = (
            "https://example.com/video?id=123&access_token=secret"
            "&signature=signed-value"
        )

        redacted = _redact_url_for_log(url)

        self.assertIn("id=123", redacted)
        self.assertIn("access_token=%5BREDACTED%5D", redacted)
        self.assertIn("signature=%5BREDACTED%5D", redacted)
        self.assertNotIn("secret", redacted)
        self.assertNotIn("signed-value", redacted)

    @patch("app._ytdlp_available", return_value=True)
    @patch("app.threading.Thread", ImmediateThread)
    @patch("app.subprocess.Popen")
    def test_failed_download_logs_url_job_and_tool_output(
        self,
        popen,
        _ytdlp_available,
    ):
        process = MagicMock()
        process.stdout = [
            "[youtube] Extracting URL: https://youtu.be/Pcm2yDp8XxI?si=W6example\n",
            "ERROR: unable to download video data: HTTP Error 403: Forbidden\n",
        ]
        process.returncode = 1
        popen.return_value = process
        submitted_url = (
            "https://youtu.be/Pcm2yDp8XxI?si=W6example&access_token=secret"
        )

        with self.assertLogs("media_converter", level="INFO") as captured:
            response = app.test_client().post(
                "/media/download",
                json={
                    "url": submitted_url,
                    "mode": "audio",
                    "audio_format": "mp3",
                },
            )

        self.assertEqual(response.status_code, 200)
        job_id = response.get_json()["job_id"]
        logs = "\n".join(captured.output)
        self.assertIn(f"job_id={job_id}", logs)
        self.assertIn("url=https://youtu.be/Pcm2yDp8XxI?si=W6example", logs)
        self.assertIn("HTTP Error 403: Forbidden", logs)
        self.assertNotIn("access_token=secret", logs)
        self.assertEqual(_youtube_jobs[job_id]["status"], "error")


if __name__ == "__main__":
    unittest.main()
