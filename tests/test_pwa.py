import unittest

from app import app


class ProgressiveWebAppTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_manifest_describes_installable_app(self):
        response = self.client.get("/manifest.webmanifest")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/manifest+json")
        manifest = response.get_json()
        self.assertEqual(manifest["start_url"], "/")
        self.assertEqual(manifest["scope"], "/")
        self.assertEqual(manifest["display"], "standalone")
        self.assertEqual(
            {icon["sizes"] for icon in manifest["icons"]},
            {"192x192", "512x512"},
        )
        response.close()

    def test_service_worker_controls_the_app_root(self):
        response = self.client.get("/service-worker.js")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Service-Worker-Allowed"], "/")
        self.assertIn(b"media-converter-shell-v1", response.data)
        response.close()

    def test_home_page_has_install_and_clipboard_controls(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'rel="manifest"', response.data)
        self.assertIn(b'id="install-app-btn"', response.data)
        self.assertIn(b"navigator.clipboard.readText()", response.data)
        response.close()


if __name__ == "__main__":
    unittest.main()
