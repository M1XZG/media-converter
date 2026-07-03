# Security Policy

## Supported versions

Media Converter is released from the `main` branch. Security fixes are applied to the
latest version. Please make sure you are running the most recent release or the current
`main` before reporting an issue.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues, pull
requests, or discussions.**

Instead, report them privately using GitHub's private vulnerability reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** (or use this direct link:
   https://github.com/M1XZG/media-converter/security/advisories/new).
3. Provide a description of the issue, the steps to reproduce it, the affected version or
   commit, and the potential impact.

You should receive an acknowledgement as soon as reasonably possible. If the report is
accepted, a fix will be prepared and a security advisory published; if it is declined, an
explanation will be provided.

## Scope and deployment notes

Media Converter is a **self-hosted tool intended to run on a trusted network**. Some
behaviours are by design rather than vulnerabilities:

- The application has **no built-in authentication** and binds to `0.0.0.0` by default.
  Do not expose it directly to the public internet. Put it behind a reverse proxy with
  authentication and TLS, or restrict access with a VPN or firewall.
- **Uploads are unlimited by default** (`MAX_CONTENT_LENGTH=0`). Set a byte limit if the
  instance is reachable by anyone other than you.
- The `downloads/` folder is **never auto-cleaned**.

See the [Security section of the README](README.md#security) for hardening guidance.

When reporting, please focus on issues that are exploitable in a reasonable deployment
(for example path traversal, remote code execution, injection, or authentication bypass in
a proxied setup), rather than the documented "no auth by default" behaviour above.

## Keeping dependencies current

FFmpeg and yt-dlp process untrusted media and URLs. Keep them and the Python dependencies
up to date so upstream security fixes are picked up:

```bash
pip install -U -r requirements.txt
```

Update FFmpeg through your package manager, or rebuild the Docker image to pull a newer
base image.
