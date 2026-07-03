# Contributing to Media Converter

Thanks for your interest in improving Media Converter. This guide explains how to propose
changes and what to expect.

## Ways to contribute

- **Report bugs** using the bug report template.
- **Suggest features** using the feature request template.
- **Improve documentation** (typos, clarifications, examples).
- **Submit code** via a pull request (see below).

For security vulnerabilities, do **not** open a public issue. Use private vulnerability
reporting instead, as described in [SECURITY.md](SECURITY.md).

## Branching and pull requests

The `main` branch is protected. All changes, including those from maintainers, go through a
pull request. Direct pushes to `main` are not accepted.

Step 1 - Fork the repository (external contributors) or create a branch (collaborators).

Step 2 - Create a topic branch from `main`, for example `fix/download-button` or
`feature/webm-two-pass`.

Step 3 - Make your change in small, focused commits with clear messages.

Step 4 - Open a pull request against `main` and fill in the pull request template.

Step 5 - Address review feedback. Once approved, a maintainer will merge it.

Keep each pull request focused on a single change so it is easy to review and, if needed,
revert.

## Development setup

Prerequisites: Python 3.8+ and [FFmpeg](README.md#installing-ffmpeg) on your PATH.

```bash
git clone https://github.com/M1XZG/media-converter.git
cd media-converter
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000.

You can also run it with Docker: `docker compose up -d`.

## Before you open a pull request

- Make sure the app still starts: `python app.py` (or `docker compose up`).
- Check that `app.py` compiles: `python -m py_compile app.py cleanup.py`.
- Test the paths your change affects: upload/convert, audio extraction, GIF, the
  downloader, and the media library, as relevant.
- If you touched the UI, test in the browser in both dark and light mode.

## Coding guidelines

- Target Python 3.8+ and keep dependencies minimal; discuss new dependencies first.
- Match the existing style: clear names, small functions, comments only where they add
  clarity.
- Build subprocess commands as argument lists (never `shell=True`) and validate any
  user-supplied input (paths, formats, URLs, sizes).
- When inserting server or user data into the DOM, escape it (see `escapeHtml` in
  `templates/index.html`) or use `textContent`.
- Do not commit secrets, credentials, or personal data.

## Reporting conduct concerns

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree
to uphold it.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
