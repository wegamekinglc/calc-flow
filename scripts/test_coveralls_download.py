from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "coveralls-linux-x86_64.tar.gz"

# Download and verification sequence from the pinned action's Linux installer:
# coverallsapp/github-action@8d6379e14d29928660c4ba802d8e85393440b329.
INSTALL = r"""
set -euo pipefail
platform_filename=coveralls-linux-x86_64.tar.gz
if ! curl -sLO "$COVERALLS_TEST_URL/$platform_filename" ||
  ! curl -sLO "$COVERALLS_TEST_URL/coveralls-checksums.txt"; then
  exit 1
fi
expected_checksum=$(
  grep "${platform_filename}" coveralls-checksums.txt | awk '{print $1}'
)
test -n "$expected_checksum"
actual_checksum=$(sha256sum "${platform_filename}" | awk '{print $1}')
test "$expected_checksum" = "$actual_checksum"
tar -xzf "${platform_filename}"
./coveralls
"""


def _reporter_archive(binary: bytes = b"#!/bin/sh\ntouch reported\n") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        member = tarfile.TarInfo("coveralls")
        member.mode = 0o755
        member.size = len(binary)
        archive.addfile(member, io.BytesIO(binary))
    return output.getvalue()


@contextmanager
def _serve(responses: dict[str, tuple[tuple[int, bytes], ...]]):
    requests: Counter[str] = Counter()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            choices = responses[self.path]
            status, body = choices[min(requests[self.path], len(choices) - 1)]
            requests[self.path] += 1
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            worker.join()


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux Coveralls installer")
class CoverallsDownloadTests(unittest.TestCase):
    def _install(self, responses, directory: Path):
        with _serve(responses) as (url, requests):
            result = subprocess.run(
                ["bash", "-c", INSTALL],
                cwd=directory,
                env={
                    **os.environ,
                    "CURL_HOME": str(ROOT / ".github/coveralls"),
                    "NO_PROXY": "127.0.0.1",
                    "no_proxy": "127.0.0.1",
                    "COVERALLS_TEST_URL": url,
                },
                capture_output=True,
                text=True,
                timeout=30,
            )
        return result, requests

    def test_transient_http_errors_recover_for_archive_and_checksum(self) -> None:
        archive = _reporter_archive()
        checksum = f"{hashlib.sha256(archive).hexdigest()}  {ARCHIVE}\n".encode()
        responses = {
            f"/{ARCHIVE}": ((504, b"gateway timeout"), (200, archive)),
            "/coveralls-checksums.txt": ((503, b"unavailable"), (200, checksum)),
        }
        with tempfile.TemporaryDirectory() as directory:
            result, requests = self._install(responses, Path(directory))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(directory) / "reported").exists())
            self.assertEqual(requests[f"/{ARCHIVE}"], 2)
            self.assertEqual(requests["/coveralls-checksums.txt"], 2)

    def test_persistent_http_errors_fail_without_executing_reporter(self) -> None:
        for failed_asset in (ARCHIVE, "coveralls-checksums.txt"):
            with (
                self.subTest(asset=failed_asset),
                tempfile.TemporaryDirectory() as directory,
            ):
                responses = {
                    f"/{ARCHIVE}": ((200, _reporter_archive()),),
                    "/coveralls-checksums.txt": ((200, b"unused"),),
                    f"/{failed_asset}": ((504, b"gateway timeout"),),
                }
                result, requests = self._install(responses, Path(directory))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(requests[f"/{failed_asset}"], 5)
                self.assertFalse((Path(directory) / "coveralls").exists())
                self.assertFalse((Path(directory) / "reported").exists())

    def test_reporter_failure_is_propagated_after_verified_download(self) -> None:
        archive = _reporter_archive(b"#!/bin/sh\nexit 9\n")
        checksum = f"{hashlib.sha256(archive).hexdigest()}  {ARCHIVE}\n".encode()
        responses = {
            f"/{ARCHIVE}": ((200, archive),),
            "/coveralls-checksums.txt": ((200, checksum),),
        }
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self._install(responses, Path(directory))
            self.assertEqual(result.returncode, 9)

    def test_invalid_checksum_fails_without_extracting_reporter(self) -> None:
        for checksum in (f"{'0' * 64}  {ARCHIVE}\n".encode(), b"<html>504</html>"):
            with (
                self.subTest(checksum=checksum),
                tempfile.TemporaryDirectory() as directory,
            ):
                responses = {
                    f"/{ARCHIVE}": ((200, _reporter_archive()),),
                    "/coveralls-checksums.txt": ((200, checksum),),
                }
                result, requests = self._install(responses, Path(directory))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(requests[f"/{ARCHIVE}"], 1)
                self.assertEqual(requests["/coveralls-checksums.txt"], 1)
                self.assertFalse((Path(directory) / "coveralls").exists())
                self.assertFalse((Path(directory) / "reported").exists())
