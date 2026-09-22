from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.dal301_groupby import profile

# Byte-identical rustc.log from run 35616026184, jobs 106386841092/106386841296.
RUSTC_LOG = """info: syncing channel updates for 1.88.0-x86_64-unknown-linux-gnu
info: latest update on 2025-06-26 for version 1.88.0 (6b00bc388 2025-06-23)
info: downloading 3 components
rustc 1.88.0 (6b00bc388 2025-06-23)
binary: rustc
commit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc
commit-date: 2025-06-23
host: x86_64-unknown-linux-gnu
release: 1.88.0
LLVM version: 20.1.5
"""


class ProfileBuildTests(unittest.IsolatedAsyncioTestCase):
    async def run_preflight(self, text, side="A", code=0):
        async def command(argv, *, log, **kwargs):
            if argv == ["rustc", "-Vv"]:
                log.write_text(text, encoding="utf-8")
                if code:
                    raise RuntimeError(f"command exited {code}")
                return
            self.assertEqual(argv[1:4], ["-m", "maturin", "build"])
            log.write_text("native build deliberately not executed\n")
            raise RuntimeError("command exited 99")

        identity = SimpleNamespace(
            source_identity=mock.AsyncMock(
                return_value={
                    "git_sha": profile.SEALS[side]["git_sha"],
                    "git_clean": True,
                }
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            (source / "pyproject.toml").write_text("strip = true\n")
            with (
                mock.patch.dict(
                    "sys.modules", {"scripts.profile_warm_stream": identity}
                ),
                mock.patch.object(profile, "command", side_effect=command),
            ):
                self.assertEqual(await profile.build(side, source, output), 1)
            self.assertEqual((output / "rustc.log").read_text(), text)
            record = json.loads((output / "rustc.command.json").read_text())
            self.assertEqual(record["argv"], ["rustc", "-Vv"])
            self.assertEqual(record["exit_code"], code)
            return (
                json.loads((output / "outcome.json").read_text()),
                (output / "build.command.json").exists(),
            )

    async def test_rustup_preamble_does_not_reject_real_ab_logs(self):
        for side in ("A", "B"):
            with self.subTest(side=side):
                outcome, reached_build = await self.run_preflight(RUSTC_LOG, side)
                self.assertTrue(reached_build, outcome)
                self.assertIn("command exited 99", outcome["error"])

    async def test_normal_verbose_release_reaches_build(self):
        text = RUSTC_LOG[RUSTC_LOG.index("rustc 1.88.0") :]
        outcome, reached_build = await self.run_preflight(text)
        self.assertTrue(reached_build, outcome)

    async def test_missing_duplicate_or_conflicting_release_stops_build(self):
        for text in (
            RUSTC_LOG.replace("release: 1.88.0\n", ""),
            RUSTC_LOG + "release: 1.88.0\n",
            RUSTC_LOG + "release: 1.89.0\n",
            RUSTC_LOG + "release:\n",
        ):
            with self.subTest(text=text):
                outcome, reached_build = await self.run_preflight(text)
                self.assertFalse(reached_build)
                self.assertIn("exactly one rustc release field", outcome["error"])

    async def test_wrong_or_empty_release_stops_build(self):
        normal = RUSTC_LOG[RUSTC_LOG.index("rustc 1.88.0") :]
        for release in ("1.89.0", "1.88.0-nightly", "", "1.88.0 extra"):
            with self.subTest(release=release):
                outcome, reached_build = await self.run_preflight(
                    normal.replace("release: 1.88.0", f"release: {release}")
                )
                self.assertFalse(reached_build)
                self.assertIn(
                    "build settings or dependency lock differ", outcome["error"]
                )

    async def test_failed_rustc_keeps_original_exit_and_stops_build(self):
        for code in (42, -9):
            with self.subTest(code=code):
                outcome, reached_build = await self.run_preflight(RUSTC_LOG, code=code)
                self.assertFalse(reached_build)
                self.assertEqual(outcome["status"], "failed")
                self.assertIn(f"command exited {code}", outcome["error"])


if __name__ == "__main__":
    unittest.main()
