"""Exercise source checks without downloading dependencies or building engines."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().with_name("reproduce.sh")
REPOSITORY = SCRIPT.parents[3]
BASELINE = "dab98a559314bd46d866ed7a6489abcdd3810afe"
SETUP_REACHED = 73


class ReproducePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        target = REPOSITORY / "target"
        target.mkdir(exist_ok=True)
        temporary = Path(self.enterContext(TemporaryDirectory(dir=target)))
        self.checkout = temporary / "checkout"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--shared",
                "--no-checkout",
                str(REPOSITORY),
                str(self.checkout),
            ],
            check=True,
            capture_output=True,
        )
        self.git("checkout", "--quiet", "--detach", BASELINE)
        self.run_dir = self.checkout / "target/arroyo-comparison-reproduction"
        bin_dir = temporary / "bin"
        bin_dir.mkdir()
        uv = bin_dir / "uv"
        uv.write_text(f"#!/usr/bin/env bash\nexit {SETUP_REACHED}\n", encoding="utf-8")
        uv.chmod(0o755)
        self.environment = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        }

    def git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments], cwd=self.checkout, check=True, capture_output=True
        )

    def reproduce(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=self.checkout,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def change_source(self) -> None:
        source = self.checkout / "python/calc_flow/__init__.py"
        source.write_text(
            source.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8"
        )

    def test_clean_baseline_passes_source_check(self) -> None:
        result = self.reproduce()
        self.assertEqual(result.returncode, SETUP_REACHED, result.stderr)
        self.assertTrue((self.run_dir / "bin").is_dir())

    def test_unstaged_source_change_is_rejected_before_setup(self) -> None:
        self.change_source()
        result = self.reproduce()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(self.run_dir.exists())

    def test_staged_source_change_is_rejected_before_setup(self) -> None:
        self.change_source()
        self.git("add", "python/calc_flow/__init__.py")
        result = self.reproduce()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(self.run_dir.exists())


if __name__ == "__main__":
    unittest.main()
