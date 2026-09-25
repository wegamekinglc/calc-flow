"""Focused checks for repairing an incomplete hosted Rust toolchain."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.repair_rust_toolchain import repair_incomplete_toolchain


class RepairRustToolchainTests(unittest.TestCase):
    def test_missing_toolchain_needs_no_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("subprocess.run") as run:
                self.assertFalse(repair_incomplete_toolchain(Path(directory)))
            run.assert_not_called()

    def test_complete_toolchain_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = (
                root
                / "toolchains/1.88.0-x86_64-unknown-linux-gnu"
                / "lib/rustlib/multirust-config.toml"
            )
            metadata.parent.mkdir(parents=True)
            metadata.write_text("[toolchain]\n", encoding="utf-8")
            (metadata.parent / "rust-installer-version").write_text(
                "3\n", encoding="utf-8"
            )
            with patch("subprocess.run") as run:
                self.assertFalse(repair_incomplete_toolchain(root))
            run.assert_not_called()

    def test_incomplete_toolchain_is_uninstalled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "toolchains/1.88.0-x86_64-unknown-linux-gnu"
            (partial / "bin").mkdir(parents=True)
            (partial / "bin/cargo-clippy").write_text("partial", encoding="utf-8")
            with patch("subprocess.run") as run:
                self.assertTrue(repair_incomplete_toolchain(root))
            run.assert_called_once_with(
                ["rustup", "toolchain", "uninstall", "1.88.0"], check=True
            )

    def test_uninstall_failure_is_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "toolchains/1.88.0-x86_64-unknown-linux-gnu"
            partial.mkdir(parents=True)
            with (
                patch(
                    "subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, ["rustup"]),
                ),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                repair_incomplete_toolchain(root)


if __name__ == "__main__":
    unittest.main()
