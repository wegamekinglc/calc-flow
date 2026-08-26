from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from shutil import rmtree

if __package__:
    from scripts.build_python_release import (
        build_commands,
        prepare_dist_dir,
        resolve_dist_dir,
    )
else:
    from build_python_release import build_commands, prepare_dist_dir, resolve_dist_dir


ROOT = Path(__file__).resolve().parents[1]


class BuildPythonReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        target = ROOT / "target"
        target.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(dir=target))
        self.addCleanup(rmtree, self.directory, ignore_errors=True)

    def test_dist_dir_must_be_a_child_of_target(self) -> None:
        self.assertEqual(
            resolve_dist_dir(self.directory, ROOT), self.directory.resolve()
        )
        for invalid in (ROOT, ROOT / "target", ROOT / "dist"):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "target"),
            ):
                resolve_dist_dir(invalid, ROOT)

    def test_prepare_dist_dir_rejects_stale_artifacts(self) -> None:
        artifact = self.directory / "stale.whl"
        artifact.write_bytes(b"stale")

        with self.assertRaisesRegex(ValueError, "not empty"):
            prepare_dist_dir(self.directory, clean=False)

        self.assertTrue(artifact.is_file())

    def test_prepare_dist_dir_cleans_only_the_validated_directory(self) -> None:
        artifact = self.directory / "stale.whl"
        artifact.write_bytes(b"stale")

        prepare_dist_dir(self.directory, clean=True)

        self.assertTrue(self.directory.is_dir())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_build_commands_cover_core_sdist_and_studio(self) -> None:
        commands = build_commands(ROOT, self.directory)

        self.assertEqual(
            [step.name for step in commands],
            [
                "core wheel",
                "core source distribution",
                "Studio frontend dependencies",
                "Studio frontend",
                "Studio wheel",
            ],
        )
        self.assertEqual(
            commands[0].command[:5],
            ("uv", "run", "maturin", "build", "--release"),
        )
        self.assertIn("--locked", commands[0].command)
        npm = "npm.cmd" if os.name == "nt" else "npm"
        self.assertEqual(commands[2].command, (npm, "ci"))
        self.assertEqual(commands[2].cwd, ROOT / "web-ui")
        self.assertEqual(commands[-1].cwd, ROOT)
        self.assertEqual(commands[-1].command[:3], ("uv", "build", "--project"))


if __name__ == "__main__":
    unittest.main()
