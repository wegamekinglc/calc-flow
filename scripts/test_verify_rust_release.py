from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.verify_rust_release import inspect_crate, validate_tag


class RustReleaseTests(unittest.TestCase):
    def test_tag_must_match_the_shared_version(self) -> None:
        validate_tag("calc-flow-v2026.11.2", "2026.11.2")
        for tag in ("calc-flow-python-v2026.11.2", "calc-flow-v2026.11.1"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                validate_tag(tag, "2026.11.2")

    def test_archive_is_self_contained_and_keeps_required_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = self._archive(Path(directory))
            inspect_crate(archive, "2026.11.2")

    def test_archive_rejects_missing_distribution_files(self) -> None:
        for missing in (
            "LICENSE",
            "README.md",
            "src/lib.rs",
            "examples/example.rs",
            "benches/core.rs",
            "Cargo.lock",
        ):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as directory,
            ):
                archive = self._archive(Path(directory), missing=missing)
                with self.assertRaises(ValueError):
                    inspect_crate(archive, "2026.11.2")

    def test_archive_rejects_repository_only_files_and_unsafe_paths(self) -> None:
        for extra in ("tests/integration.rs", "../outside.rs", "AGENTS.md"):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                archive = self._archive(Path(directory), extra=extra)
                with self.assertRaises(ValueError):
                    inspect_crate(archive, "2026.11.2")

    def test_archive_rejects_unpublished_dependency_sources(self) -> None:
        for source in (
            'path = "../core"',
            'git = "https://example.org/core"',
            "workspace = true",
            'registry = "private"',
        ):
            with (
                self.subTest(source=source),
                tempfile.TemporaryDirectory() as directory,
            ):
                archive = self._archive(Path(directory), dependency=source)
                with self.assertRaises(ValueError):
                    inspect_crate(archive, "2026.11.2")

    def test_archive_rejects_wrong_identity_and_workspace_patches(self) -> None:
        for suffix in ("\n[patch.crates-io]\n", "\n[workspace]\n"):
            with (
                self.subTest(suffix=suffix),
                tempfile.TemporaryDirectory() as directory,
            ):
                archive = self._archive(Path(directory), suffix=suffix)
                with self.assertRaises(ValueError):
                    inspect_crate(archive, "2026.11.2")
        with tempfile.TemporaryDirectory() as directory:
            archive = self._archive(Path(directory))
            with self.assertRaises(ValueError):
                inspect_crate(archive, "2026.11.3")

    @staticmethod
    def _archive(
        directory: Path,
        *,
        missing: str = "",
        extra: str = "",
        dependency: str = 'version = "1"',
        suffix: str = "",
    ) -> Path:
        files = {
            "Cargo.toml": (
                '[package]\nname = "calc-flow"\nversion = "2026.11.2"\n'
                'license = "Apache-2.0"\nreadme = "README.md"\n'
                f"[dependencies.example]\n{dependency}\n{suffix}"
            ),
            "Cargo.lock": "version = 4\n",
            "LICENSE": "Apache License\nVersion 2.0, January 2004\n",
            "README.md": "Calc Flow Rust usage\n",
            "src/lib.rs": "",
            "examples/example.rs": "",
            "benches/core.rs": "",
        }
        if missing:
            del files[missing]
        if extra:
            files[extra] = ""
        path = directory / "calc-flow-2026.11.2.crate"
        with tarfile.open(path, "w:gz") as archive:
            for name, content in files.items():
                data = content.encode("utf-8")
                member = tarfile.TarInfo(f"calc-flow-2026.11.2/{name}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return path


if __name__ == "__main__":
    unittest.main()
