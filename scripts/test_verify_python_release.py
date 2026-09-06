from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from shutil import rmtree
from tarfile import TarInfo
from tarfile import open as open_tar
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

if __package__:
    from scripts.release_baseline import resolve_release_baseline
    from scripts.verify_python_release import (
        CORE_TARGETS,
        ensure_version_is_new_on_pypi,
        validate_release,
        validate_versions,
    )
else:
    from release_baseline import resolve_release_baseline
    from verify_python_release import (
        CORE_TARGETS,
        ensure_version_is_new_on_pypi,
        validate_release,
        validate_versions,
    )


ROOT = Path(__file__).resolve().parents[1]
VERSION = "4.0.0"
PLATFORMS = {
    "linux-aarch64": "manylinux_2_28_aarch64",
    "linux-x86_64": "manylinux_2_28_x86_64",
    "macos-arm64": "macosx_11_0_arm64",
    "macos-x86_64": "macosx_10_12_x86_64",
    "windows-amd64": "win_amd64",
}


class VerifyPythonReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, self.directory)

    @staticmethod
    def _metadata(name: str, *, requires_dist: str | None = None) -> bytes:
        metadata = Message()
        metadata["Metadata-Version"] = "2.4"
        metadata["Name"] = name
        metadata["Version"] = VERSION
        metadata["License-Expression"] = "Apache-2.0"
        metadata["Requires-Python"] = ">=3.13"
        if requires_dist is not None:
            metadata["Requires-Dist"] = requires_dist
        metadata.set_payload("Calc Flow release fixture\n")
        return metadata.as_bytes()

    @staticmethod
    def _wheel_metadata(*, pure: bool, tags: tuple[str, ...]) -> bytes:
        metadata = Message()
        metadata["Wheel-Version"] = "1.0"
        metadata["Generator"] = "test"
        metadata["Root-Is-Purelib"] = str(pure).lower()
        for tag in tags:
            metadata["Tag"] = tag
        return metadata.as_bytes()

    def _core_wheel(self, target: str, *, abi: str = "abi3") -> Path:
        platform = PLATFORMS[target]
        wheel = (
            self.directory / f"calc_flow_python-{VERSION}-cp313-{abi}-{platform}.whl"
        )
        dist_info = f"calc_flow_python-{VERSION}.dist-info"
        extension = "_native.pyd" if target == "windows-amd64" else "_native.abi3.so"
        tags = tuple(f"cp313-{abi}-{part}" for part in platform.split("."))
        with ZipFile(wheel, "w") as archive:
            archive.writestr("calc_flow/__init__.py", f'__version__ = "{VERSION}"\n')
            archive.writestr(f"calc_flow/{extension}", b"native")
            archive.writestr(
                f"{dist_info}/METADATA", self._metadata("calc-flow-python")
            )
            archive.writestr(
                f"{dist_info}/WHEEL",
                self._wheel_metadata(pure=False, tags=tags),
            )
            archive.writestr(
                f"{dist_info}/licenses/LICENSE",
                b"Apache License\nVersion 2.0, January 2004\n",
            )
        return wheel

    def _studio_wheel(self) -> Path:
        wheel = self.directory / f"calc_flow_studio-{VERSION}-py3-none-any.whl"
        dist_info = f"calc_flow_studio-{VERSION}.dist-info"
        with ZipFile(wheel, "w") as archive:
            archive.writestr("calc_flow_studio/__init__.py", b"fixture")
            archive.writestr("calc_flow_studio/static/index.html", b"fixture")
            archive.writestr(
                f"{dist_info}/METADATA",
                self._metadata(
                    "calc-flow-studio",
                    requires_dist="calc-flow-python<5,>=4.0.0",
                ),
            )
            archive.writestr(
                f"{dist_info}/WHEEL",
                self._wheel_metadata(pure=True, tags=("py3-none-any",)),
            )
            archive.writestr(
                f"{dist_info}/licenses/LICENSE",
                b"Apache License\nVersion 2.0, January 2004\n",
            )
        return wheel

    def _sdist(self) -> Path:
        sdist = self.directory / f"calc_flow_python-{VERSION}.tar.gz"
        root = f"calc_flow_python-{VERSION}"
        entries = {
            "LICENSE": b"Apache License\nVersion 2.0, January 2004\n",
            "Cargo.lock": b"fixture",
            "pyproject.toml": b"fixture",
            "crates/calc-flow/Cargo.toml": b"fixture",
            "crates/calc-flow/src/lib.rs": b"fixture",
            "crates/calc-flow-python/Cargo.toml": b"fixture",
            "python/calc_flow/__init__.py": b"fixture",
        }
        with open_tar(sdist, "w:gz") as archive:
            for name, content in entries.items():
                info = TarInfo(f"{root}/{name}")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return sdist

    def _complete_release(self) -> None:
        for target in sorted(CORE_TARGETS):
            self._core_wheel(target)
        self._studio_wheel()
        self._sdist()

    def test_versions_match_the_release_tag(self) -> None:
        config = validate_versions(root=ROOT, tag="v4.0.0")

        self.assertEqual(config.version, VERSION)
        self.assertEqual(config.requires_python, ">=3.13")

    def test_pypi_preflight_checks_only_the_published_core_project(self) -> None:
        with patch(
            "scripts.verify_python_release.ensure_version_is_new_on_pypi"
        ) as check:
            validate_versions(root=ROOT, tag="v4.0.0", check_pypi=True)

        check.assert_called_once_with("calc-flow-python", VERSION)

    def test_versions_reject_a_mismatched_release_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal 'v4.0.0'"):
            validate_versions(root=ROOT, tag="v4.0.1")

    def test_complete_release_has_a_stable_relative_manifest(self) -> None:
        self._complete_release()

        manifest = validate_release(self.directory, root=ROOT)

        self.assertEqual(len(manifest), 7)
        self.assertEqual(
            manifest, sorted(manifest, key=lambda line: line.split("  ", 1)[1])
        )
        self.assertTrue(all("  calc_flow" in line for line in manifest))

    def test_release_rejects_a_missing_core_target(self) -> None:
        for target in sorted(CORE_TARGETS - {"linux-aarch64"}):
            self._core_wheel(target)
        self._studio_wheel()
        self._sdist()

        with self.assertRaisesRegex(ValueError, "missing=.*linux-aarch64"):
            validate_release(self.directory, root=ROOT)

    def test_release_rejects_a_non_abi3_core_wheel(self) -> None:
        for target in sorted(CORE_TARGETS - {"windows-amd64"}):
            self._core_wheel(target)
        self._core_wheel("windows-amd64", abi="cp313")
        self._studio_wheel()
        self._sdist()

        with self.assertRaisesRegex(ValueError, "ABI.*abi3"):
            validate_release(self.directory, root=ROOT)

    def test_pypi_check_uses_the_fixed_https_endpoint(self) -> None:
        response = MagicMock(status=404)
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch(
            "scripts.verify_python_release.HTTPSConnection",
            return_value=connection,
        ) as constructor:
            ensure_version_is_new_on_pypi("calc-flow-python", "4.0.0+candidate")

        constructor.assert_called_once_with("pypi.org", timeout=20)
        connection.request.assert_called_once_with(
            "GET",
            "/pypi/calc-flow-python/4.0.0%2Bcandidate/json",
            headers={"Accept": "application/json"},
        )
        response.read.assert_called_once_with()
        connection.close.assert_called_once_with()

    def test_pypi_check_rejects_an_existing_version(self) -> None:
        response = MagicMock(status=200)
        connection = MagicMock()
        connection.getresponse.return_value = response
        with (
            patch(
                "scripts.verify_python_release.HTTPSConnection",
                return_value=connection,
            ),
            self.assertRaisesRegex(ValueError, "already exists on PyPI"),
        ):
            ensure_version_is_new_on_pypi("calc-flow-python", VERSION)


class ReleaseBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, self.directory)
        self.git("init", "--quiet")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "release-test@example.invalid")
        self.git("commit", "--allow-empty", "-m", "baseline")
        self.baseline = self.git("rev-parse", "HEAD")
        self.git("commit", "--allow-empty", "-m", "candidate")

    def git(self, *arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=self.directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def test_first_release_requires_an_explicit_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit initial baseline"):
            resolve_release_baseline(self.directory)

    def test_rehearsal_accepts_an_explicit_ancestor_sha(self) -> None:
        self.assertEqual(
            resolve_release_baseline(self.directory, initial=self.baseline),
            self.baseline,
        )

    def test_tag_reads_the_initial_baseline_from_its_annotation(self) -> None:
        self.git(
            "tag",
            "-a",
            "v4.0.0",
            "-m",
            f"Release 4.0.0\n\nBenchmark-Baseline: {self.baseline}",
        )
        self.assertEqual(
            resolve_release_baseline(self.directory, tag="v4.0.0"),
            self.baseline,
        )

    def test_first_release_rejects_invalid_or_non_ancestor_baselines(self) -> None:
        for baseline in (self.baseline[:8], "f" * 40, self.git("rev-parse", "HEAD")):
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                resolve_release_baseline(self.directory, initial=baseline)

        self.git("checkout", "--orphan", "unrelated")
        self.git("commit", "--allow-empty", "-m", "unrelated")
        with self.assertRaisesRegex(ValueError, "ancestor"):
            resolve_release_baseline(self.directory, initial=self.baseline)

    def test_tag_rejects_duplicate_or_invalid_baseline_annotations(self) -> None:
        for index, message in enumerate(
            (
                "Benchmark-Baseline: short",
                f"Benchmark-Baseline: {self.baseline}\n"
                f"Benchmark-Baseline: {self.baseline}",
            )
        ):
            tag = f"v4.0.{index}"
            self.git("tag", "-a", tag, "-m", message)
            with self.subTest(message=message), self.assertRaises(ValueError):
                resolve_release_baseline(self.directory, tag=tag)

    def test_previous_release_cannot_be_overridden_by_a_bootstrap_baseline(
        self,
    ) -> None:
        self.git("tag", "-a", "v3.0.0", self.baseline, "-m", "Previous release")
        self.assertEqual(resolve_release_baseline(self.directory), self.baseline)
        with self.assertRaisesRegex(ValueError, "previous release"):
            resolve_release_baseline(self.directory, initial=self.baseline)

    def test_tag_cannot_disagree_with_the_explicit_baseline(self) -> None:
        self.git(
            "tag",
            "-a",
            "v4.0.0",
            "-m",
            f"Benchmark-Baseline: {self.baseline}",
        )
        with self.assertRaisesRegex(ValueError, "disagree"):
            resolve_release_baseline(self.directory, initial="f" * 40, tag="v4.0.0")


if __name__ == "__main__":
    unittest.main()
