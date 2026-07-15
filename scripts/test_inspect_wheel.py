from __future__ import annotations

import tempfile
import unittest
from pathlib import Path, PurePath
from shutil import rmtree
from tarfile import open as open_tar
from zipfile import ZipFile

if __package__:
    from scripts.inspect_wheel import (
        inspect_crate,
        inspect_sdist,
        inspect_studio_wheel,
        inspect_wheel,
    )
else:
    from inspect_wheel import (
        inspect_crate,
        inspect_sdist,
        inspect_studio_wheel,
        inspect_wheel,
    )


class InspectWheelTests(unittest.TestCase):
    @staticmethod
    def _fixture_bytes(name: str) -> bytes:
        if PurePath(name).name == "LICENSE":
            return b"Apache License\nVersion 2.0, January 2004\n"
        return b"fixture"

    def _wheel_with(self, *names: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, directory)
        wheel = directory / "calc_flow-2.0.0a1-cp313-abi3-linux_x86_64.whl"
        with ZipFile(wheel, "w") as archive:
            for name in names:
                archive.writestr(name, self._fixture_bytes(name))
        return wheel

    def _archive_with(self, filename: str, *names: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, directory)
        archive_path = directory / filename
        source = directory / "fixture"
        source.write_bytes(b"fixture")
        with open_tar(archive_path, "w:gz") as archive:
            for name in names:
                source.write_bytes(self._fixture_bytes(name))
                archive.add(source, arcname=name)
        return archive_path

    def test_accepts_only_core_package_and_wheel_metadata(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.abi3.so",
            "calc_flow/runtime.py",
            "calc_flow-2.0.0a1.dist-info/METADATA",
            "calc_flow-2.0.0a1.dist-info/licenses/LICENSE",
            "calc_flow-2.0.0a1.dist-info/sboms/calc-flow.cyclonedx.json",
        )

        self.assertEqual(inspect_wheel(wheel), 6)

    def test_accepts_windows_native_module(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.pyd",
            "calc_flow-2.0.0a1.dist-info/METADATA",
            "calc_flow-2.0.0a1.dist-info/licenses/LICENSE",
        )

        self.assertEqual(inspect_wheel(wheel), 4)

    def test_requires_core_wheel_license(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.abi3.so",
            "calc_flow-2.0.0a1.dist-info/METADATA",
        )

        with self.assertRaisesRegex(ValueError, "missing Apache-2.0 license"):
            inspect_wheel(wheel)

    def test_rejects_placeholder_core_wheel_license(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.abi3.so",
            "calc_flow-2.0.0a1.dist-info/METADATA",
        )
        with ZipFile(wheel, "a") as archive:
            archive.writestr(
                "calc_flow-2.0.0a1.dist-info/licenses/LICENSE", b"placeholder"
            )

        with self.assertRaisesRegex(ValueError, "invalid Apache-2.0 license"):
            inspect_wheel(wheel)

    def test_rejects_repository_content(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.abi3.so",
            "calc_flow-2.0.0a1.dist-info/licenses/LICENSE",
            "web-ui/index.html",
        )

        with self.assertRaisesRegex(ValueError, "unexpected wheel entry"):
            inspect_wheel(wheel)

    def test_rejects_packaged_tests_and_executable_project_data(self) -> None:
        for leaked_name in (
            "calc_flow/tests/test_runtime.py",
            "calc_flow/fixtures/project.json",
            "calc_flow/AGENTS.md",
        ):
            with self.subTest(leaked_name=leaked_name):
                wheel = self._wheel_with(
                    "calc_flow/__init__.py",
                    "calc_flow/_native.abi3.so",
                    "calc_flow-2.0.0a1.dist-info/licenses/LICENSE",
                    leaked_name,
                )

                with self.assertRaisesRegex(ValueError, "forbidden wheel entry"):
                    inspect_wheel(wheel)

    def test_requires_native_module(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/runtime.py",
            "calc_flow-2.0.0a1.dist-info/METADATA",
            "calc_flow-2.0.0a1.dist-info/licenses/LICENSE",
        )

        with self.assertRaisesRegex(ValueError, "native module"):
            inspect_wheel(wheel)

    def test_rejects_native_module_prefix_lookalike(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.abi3evil.so",
            "calc_flow-2.0.0a1.dist-info/METADATA",
            "calc_flow-2.0.0a1.dist-info/licenses/LICENSE",
        )

        with self.assertRaisesRegex(ValueError, "native module"):
            inspect_wheel(wheel)

    def test_accepts_studio_wheel_with_license(self) -> None:
        wheel = self._wheel_with(
            "calc_flow_studio/__init__.py",
            "calc_flow_studio/static/index.html",
            "calc_flow_studio-2.0.0a1.dist-info/METADATA",
            "calc_flow_studio-2.0.0a1.dist-info/licenses/LICENSE",
        )

        self.assertEqual(inspect_studio_wheel(wheel), 4)

    def test_rejects_studio_wheel_without_license(self) -> None:
        wheel = self._wheel_with(
            "calc_flow_studio/__init__.py",
            "calc_flow_studio/static/index.html",
            "calc_flow_studio-2.0.0a1.dist-info/METADATA",
        )

        with self.assertRaisesRegex(ValueError, "missing Apache-2.0 license"):
            inspect_studio_wheel(wheel)

    def test_accepts_sdist_with_license_and_no_frozen_python(self) -> None:
        sdist = self._archive_with(
            "calc_flow-2.0.0a1.tar.gz",
            "calc_flow-2.0.0a1/LICENSE",
            "calc_flow-2.0.0a1/Cargo.lock",
            "calc_flow-2.0.0a1/pyproject.toml",
            "calc_flow-2.0.0a1/crates/calc-flow/Cargo.toml",
            "calc_flow-2.0.0a1/crates/calc-flow/src/lib.rs",
            "calc_flow-2.0.0a1/crates/calc-flow-python/Cargo.toml",
            "calc_flow-2.0.0a1/python/calc_flow/__init__.py",
        )

        self.assertEqual(inspect_sdist(sdist), 7)

    def test_rejects_sdist_without_license(self) -> None:
        sdist = self._archive_with(
            "calc_flow-2.0.0a1.tar.gz",
            "calc_flow-2.0.0a1/Cargo.lock",
            "calc_flow-2.0.0a1/crates/calc-flow/src/lib.rs",
        )

        with self.assertRaisesRegex(ValueError, "missing Apache-2.0 license"):
            inspect_sdist(sdist)

    def test_rejects_sdist_without_essential_build_content(self) -> None:
        sdist = self._archive_with(
            "calc_flow-2.0.0a1.tar.gz",
            "calc_flow-2.0.0a1/LICENSE",
            "calc_flow-2.0.0a1/Cargo.lock",
            "calc_flow-2.0.0a1/crates/calc-flow/Cargo.toml",
            "calc_flow-2.0.0a1/crates/calc-flow/src/lib.rs",
            "calc_flow-2.0.0a1/crates/calc-flow-python/Cargo.toml",
            "calc_flow-2.0.0a1/python/calc_flow/__init__.py",
        )

        with self.assertRaisesRegex(ValueError, "missing sdist entries"):
            inspect_sdist(sdist)

    def test_accepts_crate_with_license_and_no_tests(self) -> None:
        crate = self._archive_with(
            "calc-flow-2.0.0-alpha.1.crate",
            "calc-flow-2.0.0-alpha.1/LICENSE",
            "calc-flow-2.0.0-alpha.1/Cargo.toml",
            "calc-flow-2.0.0-alpha.1/src/lib.rs",
        )

        self.assertEqual(inspect_crate(crate), 3)

    def test_rejects_crate_tests(self) -> None:
        crate = self._archive_with(
            "calc-flow-2.0.0-alpha.1.crate",
            "calc-flow-2.0.0-alpha.1/LICENSE",
            "calc-flow-2.0.0-alpha.1/Cargo.toml",
            "calc-flow-2.0.0-alpha.1/src/lib.rs",
            "calc-flow-2.0.0-alpha.1/tests/config.rs",
        )

        with self.assertRaisesRegex(ValueError, "forbidden crate entry"):
            inspect_crate(crate)

    def test_rejects_crate_without_license(self) -> None:
        crate = self._archive_with(
            "calc-flow-2.0.0-alpha.1.crate",
            "calc-flow-2.0.0-alpha.1/Cargo.toml",
            "calc-flow-2.0.0-alpha.1/src/lib.rs",
        )

        with self.assertRaisesRegex(ValueError, "missing Apache-2.0 license"):
            inspect_crate(crate)

    def test_rejects_crate_without_essential_source(self) -> None:
        crate = self._archive_with(
            "calc-flow-2.0.0-alpha.1.crate",
            "calc-flow-2.0.0-alpha.1/LICENSE",
            "calc-flow-2.0.0-alpha.1/Cargo.toml",
        )

        with self.assertRaisesRegex(ValueError, "missing crate entries"):
            inspect_crate(crate)

    def test_rejects_placeholder_crate_license(self) -> None:
        crate = self._archive_with(
            "calc-flow-2.0.0-alpha.1.crate",
            "calc-flow-2.0.0-alpha.1/LICENSE",
            "calc-flow-2.0.0-alpha.1/Cargo.toml",
            "calc-flow-2.0.0-alpha.1/src/lib.rs",
        )
        source = crate.parent / "fixture"
        source.write_bytes(b"placeholder")
        with open_tar(crate, "w:gz") as archive:
            archive.add(source, arcname="calc-flow-2.0.0-alpha.1/LICENSE")
            archive.add(source, arcname="calc-flow-2.0.0-alpha.1/Cargo.toml")
            archive.add(source, arcname="calc-flow-2.0.0-alpha.1/src/lib.rs")

        with self.assertRaisesRegex(ValueError, "invalid Apache-2.0 license"):
            inspect_crate(crate)


if __name__ == "__main__":
    unittest.main()
