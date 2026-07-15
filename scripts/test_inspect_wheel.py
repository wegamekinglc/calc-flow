from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile

if __package__:
    from scripts.inspect_wheel import inspect_wheel
else:
    from inspect_wheel import inspect_wheel


class InspectWheelTests(unittest.TestCase):
    def _wheel_with(self, *names: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(rmtree, directory)
        wheel = directory / "calc_flow-2.0.0a1-cp313-abi3-linux_x86_64.whl"
        with ZipFile(wheel, "w") as archive:
            for name in names:
                archive.writestr(name, b"fixture")
        return wheel

    def test_accepts_only_core_package_and_wheel_metadata(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.abi3.so",
            "calc_flow/runtime.py",
            "calc_flow-2.0.0a1.dist-info/METADATA",
            "calc_flow-2.0.0a1.dist-info/sboms/calc-flow.cyclonedx.json",
        )

        self.assertEqual(inspect_wheel(wheel), 5)

    def test_accepts_windows_native_module(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.pyd",
            "calc_flow-2.0.0a1.dist-info/METADATA",
        )

        self.assertEqual(inspect_wheel(wheel), 3)

    def test_rejects_repository_content(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.abi3.so",
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
                    leaked_name,
                )

                with self.assertRaisesRegex(ValueError, "forbidden wheel entry"):
                    inspect_wheel(wheel)

    def test_requires_native_module(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/runtime.py",
            "calc_flow-2.0.0a1.dist-info/METADATA",
        )

        with self.assertRaisesRegex(ValueError, "native module"):
            inspect_wheel(wheel)

    def test_rejects_native_module_prefix_lookalike(self) -> None:
        wheel = self._wheel_with(
            "calc_flow/__init__.py",
            "calc_flow/_native.abi3evil.so",
            "calc_flow-2.0.0a1.dist-info/METADATA",
        )

        with self.assertRaisesRegex(ValueError, "native module"):
            inspect_wheel(wheel)


if __name__ == "__main__":
    unittest.main()
