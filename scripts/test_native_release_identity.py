"""Integrity guards for exact release package identity."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from benchmarks.native_release_identity import file_hash, release_identity


def release_fixture(root):
    package = root / "calc_flow"
    package.mkdir()
    initializer = package / "__init__.py"
    initializer.write_bytes(b"release package fixture\n")
    native_path = package / "_native.so"
    native_path.write_bytes(b"native file fixture; never loaded\n")
    wheel = root / "release.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("calc_flow/", b"")
        for path in (initializer, native_path):
            archive.write(path, "calc_flow/" + path.name)
        archive.writestr("release.dist-info/METADATA", b"fixture metadata\n")
    build = {
        "profile": "release",
        "tracked_source_clean": True,
        "source_sha": "a" * 40,
        "native_sha256": file_hash(native_path),
        "wheel": str(wheel),
        "wheel_sha256": file_hash(wheel),
    }
    record = root / "build.json"
    record.write_text(json.dumps(build))
    package_module = ModuleType("calc_flow")
    package_module.__file__ = str(initializer)
    native_module = ModuleType("calc_flow._native")
    native_module.__file__ = str(native_path)
    package_module._native = native_module
    return (
        record,
        build,
        {
            "calc_flow": package_module,
            "calc_flow._native": native_module,
        },
    )


class ReleaseIdentityTests(unittest.TestCase):
    def test_file_hash_observes_empty_unicode_and_changed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "摘要.bin"
            for content in (b"", b"original", b"changed"):
                path.write_bytes(content)
                self.assertEqual(file_hash(path), hashlib.sha256(content).hexdigest())

    def test_identity_binds_native_wheel_package_and_helper_without_mutating_build(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            record, build, modules = release_fixture(Path(directory))
            original = copy.deepcopy(build)
            with patch.dict(sys.modules, modules):
                identity = release_identity(record)
            self.assertEqual(identity["build"], original)
            self.assertEqual(identity["native_sha256"], build["native_sha256"])
            self.assertEqual(identity["python_executable"], sys.executable)
            self.assertEqual(
                identity["python_package"],
                str(Path(modules["calc_flow"].__file__).parent),
            )
            self.assertEqual(
                identity["instrument_sha256"],
                file_hash(Path(sys.modules[release_identity.__module__].__file__)),
            )
            self.assertEqual(json.loads(record.read_text()), original)

    def test_dev_dirty_unknown_source_and_stale_native_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            record, build, modules = release_fixture(Path(directory))
            for changed in (
                {"profile": "dev"},
                {"tracked_source_clean": False},
                {"source_sha": "unknown"},
                {"native_sha256": "0" * 64},
            ):
                record.write_text(json.dumps({**build, **changed}))
                with (
                    self.subTest(changed=changed),
                    patch.dict(sys.modules, modules),
                    self.assertRaisesRegex(ValueError, "matching release binary"),
                ):
                    release_identity(record)

    def test_changed_wheel_is_rejected_before_reading_package_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            record, build, modules = release_fixture(Path(directory))
            Path(build["wheel"]).write_bytes(b"corrupted wheel")
            with (
                patch.dict(sys.modules, modules),
                self.assertRaisesRegex(ValueError, "wheel does not match"),
            ):
                release_identity(record)

    def test_native_and_python_modules_must_belong_to_one_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, _, modules = release_fixture(root)
            original = Path(modules["calc_flow._native"].__file__)
            other = root / "different-native.so"
            other.write_bytes(original.read_bytes())
            modules["calc_flow._native"].__file__ = str(other)
            with (
                patch.dict(sys.modules, modules),
                self.assertRaisesRegex(ValueError, "different packages"),
            ):
                release_identity(record)

    def test_loaded_python_contents_must_match_the_release_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            record, _, modules = release_fixture(Path(directory))
            Path(modules["calc_flow"].__file__).write_bytes(b"changed package\n")
            with (
                patch.dict(sys.modules, modules),
                self.assertRaisesRegex(ValueError, "loaded package differs"),
            ):
                release_identity(record)


if __name__ == "__main__":
    unittest.main()
