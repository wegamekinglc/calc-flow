from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.benchmark_suite.release import load_release


class BenchmarkReleaseTests(unittest.TestCase):
    def manifest(self, root: Path) -> tuple[Path, dict]:
        wheel = root / "calc_flow.whl"
        native = b"test native module"
        with ZipFile(wheel, "w") as output:
            output.writestr("calc_flow/_native.abi3.so", native)
        manifest = {
            "contract": "benchmark-release-v1",
            "build_profile": "release",
            "git_clean": True,
            "git_sha": "a" * 40,
            "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "native_sha256": hashlib.sha256(native).hexdigest(),
        }
        path = root / "release.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest

    def test_release_wheel_is_portable_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest = self.manifest(root)
            self.assertEqual(
                Path(load_release(path)["wheel_path"]), root / manifest["wheel"]
            )
            (root / manifest["wheel"]).write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                load_release(path)

    def test_dirty_or_non_release_manifest_is_rejected_before_measurement(self):
        for changes in (
            {"git_clean": False},
            {"git_sha": "HEAD"},
            {"build_profile": "debug"},
            {"wheel": "../escape.whl"},
        ):
            with (
                self.subTest(changes=changes),
                tempfile.TemporaryDirectory() as directory,
            ):
                path, manifest = self.manifest(Path(directory))
                path.write_text(json.dumps({**manifest, **changes}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_release(path)


if __name__ == "__main__":
    unittest.main()
