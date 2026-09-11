"""Fail-closed validation for the independent Native profiling instrument."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from benchmarks.native_memory_profile import prove_release, read_memory_status


class NativeProfileContractTests(unittest.TestCase):
    def test_release_proof_rejects_dev_or_different_loaded_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "_native.so"
            binary.write_bytes(b"exact release binary")
            manifest = {
                "profile": "release",
                "source_sha": "1" * 40,
                "tracked_source_clean": True,
                "native_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            }
            proof = prove_release(manifest, binary)
            self.assertEqual(proof["native_sha256"], manifest["native_sha256"])
            for changed in (
                {**manifest, "profile": "dev"},
                {**manifest, "tracked_source_clean": False},
                {**manifest, "native_sha256": "0" * 64},
                {**manifest, "source_sha": "unknown"},
            ):
                with self.assertRaises(ValueError):
                    prove_release(changed, binary)

    def test_missing_rss_is_not_silently_reported_as_zero(self) -> None:
        self.assertEqual(
            read_memory_status("VmRSS:\t12 kB\nVmHWM:\t20 kB\n"),
            {"rss_bytes": 12 * 1024, "lifetime_peak_rss_bytes": 20 * 1024},
        )
        for text in ("", "VmRSS: 10 kB\n", "VmRSS: 1 MB\nVmHWM: 2 MB\n"):
            with self.assertRaises(ValueError):
                read_memory_status(text)


if __name__ == "__main__":
    unittest.main()
