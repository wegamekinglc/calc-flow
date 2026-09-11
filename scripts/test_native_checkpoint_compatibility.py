"""Fail-closed provenance and ownership checks for checkpoint comparison."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.native_checkpoint_compatibility import (
    BASELINE_SHA,
    artifact_hashes,
    assert_equivalent,
    copy_checkpoint,
    require_baseline,
    verify_artifacts,
)


class CheckpointProducerTests(unittest.TestCase):
    def test_candidate_revision_cannot_be_labeled_as_the_base_producer(self) -> None:
        require_baseline({"source_sha": BASELINE_SHA})
        for source_sha in ("f" * 40, "unknown", None):
            with self.subTest(source_sha=source_sha), self.assertRaises(ValueError):
                require_baseline({"source_sha": source_sha})

    def test_recovery_copy_cannot_overwrite_or_modify_the_base_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            checkpoint = fixture / "checkpoint"
            checkpoint.mkdir(parents=True)
            state = checkpoint / "state.arrow"
            state.write_bytes(b"original base state")
            hashes = artifact_hashes(fixture)
            (fixture / "provenance.json").write_text(
                json.dumps(
                    {
                        "producer": {"build": {"source_sha": BASELINE_SHA}},
                        "artifact_hashes": hashes,
                    }
                )
            )
            recovered = root / "recovered"
            copy_checkpoint(fixture, recovered)
            (recovered / "state.arrow").write_bytes(b"candidate state after append")
            self.assertEqual(state.read_bytes(), b"original base state")
            verify_artifacts(fixture, hashes)
            with self.assertRaisesRegex(ValueError, "outside the immutable fixture"):
                copy_checkpoint(fixture, fixture / "new-checkpoint")
            state.write_bytes(b"corrupted base state")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                copy_checkpoint(fixture, root / "must-not-exist")
            self.assertFalse((root / "must-not-exist").exists())

    def test_arrow_comparison_preserves_validity_and_nonfinite_classification(
        self,
    ) -> None:
        import pyarrow as pa

        expected = pa.table(
            {"value": [None, float("nan"), float("inf"), float("-inf"), 1.0]}
        )
        assert_equivalent(expected, expected)
        for changed in (
            [float("nan"), float("nan"), float("inf"), float("-inf"), 1.0],
            [None, None, float("inf"), float("-inf"), 1.0],
            [None, float("nan"), float("-inf"), float("-inf"), 1.0],
            [None, float("nan"), float("inf"), float("-inf"), 1.000001],
        ):
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                assert_equivalent(pa.table({"value": changed}), expected)
        with self.assertRaisesRegex(AssertionError, "schemas or metadata"):
            assert_equivalent(
                expected.replace_schema_metadata({b"source": b"changed"}), expected
            )
        assert_equivalent(
            pa.table({"value": [1.0 + 0.5e-12]}), pa.table({"value": [1.0]})
        )


if __name__ == "__main__":
    unittest.main()
