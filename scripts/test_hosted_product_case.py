from __future__ import annotations

import unittest
from datetime import UTC

from scripts import hosted_duration_sum as evidence


class BenchCase(unittest.TestCase):
    def test_single_duration_case_uses_one_native_worker(self):
        from scripts.benchmark_suite.catalog import worker_thread_count

        case = evidence.case()
        self.assertEqual(case["id"], "python/public-duration-sum-60/fresh-stream")
        self.assertEqual(case["rows"], 100_000)
        self.assertEqual(worker_thread_count(case), 1)

    def test_deadline_counts_prior_setup_and_keeps_worker_cleanup_time(self):
        self.assertEqual(evidence.remaining_seconds(100, 120), 105 - 20 - 45)
        with self.assertRaises(TimeoutError):
            evidence.remaining_seconds(100, 175)

    def test_only_sixty_second_sum_reconstructs_open_boundary_and_nulls(self):
        import asyncio
        from datetime import datetime, timedelta

        import pyarrow as pa

        from benchmarks.public_duration_sum import (
            INPUT,
            declaration,
            expected,
            observe,
            plan,
            validate,
        )

        base = datetime(2026, 1, 1, tzinfo=UTC)
        source = pa.table(
            {
                "event_time": [
                    base + timedelta(seconds=s) for s in (0, 0, 29, 30, 60, 90)
                ],
                "sequence": list(range(6)),
                "symbol": ["a", "z", "a", "a", "a", "a"],
                "price": [1.0, None, 3.0, 4.0, None, 2.0],
            },
            schema=INPUT,
        )
        oracle = expected(source)
        self.assertEqual(oracle.column_names, ["sequence", "symbol", "sum_60"])
        self.assertEqual(oracle["sum_60"].to_pylist(), [1.0, None, 4.0, 8.0, 7.0, 2.0])
        proof = plan(declaration())
        self.assertEqual([o["frame"]["micros"] for o in proof["outputs"]], [60_000_000])
        _, outputs = asyncio.run(observe((source.slice(0, 2), source.slice(2))))
        self.assertEqual(validate(outputs, oracle), [6, 1])
        wrong = pa.array([1.0, None, 4.0, 8.0, 8.0, 2.0])
        with self.assertRaises(ValueError):
            validate([oracle.set_column(2, oracle.schema.field(2), wrong)], oracle)

    def test_modified_fixture_refuses_collection(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duration.arrow"
            path.write_bytes(b"different IPC bytes")
            with self.assertRaisesRegex(ValueError, "fixture"):
                evidence.verify_fixture(path)

    def test_exact_download_fills_immutable_wheel_and_native_receipt(self):
        import hashlib
        import tempfile
        import zipfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "fixture.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("calc_flow/_native.abi3.so", b"native fixture")
            receipt = {"checkout": "fixture"}
            actual = evidence.download_receipt(receipt, wheel)
            self.assertEqual(receipt, {"checkout": "fixture"})
            self.assertEqual(actual["wheel"], wheel.name)
            self.assertEqual(
                actual["wheel_sha256"], hashlib.sha256(wheel.read_bytes()).hexdigest()
            )
            self.assertEqual(
                actual["native_sha256"], hashlib.sha256(b"native fixture").hexdigest()
            )

    def test_exact_job_log_artifact_and_product_source_are_required(self):
        from unittest.mock import patch

        ref = {
            "workflow_run": 3,
            "job": 7,
            "requested_sha": "a" * 40,
            "checkout": "c" * 40,
            "tree": "d" * 40,
        }
        artifact = {
            "id": 10,
            "name": "python-distributions",
            "expired": False,
            "digest": "sha256:" + "e" * 64,
            "workflow_run": {"id": 3, "head_sha": ref["requested_sha"]},
        }
        run = {
            "id": 3,
            "head_sha": ref["requested_sha"],
            "path": ".github/workflows/ci-linux.yml",
            "run_attempt": 1,
        }
        job = {
            "id": 7,
            "name": "Build + isolated wheel smoke test",
            "conclusion": "success",
        }
        log = "\n".join(
            [
                "Image: ubuntu-26.04",
                "Version: same",
                "git log -1 --format=%H",
                "time " + ref["checkout"],
                "uv build --wheel --config-setting "
                "'build-args=--features pyo3/abi3-py313'",
                "Finished `release` profile [optimized]",
                "Artifact python-distributions successfully finalized. Artifact ID 10",
            ]
        )
        import json

        records = {
            "actions/runs/3": run,
            "actions/runs/3/attempts/1/jobs?per_page=100": {"jobs": [job]},
            "actions/jobs/7/logs": log,
            "actions/artifacts/10": artifact,
            "git/commits/" + ref["checkout"]: {"tree": {"sha": ref["tree"]}},
        }

        def api(_repo, path):
            value = records[path]
            return (value if isinstance(value, str) else json.dumps(value)).encode()

        with patch.object(evidence, "_api", api):
            good = evidence.ci_receipt("owner/repo", ref)
            self.assertEqual(
                (good["artifact"], good["git_sha"], good["image_version"]),
                (10, ref["checkout"], "same"),
            )
            for bad in (
                {**ref, "job": 8},
                {**ref, "tree": "f" * 40},
                {**ref, "checkout": "d" * 40},
            ):
                with self.subTest(ref=bad), self.assertRaises(ValueError):
                    evidence.ci_receipt("owner/repo", bad)
