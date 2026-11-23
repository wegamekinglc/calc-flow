from __future__ import annotations

import unittest

from scripts import sparse_changelog_ci as ci
from scripts import sparse_changelog_wheels as wheels


class SparseChangelogCiTests(unittest.TestCase):
    def test_build_proof_rejects_wrong_workflow_commit_or_profile(self):
        sha = "a" * 40
        checkout = "b" * 40
        artifact = {
            "id": 7,
            "name": "python-distributions",
            "expired": False,
            "workflow_run": {"id": 8, "head_sha": sha},
        }
        run = {
            "id": 8,
            "path": ".github/workflows/ci-linux.yml",
            "head_sha": sha,
            "event": "pull_request",
        }
        log = "\n".join(
            [
                "Image: ubuntu-26.04",
                "Version: 20261116.202.1",
                "git log -1 --format=%H",
                f"2026-11-23T00:00:00Z {checkout}",
                "uv build --wheel --config-setting "
                "'build-args=--features pyo3/abi3-py313'",
                "Finished `release` profile [optimized]",
                "Artifact python-distributions successfully finalized. Artifact ID 7",
            ]
        )
        profile = wheels.build_proof(artifact, run, sha, log)
        self.assertEqual(profile["checkout"], checkout)
        for changed, modified_run, badlog in [
            ({**artifact, "expired": True}, run, log),
            (artifact, {**run, "path": ".github/workflows/other.yml"}, log),
            (artifact, {**run, "head_sha": "c" * 40}, log),
            (artifact, run, log.replace("release", "dev")),
            (artifact, run, log.replace("abi3-py313", "abi3-py39")),
            (artifact, run, log.replace("Artifact ID 7", "Artifact ID 70")),
        ]:
            with (
                self.subTest(changed=changed, modified_run=modified_run),
                self.assertRaises(ValueError),
            ):
                wheels.build_proof(changed, modified_run, sha, badlog)

    def test_pressure_guard_checks_startup_paging_and_smt(self):
        counters = {"pgmajfault": 10, "pswpin": 1, "pswpout": 2}
        row = ci.interval(
            counters,
            {**counters, "pgmajfault": 349},
            [1.0, 0.0],
            29 * 1024**3,
            0,
            (1,),
            preflight=False,
        )
        self.assertFalse(row["suitable"])
        self.assertEqual(row["major_faults"], 339)
        self.assertTrue(
            ci.interval(
                counters, counters, [99, 0], 12 * 1024**3, 0, (1,), preflight=False
            )["suitable"]
        )
        self.assertFalse(
            ci.interval(
                counters, counters, [90, 70], 12 * 1024**3, 0, (1,), preflight=False
            )["suitable"]
        )

    def test_sparse_worker_uses_one_tokio_worker_and_does_not_enter_catalog(self):
        from scripts.benchmark_suite.catalog import engine_cases, worker_thread_count

        self.assertEqual(worker_thread_count(ci.CASE), 1)
        self.assertNotIn(ci.CASE["id"], {case["id"] for case in engine_cases(100)})

    def test_sparse_job_is_manual_read_only_and_preserves_failed_evidence(self):
        definition = (ci.ROOT / ".github/workflows/benchmarks.yml").read_text()
        self.assertIn("  sparse-changelog:\n", definition)
        job = definition.split("  sparse-changelog:\n", 1)[1].split(
            "\n  suite-recovery:", 1
        )[0]
        for expected in [
            "inputs.mode == 'sparse-changelog'",
            "runs-on: ubuntu-latest",
            "timeout-minutes: 10",
            "persist-credentials: false",
            "inputs.baseline_wheel_artifact",
            "inputs.candidate_wheel_artifact",
            "scripts.sparse_changelog_ci seal",
            "scripts.sparse_changelog_ci measure",
            "if: always()",
            "resources.jsonl",
            "measurement/",
        ]:
            self.assertIn(expected, job)
        self.assertNotIn("cargo ", job)

    def test_sampling_timeout_settles_owned_task_and_records_stop(self):
        import asyncio
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        settled = []

        async def pending(_root):
            try:
                await asyncio.Event().wait()
            finally:
                settled.append(True)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(ci, "_measure", pending),
        ):
            root = Path(directory)
            with self.assertRaises(TimeoutError):
                asyncio.run(ci.measure(root, seconds=0.01))
            self.assertEqual(settled, [True])
            record = json.loads((root / "outcome.json").read_text())
            self.assertEqual(record["state"], "stopped")
            self.assertIn("TimeoutError", record["error"])
            self.assertFalse((root / "result.json").exists())

    def test_pressure_in_final_interval_disqualifies_completed_measurement(self):
        import asyncio
        from unittest.mock import patch

        async def exercise():
            finished = asyncio.Event()

            async def late_pressure(_path, cpu):
                await finished.wait()
                raise ValueError("paging in final interval")

            async def observed(*_):
                finished.set()
                return {"status": "ok"}

            with (
                patch.object(ci, "_resource", late_pressure),
                patch.object(ci, "measure_case", observed),
            ):
                with self.assertRaises(ExceptionGroup) as caught:
                    await ci.guarded_measure(ci.ROOT, 0, {}, {}, {})
                self.assertIn("paging in final interval", repr(caught.exception))

        asyncio.run(exercise())

    def test_zip_extraction_requires_one_verified_linux_cp313_wheel(self):
        import hashlib
        import io
        import tempfile
        import zipfile
        from pathlib import Path

        def artifact(*members):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                for name in members:
                    archive.writestr(name, b"fixture")
            content = stream.getvalue()
            return content, "sha256:" + hashlib.sha256(content).hexdigest()

        wheel = "calc_flow_python-2026.11.14-cp313-abi3-linux_x86_64.whl"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid, digest = artifact(wheel, "source.tar.gz")
            self.assertEqual(
                wheels._wheel(valid, digest, root).read_bytes(), b"fixture"
            )
            for content, digest in (
                (valid, "sha256:bad"),
                artifact("../" + wheel),
                artifact(wheel.replace("cp313", "cp39")),
                artifact(wheel, wheel.replace("11.14", "11.15")),
            ):
                with self.subTest(digest=digest), self.assertRaises(ValueError):
                    wheels._wheel(content, digest, root)

    def test_api_rejects_unknown_endpoints_before_starting_gh(self):
        from unittest.mock import patch

        with patch("subprocess.check_output") as subprocess_run:
            for repository, path in (
                ("--hostname", "actions/artifacts/7"),
                ("owner/repo", "../other"),
                ("owner/repo", "actions/runs/7/cancel"),
                ("owner/repo", "git/commits/branch-name"),
            ):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    wheels._api(repository, path)
            subprocess_run.assert_not_called()
