from __future__ import annotations

import asyncio
import json
import os
import subprocess  # nosec B404  # fixed workflow step and local fixtures
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class ContractTests(unittest.TestCase):
    def test_plan_pins_original_wheels_one_workload_and_five_cases(self):
        from scripts.groupby_diagnostic import issue406

        plan = issue406.plan()
        self.assertEqual(plan["case"]["rows"], 10_000_000)
        self.assertEqual(plan["case"]["scenario"], "group_by")
        self.assertEqual(plan["comparisons"], [["A", "A"], ["B", "B"], ["A", "B"]])
        self.assertEqual((plan["rounds"], plan["pairs"]), (2, 10))
        self.assertEqual(plan["profile_sides"], ["A", "B"])
        self.assertEqual(plan["budget_seconds"], 600)
        self.assertEqual(plan["build_seconds"], 0)
        self.assertEqual(plan["source_run"], 38178694560)
        self.assertEqual(
            plan["checkpoint"], "enabled/24h; timer excludes terminal capture"
        )

    def test_release_rejects_current_main_or_a_rebuilt_native(self):
        from scripts.groupby_diagnostic import issue406

        expected = issue406.SEALS["A"]
        for patch in ({}, {"git_sha": "f" * 40}, {"native_sha256": "e" * 64}):
            with mock.patch.object(
                issue406, "load_release", return_value=expected | patch
            ):
                if patch:
                    with self.assertRaises(ValueError):
                        issue406.sealed("A", Path("unused"))
                else:
                    self.assertEqual(issue406.sealed("A", Path("unused")), expected)

    def test_raw_profiles_require_native_addresses_and_reject_loss(self):
        from scripts.groupby_diagnostic.issue406_profile import validate_trace

        decoded = "7f1234 [unknown] (/tmp/site/calc_flow/_native.abi3.so)"
        validate_trace("PERF_RECORD_SAMPLE", decoded)
        for raw, script in (
            ("PERF_RECORD_LOST_SAMPLES", decoded),
            ("PERF_RECORD_THROTTLE", decoded),
            ("PERF_RECORD_SAMPLE", "python-only"),
            ("PERF_RECORD_SAMPLE", "warning: /tmp/calc_flow/_native.abi3.so"),
            ("PERF_RECORD_SAMPLE", "0 [unknown] (/tmp/calc_flow/_native.abi3.so)"),
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_trace(raw, script)

    def test_worker_refuses_another_case_before_dispatch(self):
        from scripts.groupby_diagnostic import issue406

        request = json.dumps({"operation": "prepare", "case": {"rows": 1}})
        with (
            mock.patch.object(issue406.sys, "stdin", [request]),
            mock.patch.object(issue406, "dispatch") as dispatch,
            self.assertRaisesRegex(ValueError, "fixed 10M"),
        ):
            issue406.worker(Path("unused"))
        dispatch.assert_not_called()


class WorkflowTests(unittest.TestCase):
    def test_failed_collection_preserves_exit_logs_and_restores_permissions(self):
        source = (
            Path(__file__).resolve().parents[1] / ".github/workflows/benchmarks.yml"
        ).read_text()
        step = source.split(
            "      - name: Run fixed controls and original-native address profiles\n", 1
        )[1].split("      - ", 1)[0]
        script = "\n".join(
            line[10:] for line in step.split("        run: |\n", 1)[1].splitlines()
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "target/diagnostic-venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(
                '#!/bin/bash\nprintf "out\\n"\nprintf "err\\n" >&2\nexit 42\n'
            )
            python.chmod(0o755)
            bin_path = root / "bin"
            bin_path.mkdir()
            sudo = bin_path / "sudo"
            sudo.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> sudo.log\n')
            sudo.chmod(0o755)
            result = subprocess.run(  # nosec B603  # checked-in shell step; harmless sudo fixture
                ["/bin/bash", "-e", "-o", "pipefail", "-c", script],
                cwd=root,
                env={
                    **os.environ,
                    "PATH": f"{bin_path}:{os.environ['PATH']}",
                    "ISSUE406_PERF": "/bin/true",
                },
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 42)
            output = root / "target/issue406-evidence"
            self.assertEqual((output / "stdout.log").read_text(), "out\n")
            self.assertEqual((output / "stderr.log").read_text(), "err\n")
            self.assertEqual((output / "exit-code.txt").read_text(), "42\n")
            previous = (output / "previous-paranoid.txt").read_text().strip()
            self.assertEqual(
                (root / "sudo.log").read_text().splitlines(),
                [
                    "sysctl -w kernel.perf_event_paranoid=1",
                    f"sysctl -w kernel.perf_event_paranoid={previous}",
                ],
            )


class OwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_during_worker_termination_waits_for_exit(self):
        from scripts.groupby_diagnostic.runtime import AuditWorker

        waiting, release = asyncio.Event(), asyncio.Event()
        calls = iter((True, False))
        process = SimpleNamespace(
            pid=123,
            returncode=None,
            terminate=mock.Mock(),
            stdin=SimpleNamespace(close=mock.Mock()),
        )

        async def wait():
            if next(calls):
                raise TimeoutError
            waiting.set()
            await release.wait()
            process.returncode = -15
            return -15

        process.wait = wait
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = AuditWorker(process, (root / "stderr.log").open("wb"))
            task = asyncio.create_task(worker.close())
            await waiting.wait()
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(process.returncode, -15)
            self.assertEqual(
                json.loads((root / "exit.json").read_text())["exit_code"], -15
            )
            self.assertTrue(worker.log.closed)

    async def test_cancellation_during_worker_spawn_settles_the_created_process(self):
        from scripts.groupby_diagnostic import runtime

        started, release = asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(
            returncode=None, terminate=mock.Mock(), wait=mock.AsyncMock()
        )

        async def spawn(*args, **kwargs):
            started.set()
            await release.wait()
            return process

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                runtime.asyncio, "create_subprocess_exec", side_effect=spawn
            ):
                task = asyncio.create_task(
                    runtime.AuditWorker.start(Path(temporary), Path(temporary))
                )
                await started.wait()
                task.cancel()
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            process.terminate.assert_called_once()
            process.wait.assert_awaited_once()

    async def test_permission_failure_prevents_sampling_and_closes_worker(self):
        from scripts.groupby_diagnostic import issue406_profile as profile

        worker = SimpleNamespace(
            process=SimpleNamespace(pid=123), close=mock.AsyncMock()
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(profile.Issue406Worker, "start", return_value=worker),
                mock.patch.object(profile, "_prepare", autospec=True),
                mock.patch.object(
                    profile, "preflight", side_effect=ValueError("denied")
                ),
                mock.patch.object(
                    profile, "record", new_callable=mock.AsyncMock
                ) as record,
                self.assertRaisesRegex(ValueError, "denied"),
            ):
                await profile.collect({}, root, root, "/usr/bin/perf")
            record.assert_not_called()
            worker.close.assert_awaited_once()

    async def test_bad_host_is_reported_without_installing_or_measuring(self):
        from scripts.groupby_diagnostic import issue406

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(issue406, "host", return_value={}),
                mock.patch.object(
                    issue406, "require_host", side_effect=ValueError("host")
                ),
                mock.patch.object(
                    issue406, "collect", new_callable=mock.AsyncMock
                ) as collect,
            ):
                self.assertEqual(await issue406.run(root, root / "releases", "perf"), 1)
            collect.assert_not_called()
            outcome = json.loads((root / "outcome.json").read_text())
            self.assertIn("host", outcome["error"])
            self.assertFalse(outcome["measurement_complete"])


if __name__ == "__main__":
    unittest.main()
