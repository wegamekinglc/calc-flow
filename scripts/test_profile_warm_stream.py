"""Fail-closed checks for paired warm-stream profiling."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

if any(importlib.util.find_spec(name) is None for name in ("numpy", "psutil")):
    raise unittest.SkipTest(
        "warm-stream profiling tests require numpy and psutil (dev dependencies)"
    )

import psutil  # noqa: E402

from scripts import profile_warm_stream as profile  # noqa: E402
from scripts import toolkit  # noqa: E402
from scripts.profile_warm_stream import matrix_points, paired_summary  # noqa: E402


def exit_worker(pid, code):
    return profile.Worker(
        SimpleNamespace(
            pid=pid,
            returncode=None,
            stdin=Mock(),
            wait=AsyncMock(return_value=code),
            terminate=Mock(),
        )
    )


async def fresh_workers(workers, directory, measurement):
    with (
        patch.object(profile.Worker, "start", AsyncMock(side_effect=workers)),
        patch.object(profile, "_worker_environments", AsyncMock(return_value=[])),
        patch.object(profile, "_measure_case", measurement),
    ):
        return await profile._fresh_case(
            [{}, {}], {}, 2, False, profile.WorkerPairOptions(Path(directory))
        )


class WarmWorkerExitTests(unittest.IsolatedAsyncioTestCase):
    async def test_installer_is_reaped_when_installation_is_cancelled(self):
        entered = asyncio.Event()
        installer = SimpleNamespace(returncode=None, terminate=Mock(), kill=Mock())

        async def communicate():
            entered.set()
            await asyncio.Event().wait()

        async def reap():
            installer.returncode = -15
            return -15

        installer.communicate = AsyncMock(side_effect=communicate)
        installer.wait = AsyncMock(side_effect=reap)
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "test.whl"
            wheel.touch()
            with patch.object(
                profile.asyncio,
                "create_subprocess_exec",
                AsyncMock(return_value=installer),
            ) as spawn:
                task = asyncio.create_task(
                    profile.Worker.start({"wheel": str(wheel)}, Path(directory))
                )
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        installer.terminate.assert_called_once()
        installer.wait.assert_awaited_once()
        self.assertEqual(installer.returncode, -15)
        self.assertEqual(spawn.await_count, 1)

    async def test_parent_recancellation_still_reaps_both_workers(self):
        entered = asyncio.Queue()
        release = asyncio.Event()

        async def wait():
            entered.put_nowait(None)
            await release.wait()
            return 0

        workers = [exit_worker(41, 0), exit_worker(42, 0)]
        for worker in workers:
            worker.process.wait.side_effect = wait
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.json"
            task = asyncio.create_task(profile._owned_close_workers(workers, path))
            await entered.get()
            await entered.get()
            task.cancel("first cancellation")
            await asyncio.sleep(0)
            task.cancel("second cancellation")
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaisesRegex(asyncio.CancelledError, "first cancellation"):
                await task
            receipts = json.loads(path.read_text())["workers"]
            self.assertEqual([r["pid"] for r in receipts], [41, 42])
            self.assertTrue(all(r["reaped"] and r["exit_code"] == 0 for r in receipts))

    async def test_normal_close_returns_actual_reaped_exit_receipt(self):
        worker = exit_worker(41, 0)
        self.assertEqual(
            await worker.close(),
            {"pid": 41, "exit_code": 0, "reaped": True, "close_result": "normal"},
        )
        worker.process.wait.assert_awaited_once()

    async def test_normal_close_rejects_nonzero_exit(self):
        worker = exit_worker(41, 7)
        with self.assertRaisesRegex(RuntimeError, "close cleanly"):
            await worker.close()

    async def test_close_reaps_worker_when_stdin_close_fails(self):
        worker = exit_worker(41, 0)
        original = OSError("pipe close failed")
        worker.process.stdin.close.side_effect = original
        with (
            patch.object(profile, "stop", AsyncMock()) as stop,
            self.assertRaises(OSError) as caught,
        ):
            await worker.close()
        self.assertIs(caught.exception, original)
        stop.assert_awaited_once_with(worker.process)
        self.assertTrue(worker.close_receipt["reaped"])

    async def test_forced_close_cannot_pass_with_zero_exit(self):
        worker = exit_worker(41, 0)
        worker.process.wait.side_effect = [TimeoutError(), 0]
        with (
            patch.object(profile, "stop", AsyncMock(), create=True),
            self.assertRaisesRegex(RuntimeError, "close cleanly"),
        ):
            await worker.close()

    async def test_fresh_case_persists_both_successful_worker_receipts(self):
        workers = [exit_worker(41, 0), exit_worker(42, 0)]
        with tempfile.TemporaryDirectory() as directory:
            case, environments = await fresh_workers(
                workers, directory, AsyncMock(return_value={})
            )
            receipts = case["worker_receipts"]
            self.assertEqual([r["pid"] for r in receipts], [41, 42])
            self.assertEqual([r["exit_code"] for r in receipts], [0, 0])
            journal = json.loads(Path(case["worker_receipt_path"]).read_text())
            self.assertEqual(journal["workers"], receipts)
            self.assertEqual(environments, [])

    async def test_fresh_case_settles_second_worker_after_first_exit_failure(self):
        workers = [exit_worker(41, 7), exit_worker(42, 0)]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "close cleanly"):
                await fresh_workers(workers, directory, AsyncMock(return_value={}))
            journal = json.loads(
                next(Path(directory).glob("*-workers.json")).read_text()
            )
            self.assertEqual([r["exit_code"] for r in journal["workers"]], [7, 0])
        for worker in workers:
            worker.process.wait.assert_awaited_once()

    async def test_fresh_case_preserves_measurement_error_after_exit_failure(self):
        workers = [exit_worker(41, 7), exit_worker(42, 0)]
        original = ValueError("invalid sample")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                await fresh_workers(workers, directory, AsyncMock(side_effect=original))
            self.assertIs(caught.exception, original)
        for worker in workers:
            worker.process.wait.assert_awaited_once()


class WarmProfileTests(unittest.TestCase):
    def test_benchmark_wheel_name_comes_from_each_source_revision(self) -> None:
        for project_name in ("calc-flow", "calc-flow-python"):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / "pyproject.toml").write_text(
                    f'[project]\nname = "{project_name}"\n', encoding="utf-8"
                )
                distribution = project_name.replace("-", "_")
                wheel = directory / f"{distribution}-4.0.0-cp313-abi3-linux_x86_64.whl"
                wheel.touch()
                (directory / "unrelated_package-4.0.0-py3-none-any.whl").touch()
                self.assertEqual(profile._core_wheel(directory, directory), wheel)

    def test_benchmark_build_rejects_missing_or_ambiguous_core_wheels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "pyproject.toml").write_text(
                '[project]\nname = "calc-flow-python"\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "exactly one core wheel"):
                profile._core_wheel(directory, directory)
            for version in ("4.0.0", "4.0.1"):
                (
                    directory / f"calc_flow_python-{version}-cp313-abi3-win_amd64.whl"
                ).touch()
            with self.assertRaisesRegex(RuntimeError, "exactly one core wheel"):
                profile._core_wheel(directory, directory)

    def test_worker_thread_override_is_explicit_validated_and_immutable(self) -> None:
        original = {"TOKIO_WORKER_THREADS": "4", "PATH": "unchanged"}
        configured = profile._configured_worker_environment(original, 1)
        self.assertEqual(configured["TOKIO_WORKER_THREADS"], "1")
        self.assertEqual(original["TOKIO_WORKER_THREADS"], "4")
        self.assertEqual(
            profile._configured_worker_environment(original, None), original
        )
        for inherited in ({"TOKIO_WORKER_THREADS": "32"}, {}):
            configured = profile._configured_worker_environment(inherited, None)
            self.assertEqual(configured["TOKIO_WORKER_THREADS"], "4")
            self.assertEqual(
                inherited.get("TOKIO_WORKER_THREADS"), "32" if inherited else None
            )
        self.assertEqual(
            profile._configured_worker_environment(original, 4)["TOKIO_WORKER_THREADS"],
            "4",
        )
        for value in (0, -1, True, 1.5, 2, 8, 16, 32):
            with self.assertRaises(ValueError):
                profile._configured_worker_environment(original, value)

    def test_sparse_matrix_is_seeded_and_keeps_layout_explicit(self) -> None:
        args = SimpleNamespace(
            history_rows=[128, 256],
            append_rows=[1, 4],
            append_entities=[1, 4],
            indicators=["rolling_mean"],
            gc_modes=["forced", "normal"],
            seed=7,
        )
        cases = profile.matrix_cases(args)
        self.assertEqual(len(cases), 16)
        self.assertEqual(cases, profile.matrix_cases(args))
        args.seed = 8
        self.assertNotEqual(cases, profile.matrix_cases(args))
        self.assertEqual({config["append_entities"] for config, _ in cases}, {1, 4})
        self.assertEqual({config["history_rows"] for config, _ in cases}, {128, 256})

    def test_original_matrix_never_silently_uses_sparse_timestamps(self) -> None:
        args = SimpleNamespace(
            history_rows=profile.HISTORY_ROWS,
            append_rows=profile.APPEND_ROWS,
            append_entities=None,
            indicators=["rolling_mean", "dual_sma_spread"],
            gc_modes=["forced", "normal"],
            seed=7,
        )
        cases = profile.matrix_cases(args)
        self.assertEqual(len(cases), 28)
        self.assertTrue(all("append_entities" not in config for config, _ in cases))

    def test_phase_summary_preserves_conversion_as_a_subset(self) -> None:
        first = {
            "seconds": 0.010,
            "phases_seconds": {
                "enqueue_to_source_data": 0.001,
                "source_data_to_source_watermark": 0.002,
                "source_watermark_to_sink": 0.003,
                "sink_to_receive": 0.004,
                "to_pyarrow": 0.0005,
            },
        }
        second = {
            "seconds": 0.020,
            "phases_seconds": {
                name: value * 2 for name, value in first["phases_seconds"].items()
            },
        }
        result = profile._phase_summary([first, second])
        self.assertAlmostEqual(result["to_pyarrow"]["p50_seconds"], 0.00075)
        self.assertAlmostEqual(result["sink_to_receive"]["p50_seconds"], 0.006)
        self.assertAlmostEqual(result["sink_to_receive"]["p95_seconds"], 0.0078)
        self.assertEqual(first["phases_seconds"]["to_pyarrow"], 0.0005)

    def test_phase_summary_rejects_invalid_or_double_counted_boundaries(self) -> None:
        phases = {
            "enqueue_to_source_data": 1.0,
            "source_data_to_source_watermark": 2.0,
            "source_watermark_to_sink": 3.0,
            "sink_to_receive": 4.0,
            "to_pyarrow": 0.5,
        }
        for name, value in (
            ("enqueue_to_source_data", -1),
            ("source_data_to_source_watermark", float("nan")),
            ("source_watermark_to_sink", 9),
            ("to_pyarrow", 5),
        ):
            with self.assertRaises(ValueError):
                profile._phase_summary(
                    [
                        {
                            "seconds": 10,
                            "phases_seconds": {**phases, name: value},
                        }
                    ]
                )
        with self.assertRaises(ValueError):
            profile._phase_summary([{"seconds": 10, "phases_seconds": {}}])
        with self.assertRaises(ValueError):
            profile._phase_summary([])

    def test_python_launch_environment_selects_the_current_interpreter(self) -> None:
        supplied = {"PATH": "caller-path", "PYTHONPATH": "caller-modules"}
        with patch.object(
            profile.shutil, "which", return_value=sys.executable
        ) as which:
            environment = profile._python_environment(supplied)
        self.assertEqual(environment["PYTHONPATH"], "caller-modules")
        self.assertEqual(supplied["PATH"], "caller-path")
        self.assertEqual(
            environment["PATH"].split(os.pathsep)[0], str(Path(sys.executable).parent)
        )
        which.assert_called_once_with("python", path=environment["PATH"])

    def test_python_launch_environment_rejects_a_different_interpreter(self) -> None:
        for selected in (None, __file__):
            with (
                patch.object(profile.shutil, "which", return_value=selected),
                self.assertRaisesRegex(ValueError, "current Python interpreter"),
            ):
                profile._python_environment({"PATH": "caller-path"})

    def test_affinity_unavailable_does_not_abort_worker_handshake(self) -> None:
        for error in (psutil.AccessDenied(pid=1), NotImplementedError()):
            process = Mock(cpu_affinity=Mock(side_effect=error))
            with patch.object(psutil, "Process", return_value=process):
                self.assertIsNone(toolkit._cpu_affinity())

    def test_matrix_covers_history_and_increment_scales_without_duplicates(
        self,
    ) -> None:
        points = matrix_points()
        self.assertEqual(len(points), 7)
        self.assertIn((10_240_000, 64), points)
        self.assertIn((1_024_000, 64_000), points)
        self.assertEqual(len(set(points)), len(points))

    def test_paired_summary_preserves_speedup_direction_and_tail(self) -> None:
        result = paired_summary([2, 4, 6, 8], [1, 2, 3, 4], rows=64)
        self.assertEqual(result["paired_speedup_median"], 2)
        self.assertEqual(result["paired_speedup_ci95"], [2, 2])
        self.assertAlmostEqual(result["candidate"]["p95_seconds"], 3.85)

    def test_unpaired_or_invalid_samples_cannot_produce_evidence(self) -> None:
        for baseline, candidate in (
            ([1, 2], [1]),
            ([1, 2], [0, 1]),
            ([1, 2], [float("nan"), 1]),
            ([], []),
            ([1], [1]),
        ):
            with self.assertRaises(ValueError):
                paired_summary(baseline, candidate, rows=64)


class WarmProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_comparison_rejects_different_python_wheels(self) -> None:
        manifests = [
            {"native_sha256": "same", "wheel_sha256": side}
            for side in ("left", "right")
        ]
        args = SimpleNamespace(
            baseline_build=Path("left"),
            candidate_build=Path("right"),
            worker_threads=(4, 1),
        )
        with (
            patch.object(
                profile, "_compatible_builds", AsyncMock(return_value=manifests)
            ),
            self.assertRaisesRegex(ValueError, "same native wheel"),
        ):
            await profile.compare(args)

    async def test_scheduler_comparison_requires_the_exact_declared_thread_override(
        self,
    ) -> None:
        workers = [
            SimpleNamespace(
                request=AsyncMock(
                    return_value={
                        "native_sha256": "same",
                        "tokio_worker_threads": str(count),
                    }
                )
            )
            for count in (4, 1)
        ]
        manifests = [{"native_sha256": "same"}] * 2
        result = await profile._worker_environments(
            workers, manifests, thread_counts=(4, 1)
        )
        self.assertEqual(result[1]["tokio_worker_threads"], "1")
        with self.assertRaisesRegex(ValueError, "thread override"):
            await profile._worker_environments(workers, manifests, thread_counts=(4, 2))
        with self.assertRaisesRegex(ValueError, "default four"):
            await profile._worker_environments(workers, manifests)

    def test_default_worker_environments_require_four_threads(self) -> None:
        valid = [{"tokio_worker_threads": "4"}] * 2
        self.assertEqual(
            profile._allowed_fingerprint_differences(valid, None), {"native_sha256"}
        )
        for count in ("32", None):
            invalid = [{"tokio_worker_threads": count}] * 2
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "default four"),
            ):
                profile._allowed_fingerprint_differences(invalid, None)

    async def test_fresh_case_closes_both_workers_on_measurement_failure(self) -> None:
        workers = [SimpleNamespace(close=AsyncMock()) for _ in range(2)]
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(profile.Worker, "start", AsyncMock(side_effect=workers)),
            patch.object(profile, "_worker_environments", AsyncMock(return_value=[])),
            patch.object(
                profile,
                "_measure_case",
                AsyncMock(side_effect=ValueError("invalid sample")),
            ),
            self.assertRaisesRegex(ValueError, "invalid sample"),
        ):
            await profile._fresh_case(
                [{}, {}], {}, 2, False, profile.WorkerPairOptions(Path(temporary))
            )
        for worker in workers:
            worker.close.assert_awaited_once()

    async def test_worker_fingerprint_rejects_a_different_loaded_native_module(
        self,
    ) -> None:
        workers = [
            SimpleNamespace(request=AsyncMock(return_value={"native_sha256": value}))
            for value in ("baseline", "unexpected")
        ]
        manifests = [{"native_sha256": value} for value in ("baseline", "candidate")]
        with self.assertRaisesRegex(ValueError, "different native module"):
            await profile._worker_environments(workers, manifests)

    async def test_worker_fingerprint_requires_identical_runtime_dependencies(
        self,
    ) -> None:
        workers = [
            SimpleNamespace(
                request=AsyncMock(
                    return_value={
                        "native_sha256": name,
                        "numpy": version,
                        "tokio_worker_threads": "4",
                    }
                )
            )
            for name, version in (("baseline", "1"), ("candidate", "2"))
        ]
        manifests = [{"native_sha256": value} for value in ("baseline", "candidate")]
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            await profile._worker_environments(workers, manifests)

    async def test_matching_worker_fingerprints_keep_optional_affinity(self) -> None:
        environments = [
            {
                "native_sha256": name,
                "numpy": "1",
                "cpu_affinity": None,
                "tokio_worker_threads": "4",
            }
            for name in ("baseline", "candidate")
        ]
        workers = [
            SimpleNamespace(request=AsyncMock(return_value=environment))
            for environment in environments
        ]
        manifests = [{"native_sha256": value} for value in ("baseline", "candidate")]
        self.assertEqual(
            await profile._worker_environments(workers, manifests), environments
        )

    async def test_worker_session_requires_start_and_cancels_unfinished_scenario(
        self,
    ) -> None:
        session = profile.WorkerSession(Path("."))
        for operation in ("sample", "finish"):
            with self.assertRaisesRegex(RuntimeError, "not started"):
                await session.dispatch({"operation": operation, "collect_gc": False})
        with self.assertRaisesRegex(ValueError, "unknown"):
            await session.dispatch({"operation": "invalid"})
        scenario = SimpleNamespace(job=SimpleNamespace(cancel_async=AsyncMock()))
        session.scenario = scenario
        await session.close()
        scenario.job.cancel_async.assert_awaited_once()
        await session.close()
        scenario.job.cancel_async.assert_awaited_once()

    async def test_metadata_process_failure_cannot_produce_a_fingerprint(self) -> None:
        process = SimpleNamespace(
            communicate=AsyncMock(return_value=(b"partial", b"metadata failed")),
            returncode=2,
        )
        with self.assertRaisesRegex(RuntimeError, "metadata failed"):
            await profile._command_output(process)

    async def test_git_metadata_rejects_non_metadata_commands_before_spawn(
        self,
    ) -> None:
        with (
            patch.object(
                asyncio, "create_subprocess_exec", new_callable=AsyncMock
            ) as spawn,
            self.assertRaisesRegex(ValueError, "metadata command"),
        ):
            await profile._git(Path("."), "checkout", "main")
        spawn.assert_not_awaited()

    async def test_git_metadata_uses_a_fixed_executable_and_argument_vector(
        self,
    ) -> None:
        process = SimpleNamespace(
            communicate=AsyncMock(return_value=(b"abc\n", b"")),
            returncode=0,
        )
        with patch.object(
            asyncio,
            "create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as spawn:
            self.assertEqual(await profile._git(Path("."), "rev-parse", "HEAD"), "abc")
        self.assertEqual(spawn.call_args.args, ("git", "rev-parse", "HEAD"))
        self.assertNotIn("shell", spawn.call_args.kwargs)

    async def test_missing_worker_pipes_raise_an_explicit_runtime_error(self) -> None:
        worker = profile.Worker(SimpleNamespace(stdin=None, stdout=None))
        with self.assertRaisesRegex(RuntimeError, "pipes"):
            await worker.request(operation="hello")

    async def test_worker_rejects_a_non_object_response(self) -> None:
        process = SimpleNamespace(
            stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()),
            stdout=SimpleNamespace(readline=AsyncMock(return_value=b"[]\n")),
        )
        with self.assertRaisesRegex(ValueError, "object"):
            await profile.Worker(process).request(operation="hello")


if __name__ == "__main__":
    unittest.main()
