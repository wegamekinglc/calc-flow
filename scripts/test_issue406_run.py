from __future__ import annotations

import asyncio
import json
import sys
import unittest
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite.issue406_acceptance import acceptance_inventory
from scripts.benchmark_suite.issue406_run import (
    AcceptanceRunner,
    collect,
    collection_groups,
    run,
)


class FakeRunner:
    def __init__(self, *, fail_at: int | None = None):
        self.calls = []
        self.closed = 0
        self.fail_at = fail_at

    async def prepare(self, group, side, round_index):
        self.calls.append(("warmup", group[0].id, side, round_index, None))
        return {case.id: {"warmup": True} for case in group}

    async def sample(self, group, side, round_index, index):
        self.calls.append(("sample", group[0].id, side, round_index, index))
        if len(self.calls) == self.fail_at:
            raise asyncio.CancelledError()
        return {case.id: {"sample": index} for case in group}

    async def finish_round(self):
        self.closed += 1


async def settle_fixture_runner(runner, task):
    if task is not None and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    for process in tuple(runner.processes.values()):
        if process.returncode is None:
            process.kill()
        await process.communicate()
    await runner.settle()


class AcceptanceDriverTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_join_preflight_records_refusal_control_once(self):
        from scripts.test_benchmark_native_acceptance import (
            dictionary_refusal_observation,
        )
        from scripts.toolkit import sha256_file

        with TemporaryDirectory() as raw:
            root = Path(raw)
            binary = root / "native"
            binary.write_bytes(b"sealed candidate binary fixture")
            identity = {"path": str(binary), "sha256": sha256_file(binary)}
            runner = AcceptanceRunner(
                {
                    "cpus": [0],
                    "releases": {"candidate": {"product_commit": "c" * 40}},
                    "native_binaries": {
                        "candidate": {"stream_join_perf/enabled": identity}
                    },
                },
                root,
                0,
            )
            case = next(
                case
                for case in acceptance_inventory()
                if case.workload == "join_capture"
            )
            control = AsyncMock(return_value=dictionary_refusal_observation())
            with (
                patch.object(
                    runner, "_run_dictionary_refusal_control", control, create=True
                ),
                patch.object(
                    runner, "sample", AsyncMock(return_value={"warmup": True})
                ),
            ):
                for side, round_index in (
                    ("candidate", 0),
                    ("candidate", 1),
                    ("baseline", 0),
                ):
                    await runner.prepare((case,), side, round_index)
            self.assertEqual(control.await_count, 1)
            observation = json.loads(
                (root / "correctness" / "join-dictionary-refusal.json").read_text()
            )
            self.assertEqual(observation["product_commit"], "c" * 40)
            self.assertEqual(observation["binary_sha256"], identity["sha256"])
            self.assertEqual(observation["control"], dictionary_refusal_observation())

    def test_native_replay_uses_the_exact_source_sample_trace(self):
        with TemporaryDirectory() as raw_root:
            runner = AcceptanceRunner({}, Path(raw_root), 0)
            case = next(
                case
                for case in acceptance_inventory()
                if case.row == 7 and case.workload == "asof" and case.mode == "off"
            )
            raw = {
                "match_identity": {"project_sha256": "a" * 64},
                "dispatch_trace": {"events": []},
            }

            def normalized(_group, _side, value, observation_id, raw_digest):
                return {
                    case.id: {
                        **value,
                        "observation_id": observation_id,
                        "raw_observation_sha256": raw_digest,
                    }
                }

            with patch.object(runner, "_normalize", side_effect=normalized):
                result = runner._persist((case,), "candidate", 0, 3, raw)[case.id]
            native = replace(case, workload="asof_operator")
            path = runner.dispatch_trace_path(native, "candidate", 0, 3)
            self.assertEqual(str(path), result["dispatch_trace_envelope_path"])
            envelope = json.loads(path.read_text())
            self.assertEqual(
                envelope["source_observation_id"], result["observation_id"]
            )
            with self.assertRaises(FileNotFoundError):
                runner.dispatch_trace_path(native, "candidate", 0, 4)
            path.write_text(
                json.dumps({**envelope, "source_observation_id": "another-sample"})
            )
            with self.assertRaisesRegex(ValueError, "trace"):
                runner.dispatch_trace_path(native, "candidate", 0, 3)

    def test_row7_selects_only_its_declared_instrumented_release(self):
        runner = AcceptanceRunner(
            {
                "releases": {"candidate": {"native_sha256": "production"}},
                "row7_releases": {"candidate": {"native_sha256": "traced"}},
            },
            Path("unused"),
            0,
        )
        for case in acceptance_inventory():
            self.assertEqual(
                runner.release(case, "candidate")["native_sha256"],
                "traced" if case.row == 7 else "production",
            )
        runner.plan.pop("row7_releases")
        with self.assertRaises(KeyError):
            runner.release(
                next(case for case in acceptance_inventory() if case.row == 7),
                "candidate",
            )

    async def test_cancelled_native_keeps_large_stdout_and_settles(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            case = next(
                case
                for case in acceptance_inventory()
                if case.workload == "join_capture"
            )
            runner = AcceptanceRunner(
                {
                    "cpus": [0],
                    "native_binaries": {
                        "candidate": {
                            "stream_join_perf/enabled": {
                                "path": sys.executable,
                                "sha256": "a" * 64,
                            },
                        }
                    },
                },
                root,
                0,
            )
            script = "import os,time; os.write(1,b'x'*262144); time.sleep(60)"
            task = None
            create_process = asyncio.create_subprocess_exec

            async def create_fixture_native(*arguments, **options):
                self.assertEqual(arguments[1:3], ("--cpu-list", "0"))
                return await create_process(sys.executable, "-c", script, **options)

            try:
                with (
                    patch(
                        "scripts.benchmark_suite.issue406_run._taskset_executable",
                        return_value="taskset",
                    ),
                    patch(
                        "scripts.benchmark_suite.issue406_run.asyncio.create_subprocess_exec",
                        side_effect=create_fixture_native,
                    ),
                ):
                    task = asyncio.create_task(
                        runner._native((case,), "candidate", 0, 0)
                    )
                    destination = (
                        runner.destination((case,), "candidate", 0, 0) / "stdout.log"
                    )
                    async with asyncio.timeout(2):
                        while (
                            not destination.exists()
                            or destination.stat().st_size < 262144
                        ):
                            await asyncio.sleep(0.001)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    async with asyncio.timeout(2):
                        result = await runner.settle()
                self.assertEqual(result["forced_kill_pids"], [])
                self.assertEqual(destination.stat().st_size, 262144)
            finally:
                await settle_fixture_runner(runner, task)

    def test_matched_native_samples_receive_validated_primary_metrics(self):
        from scripts.test_benchmark_native_acceptance import matched_observation

        runner = AcceptanceRunner(
            {
                "cpus": [0],
                "releases": {"candidate": {"product_commit": "c" * 40}},
            },
            Path("unused"),
            0,
        )
        for workload, scenario, metric in (
            ("asof_operator", "asof_join", "processing_seconds"),
            ("join_small_batch", "join_sealed", "handler_seconds"),
        ):
            case = next(
                case
                for case in acceptance_inventory()
                if case.workload == workload and case.mode == "off"
            )
            result = runner._normalize(
                (case,),
                "candidate",
                matched_observation(scenario),
                "observation",
                "a" * 64,
            )[case.id]
            self.assertGreater(result[metric], 0)
            self.assertTrue(result["oracle_passed"])
            if case.row == 7:
                self.assertTrue(result["processing_settled"])
                self.assertTrue(result["processing_checkpoint_free"])
                self.assertEqual(result["nonterminal_epochs"], [])

    async def test_budget_expiry_and_resource_failure_settle_collection(self):
        for fail_resources in (False, True):
            with (
                self.subTest(fail_resources=fail_resources),
                TemporaryDirectory() as raw,
            ):
                entered, settled = asyncio.Event(), asyncio.Event()

                async def pending_collection(
                    _runner, _output, entered=entered, settled=settled
                ):
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        settled.set()

                async def guard(runner, entered=entered, fail_resources=fail_resources):
                    await entered.wait()
                    if fail_resources:
                        raise RuntimeError("controlled resource failure")
                    await runner.stop.wait()

                plan = {
                    "cpus": [0],
                    "budget_seconds": 0.025,
                    "settlement_seconds": 0.005,
                }
                with (
                    patch("scripts.benchmark_suite.issue406_run.validate_plan"),
                    patch(
                        "scripts.benchmark_suite.issue406_run.collect",
                        pending_collection,
                    ),
                    patch.object(AcceptanceRunner, "resources_guard", guard),
                ):
                    result = await run(plan, Path(raw))
                self.assertTrue(settled.is_set())
                self.assertEqual(result["state"], "stopped")
                expected = (
                    "controlled resource failure" if fail_resources else "TimeoutError"
                )
                self.assertIn(expected, result["error"])
                self.assertTrue(result["settlement"]["settled"])
                self.assertTrue((Path(raw) / "evidence-sha256.json").is_file())

    async def test_cancellation_during_cleanup_does_not_start_another_round(self):
        with TemporaryDirectory() as raw:
            entered, release = asyncio.Event(), asyncio.Event()

            class WaitingRunner(FakeRunner):
                async def finish_round(self):
                    self.closed += 1
                    entered.set()
                    await release.wait()

            runner = WaitingRunner()
            task = asyncio.create_task(collect(runner, Path(raw)))
            await entered.wait()
            count = len(runner.calls)
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(len(runner.calls), count)
            self.assertEqual(runner.closed, 1)

    async def test_worker_reads_checkpoint_evidence_larger_than_64k(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            case = acceptance_inventory()[0]
            identity = {
                "native_sha256": "native",
                "cpu_affinity": [0],
                "tokio_worker_threads": "1",
                "large_evidence": "x" * 100_000,
            }
            script = (
                "import json,sys\n"
                "for line in sys.stdin:\n"
                " request=json.loads(line)\n"
                f" value={identity!r} if request['operation']=='hello'"
                " else {'state':'completed'}\n"
                " print(json.dumps(value),flush=True)\n"
            )
            runner = AcceptanceRunner(
                {
                    "cpus": [0],
                    "releases": {
                        "candidate": {
                            "site": str(root),
                            "native_sha256": "native",
                        }
                    },
                },
                root,
                0,
            )
            create_process = asyncio.create_subprocess_exec

            async def create_fixture_worker(*arguments, **options):
                self.assertEqual(arguments[1:3], ("--cpu-list", "0"))
                return await create_process(sys.executable, "-c", script, **options)

            try:
                with (
                    patch(
                        "scripts.benchmark_suite.issue406_run._taskset_executable",
                        return_value="taskset",
                    ),
                    patch(
                        "scripts.benchmark_suite.issue406_run.asyncio.create_subprocess_exec",
                        side_effect=create_fixture_worker,
                    ),
                ):
                    await runner._start_worker(case, "candidate", 0)
                await runner.finish_round()
            finally:
                await runner.settle()
            record = next(root.rglob("environment.json"))
            self.assertEqual(
                len(json.loads(record.read_text())["large_evidence"]), 100_000
            )

    async def test_cancellation_during_creation_registers_child_and_propagates(self):
        with TemporaryDirectory() as raw:
            runner = AcceptanceRunner({}, Path(raw), 0)
            ready = asyncio.Event()
            release = asyncio.Event()
            process = SimpleNamespace(pid=123)

            async def create():
                ready.set()
                await release.wait()
                return process

            task = asyncio.create_task(
                runner.register_process(asyncio.create_task(create()))
            )
            await ready.wait()
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIs(runner.processes[123], process)

    def test_native_shared_metrics_keep_all_semantic_cases(self):
        groups = collection_groups()
        self.assertEqual(len(groups), 36)
        self.assertEqual(sum(len(group[0].sides) for group in groups), 60)
        self.assertCountEqual(
            [case.id for group in groups for case in group],
            [case.id for case in acceptance_inventory()],
        )
        self.assertEqual([len(group) for group in groups].count(2), 2)

    async def test_fixed_pairs_warmups_and_all_arms_are_preserved(self):
        with TemporaryDirectory() as raw:
            runner = FakeRunner()
            report = await collect(runner, Path(raw))
            self._assert_fixed_collection_counts(report, runner.calls)
            first = next(
                group for group in collection_groups() if len(group[0].sides) == 2
            )
            calls = [
                call
                for call in runner.calls
                if call[0] == "sample" and call[1] == first[0].id
            ]
            self.assertEqual(
                [call[2] for call in calls[:4]],
                ["baseline", "candidate", "candidate", "baseline"],
            )
            self.assertEqual(runner.closed, 72)

    def _assert_fixed_collection_counts(self, report, calls):
        self.assertEqual(len(report["measurements"]), 64)
        self.assertEqual(sum(call[0] == "sample" for call in calls), 1200)
        self.assertEqual(sum(call[0] == "warmup" for call in calls), 120)
        for record in report["measurements"]:
            self.assertEqual([len(values) for values in record["rounds"]], [10, 10])

    async def test_cancellation_settles_and_preserves_partial_without_retry(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            runner = FakeRunner(fail_at=3)
            with self.assertRaises(asyncio.CancelledError):
                await collect(runner, root)
            self.assertEqual(len(runner.calls), 3)
            self.assertEqual(runner.closed, 1)
            self.assertTrue((root / "partial.json").is_file())
            attempts = [
                json.loads(line)
                for line in (root / "attempts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(attempts[-1]["state"], "failed")
            self.assertEqual(attempts[-1]["error"], "CancelledError: ")
