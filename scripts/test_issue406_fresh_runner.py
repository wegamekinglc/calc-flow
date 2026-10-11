from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import sys
import textwrap
import time
import unittest
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite.issue406_fresh import (
    CANDIDATE,
    PREVIOUS,
    retained_cases,
)
from scripts.benchmark_suite.issue406_run import AcceptanceRunner, run
from scripts.test_issue406_failures import retained_failure
from scripts.test_issue406_fresh import fresh_plan
from scripts.test_issue406_report import selected


class FreshRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_exits_with_owned_failure_when_creation_resists_cancellation(
        self,
    ):
        program = textwrap.dedent("""
            import asyncio
            import sys
            import time
            from pathlib import Path
            from unittest.mock import patch
            from scripts.benchmark_suite.issue406_run import (
                AcceptanceRunner, _run_cli, _settle_acceptance_collection,
            )
            from scripts.test_issue406_fresh import fresh_plan
            from scripts.toolkit import write_json

            async def stalled_run(plan, output):
                runner = AcceptanceRunner(plan, output, time.monotonic())
                ready = asyncio.Event()
                async def create():
                    process = await asyncio.create_subprocess_exec(
                        sys.executable, "-c", "import time; time.sleep(60)",
                        start_new_session=True,
                    )
                    write_json(output / "child.json", {"pid": process.pid})
                    ready.set()
                    while True:
                        try:
                            await asyncio.Event().wait()
                        except asyncio.CancelledError:
                            pass
                collection = asyncio.create_task(
                    runner.register_process(asyncio.create_task(create()))
                )
                guard = asyncio.create_task(runner.stop.wait())
                await ready.wait()
                report = {"assessment": {"environment_suitable": True}}
                await _settle_acceptance_collection(
                    runner, collection, guard, report, output
                )
                write_json(output / "result.json", report)
                return report

            plan = {**fresh_plan(), "settlement_seconds": 0.08}
            with patch("scripts.benchmark_suite.issue406_run.run", stalled_run):
                asyncio.run(_run_cli(plan, Path(sys.argv[1])))
            raise SystemExit("unexpected normal return")
        """)
        with TemporaryDirectory() as raw:
            root = Path(raw)
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    program,
                    raw,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _, stderr = await asyncio.wait_for(process.communicate(), 4)
                finally:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
                self.assertEqual(process.returncode, 1)
                self.assertTrue(
                    (root / "ownership-failure.json").exists(), stderr.decode()
                )
                failure = json.loads((root / "ownership-failure.json").read_text())
                self.assertFalse(failure["settled"])
                self.assertFalse(failure["surviving_pids"])
                self.assertGreater(failure["pending_tasks"], 0)
                pid = json.loads((root / "child.json").read_text())["pid"]
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            finally:
                if (root / "child.json").exists():
                    pid = json.loads((root / "child.json").read_text())["pid"]
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)

    async def test_never_completed_creation_is_cancelled_before_settlement_deadline(
        self,
    ):
        from scripts.benchmark_suite.issue406_run import _settle_acceptance_collection

        with TemporaryDirectory() as raw:
            plan = {**fresh_plan(), "settlement_seconds": 0.08}
            runner = AcceptanceRunner(plan, Path(raw), time.monotonic())
            ready, release = asyncio.Event(), asyncio.Event()

            async def create():
                ready.set()
                await release.wait()
                return SimpleNamespace(pid=123, returncode=0)

            creation = asyncio.create_task(create())
            collection = asyncio.create_task(runner.register_process(creation))
            guard = asyncio.create_task(runner.stop.wait())
            await ready.wait()
            report = {"assessment": {"environment_suitable": True}}
            settlement = asyncio.create_task(
                _settle_acceptance_collection(
                    runner, collection, guard, report, Path(raw)
                )
            )
            try:
                await asyncio.wait_for(asyncio.shield(settlement), 0.3)
                self.assertTrue(creation.cancelled())
                self.assertTrue(collection.cancelled())
                self.assertTrue(report["settlement"]["settled"])
                self.assertEqual(report["settlement"]["pending_creations"], 0)
            finally:
                creation.cancel()
                release.set()
                await asyncio.gather(
                    settlement, collection, guard, return_exceptions=True
                )

    def test_exhausted_final_seal_reserve_revokes_all_verdicts(self):
        import json

        from scripts.benchmark_suite.issue406_run import _seal_acceptance_report

        with TemporaryDirectory() as raw:
            output = Path(raw)
            report = {
                "state": "complete",
                "assessment": {
                    "compatibility_verified": True,
                    "environment_suitable": True,
                },
            }
            with patch(
                "scripts.benchmark_suite.issue406_run.time.monotonic", return_value=5397
            ):
                _seal_acceptance_report(fresh_plan(), output, 0, report, [])
            self.assertFalse(report["budget_satisfied"])
            self.assertFalse(report["assessment"]["environment_suitable"])
            self.assertFalse(
                json.loads((output / "budget.json").read_text())["evidence_complete"]
            )
            self.assertIn("budget", (output / "performance.md").read_text())

    def test_evidence_write_deadline_revokes_and_reseals_every_verdict(self):
        from scripts.benchmark_suite.issue406_run import _seal_acceptance_report
        from scripts.toolkit import sha256_file, write_json

        with TemporaryDirectory() as raw:
            output = Path(raw)
            clock = [5390]
            report = {
                "state": "complete",
                "assessment": {
                    "compatibility_verified": True,
                    "environment_suitable": True,
                },
            }

            def write(path, value):
                write_json(path, value)
                if path.name == "evidence-sha256.json":
                    clock[0] = 5397

            with (
                patch(
                    "scripts.benchmark_suite.issue406_run.time.monotonic",
                    side_effect=lambda: clock[0],
                ),
                patch(
                    "scripts.benchmark_suite.issue406_run.write_json", side_effect=write
                ),
                patch(
                    "scripts.benchmark_suite.issue406_report.summarize_acceptance",
                    return_value={"verdict": "pass"},
                ),
                patch(
                    "scripts.benchmark_suite.issue406_report.render_acceptance_markdown",
                    return_value="ACCEPTANCE PASS\n",
                ),
            ):
                _seal_acceptance_report(fresh_plan(), output, 0, report, [])
            result = json.loads((output / "result.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            budget = json.loads((output / "budget.json").read_text())
            hashes = json.loads((output / "evidence-sha256.json").read_text())
            self.assertEqual(result["state"], "stopped")
            self.assertFalse(result["budget_satisfied"])
            self.assertFalse(result["assessment"]["environment_suitable"])
            self.assertEqual(summary["verdict"], "inconclusive")
            self.assertNotIn("ACCEPTANCE PASS", (output / "performance.md").read_text())
            self.assertFalse(budget["evidence_complete"])
            self.assertFalse(budget["budget_satisfied"])
            for name in ("result.json", "summary.json", "performance.md"):
                self.assertEqual(hashes[name], sha256_file(output / name))

    def test_primary_summary_renders_supplementary_control(self):
        from scripts.benchmark_suite.issue406_report import (
            render_acceptance_markdown,
            summarize_acceptance,
        )

        with patch(
            "scripts.benchmark_suite.issue406_fresh_report.summarize_retained_control",
            return_value={"kept": True},
        ):
            summary = summarize_acceptance({"retained_control": {}})
        self.assertEqual(summary["retained_control"], {"kept": True})
        with patch(
            "scripts.benchmark_suite.issue406_fresh_report.render_retained_control",
            return_value=["supplementary marker"],
        ):
            self.assertIn("supplementary marker", render_acceptance_markdown(summary))

    async def test_settlement_waits_for_cancelled_creation_registration(self):
        with TemporaryDirectory() as raw:
            runner = AcceptanceRunner(fresh_plan(), Path(raw), time.monotonic())
            ready, release = asyncio.Event(), asyncio.Event()
            process = SimpleNamespace(
                pid=123, returncode=0, communicate=AsyncMock(return_value=(None, None))
            )

            async def create():
                ready.set()
                await release.wait()
                return process

            task = asyncio.create_task(
                runner.register_process(asyncio.create_task(create()))
            )
            await ready.wait()
            task.cancel()
            with patch.object(
                runner,
                "group_members",
                side_effect=lambda: {pid: [] for pid in runner.groups},
            ):
                settlement = asyncio.create_task(runner.settle())
                await asyncio.sleep(0)
                self.assertFalse(settlement.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                result = await settlement
            self.assertTrue(result["settled"])
            self.assertFalse(runner.groups)
            self.assertFalse(runner.processes)

    def test_fresh_validation_keeps_native_and_traced_build_provenance_separate(self):
        from scripts.benchmark_suite.issue406_reuse import (
            FROZEN,
            HISTORICAL,
            WITNESS_SHA256,
        )
        from scripts.benchmark_suite.issue406_run import validate_plan

        plan = fresh_plan()
        controller = "d" * 40
        plan["harness_git_sha"] = controller
        old = {"original_build_harness_git_sha": HISTORICAL}
        frozen = {"original_build_harness_git_sha": FROZEN}
        new = {"original_build_harness_git_sha": controller}
        plan["native_harness_reuse"] = {"baseline": old, "candidate": new}
        plan["row7_harness_reuse"] = {"baseline": frozen, "candidate": new}
        plan["historical_harness_witness"] = {
            "path": "/reviewed/witness.json",
            "sha256": WITNESS_SHA256,
        }
        plan["expected_native_rustc"] = "rustc"
        files = {"/reviewed/witness.json": WITNESS_SHA256}

        with (
            patch(
                "scripts.benchmark_suite.issue406_run._validate_plan_inventory",
                return_value=files,
            ),
            patch(
                "scripts.benchmark_suite.issue406_run.competing_work", return_value=[]
            ),
            patch(
                "scripts.benchmark_suite.issue406_run.native_harness_reuse",
                side_effect=lambda revision, **kwargs: {
                    "original_build_harness_git_sha": revision
                },
            ) as reuse,
            patch(
                "scripts.benchmark_suite.issue406_reuse.validate_ordinary_release",
                return_value={"rustc": "rustc"},
            ) as ordinary,
            patch(
                "scripts.benchmark_suite.issue406_failures.validate_main_failure_identity"
            ),
            patch(
                "scripts.benchmark_suite.issue406_run._validate_plan_wheel",
                return_value={
                    "environment": {"CARGO_TARGET_DIR": "/cache"},
                    "validated_artifacts": [],
                },
            ) as wheel,
            patch(
                "scripts.benchmark_suite.issue406_run._validate_plan_binaries"
            ) as native,
        ):
            validate_plan(plan)
            for key, side, wrong in (
                ("native_harness_reuse", "baseline", FROZEN),
                ("native_harness_reuse", "baseline", controller),
                ("row7_harness_reuse", "baseline", HISTORICAL),
                ("row7_harness_reuse", "baseline", controller),
                ("native_harness_reuse", "candidate", FROZEN),
                ("row7_harness_reuse", "candidate", FROZEN),
                ("native_harness_reuse", "candidate", "e" * 40),
                ("row7_harness_reuse", "candidate", "e" * 40),
            ):
                invalid = {
                    **plan,
                    key: {
                        **plan[key],
                        side: {"original_build_harness_git_sha": wrong},
                    },
                }
                with (
                    self.subTest(key=key, side=side, wrong=wrong),
                    self.assertRaisesRegex(ValueError, "reuse witness"),
                ):
                    validate_plan(invalid)
            with self.assertRaisesRegex(ValueError, "reuse witness"):
                validate_plan({**plan, "harness_git_sha": "e" * 40})
        self.assertEqual(
            [
                (call.args[0], call.kwargs["compatibility"])
                for call in reuse.call_args_list[:4]
            ],
            [
                (HISTORICAL, Path("/reviewed/witness.json")),
                (controller, None),
                (FROZEN, None),
                (controller, None),
            ],
        )
        self.assertEqual(ordinary.call_count, 3)
        self.assertEqual(
            [call.args[0]["product_commit"] for call in ordinary.call_args_list],
            [
                plan["releases"]["baseline"]["product_commit"],
                CANDIDATE,
                PREVIOUS,
            ],
        )
        self.assertEqual(
            [call.args[3] for call in wheel.call_args_list], [FROZEN, controller]
        )
        self.assertEqual(
            [call.args[3] for call in native.call_args_list], [HISTORICAL, controller]
        )

    async def test_known_failure_requires_prepared_identity_and_natural_empty_group(
        self,
    ):
        from dataclasses import asdict

        from scripts.benchmark_suite.issue406_failures import MAIN_WHEEL_NATIVE
        from scripts.benchmark_suite.issue406_fresh import MAIN
        from scripts.toolkit import write_json

        with TemporaryDirectory() as raw:
            plan = fresh_plan()
            plan["cpus"] = [0]
            plan["releases"]["baseline"]["native_sha256"] = MAIN_WHEEL_NATIVE
            runner = AcceptanceRunner(plan, Path(raw), time.monotonic())
            case = selected(3, mode="on")
            root = Path(raw) / "workers" / case.id / "round-0" / "baseline"
            root.mkdir(parents=True)
            failure = root / "failed-sample-1.json"
            write_json(failure, retained_failure())
            write_json(
                root / "environment.json",
                {
                    "native_sha256": MAIN_WHEEL_NATIVE,
                    "cpu_affinity": [0],
                    "tokio_worker_threads": "1",
                },
            )
            (root / "stderr.log").write_text(
                "benchmarks.checkpoint_cycles.CycleValidationError: "
                "invalid acceptance sample\n"
            )
            process = SimpleNamespace(
                pid=123,
                returncode=1,
                stdin=SimpleNamespace(close=lambda: None),
                wait=AsyncMock(return_value=1),
            )
            worker = SimpleNamespace(process=process, log=io.BytesIO())
            runner.workers["baseline"] = worker
            runner.processes[123] = runner.groups[123] = process
            error = RuntimeError(
                "invalid worker result: {'error': 'CycleValidationError: "
                "invalid acceptance sample'}"
            )
            self.assertIsNone(
                await runner.known_main_failure((case,), "baseline", 0, None, error)
            )
            runner.prepared_cases[(case.id, "baseline", 0)] = {
                **asdict(case),
                "family": "join-asof-acceptance",
                "side": "baseline",
                "product_commit": MAIN,
                "fixture_root": str(runner.fixture_root("baseline", "on", 0)),
            }
            with (
                patch.object(runner, "group_members", return_value={123: [456]}),
                self.assertRaisesRegex(RuntimeError, "descendant"),
            ):
                await runner.known_main_failure((case,), "baseline", 0, None, error)
            self.assertIn("baseline", runner.workers)
            with patch.object(runner, "group_members", return_value={123: []}):
                record = await runner.known_main_failure(
                    (case,), "baseline", 0, None, error
                )
            self.assertIn("cause_unresolved", record["signature"])
            self.assertFalse(runner.groups)
            self.assertFalse(runner.workers)
            self.assertEqual(record["natural_exit"]["returncode"], 1)

    def test_previous_release_is_only_available_to_retained_off_controls(self):
        plan = fresh_plan()
        runner = AcceptanceRunner(plan, Path("unused"), 0)
        case = retained_cases()[0]
        self.assertEqual(runner.release(case, "previous")["product_commit"], PREVIOUS)
        from dataclasses import replace

        with self.assertRaisesRegex(ValueError, "retained"):
            runner.release(replace(case, mode="on"), "previous")

    def test_previous_normalization_preserves_actual_revision(self):
        plan = fresh_plan()
        plan["cpus"] = [0]
        runner = AcceptanceRunner(plan, Path("unused"), 0)
        case = retained_cases()[0]
        raw = {"product_commit": PREVIOUS, "cpu_affinity": [0]}
        with patch(
            "scripts.benchmark_suite.issue406_run.validate_observation"
        ) as validate:
            result = runner._normalize((case,), "previous", raw, "id", "a" * 64)
        self.assertEqual(result[case.id]["product_commit"], PREVIOUS)
        self.assertEqual(validate.call_args.args[1], "candidate")
        with self.assertRaisesRegex(ValueError, "product|affinity"):
            runner._normalize(
                (case,),
                "previous",
                {**raw, "product_commit": CANDIDATE},
                "id",
                "a" * 64,
            )

    async def test_finish_requires_natural_exit_and_exact_invocation_count(self):
        with TemporaryDirectory() as raw:
            runner = AcceptanceRunner(fresh_plan(), Path(raw), time.monotonic())
            process = SimpleNamespace(
                pid=123,
                returncode=0,
                stdin=SimpleNamespace(close=lambda: None),
                wait=AsyncMock(return_value=0),
            )
            worker = SimpleNamespace(
                process=process,
                request=AsyncMock(return_value={"state": "completed", "samples": 11}),
                log=io.BytesIO(),
                close=AsyncMock(),
            )
            runner.workers["candidate"] = worker
            runner.processes[123] = process
            runner.groups[123] = process
            with patch.object(runner, "group_members", return_value={123: []}):
                await runner.finish_round("candidate")
            worker.close.assert_not_called()
            self.assertFalse(runner.groups)
            self.assertFalse(runner.workers)
            self.assertTrue(worker.log.closed)

    async def test_live_descendant_prevents_retirement(self):
        with TemporaryDirectory() as raw:
            runner = AcceptanceRunner(fresh_plan(), Path(raw), time.monotonic())
            process = SimpleNamespace(pid=123, returncode=0)
            runner.processes[123] = runner.groups[123] = process
            with (
                patch.object(runner, "group_members", return_value={123: [456]}),
                self.assertRaisesRegex(RuntimeError, "descendant"),
            ):
                await runner.retire_process(123)
            self.assertIn(123, runner.groups)
            self.assertIn(123, runner.processes)

    async def test_v2_run_selects_fresh_collector_and_preserves_partial_control(self):
        with TemporaryDirectory() as raw:
            from scripts.toolkit import write_json

            async def collect(_runner, output):
                write_json(
                    output / "partial.json",
                    {
                        "measurements": [],
                        "retained_control": {"marker": "kept"},
                        "known_failures": ["kept"],
                    },
                )
                raise RuntimeError("synthetic unknown failure")

            plan = fresh_plan()
            plan["cpus"] = [0]
            with (
                patch("scripts.benchmark_suite.issue406_run.validate_plan"),
                patch("scripts.benchmark_suite.issue406_fresh.collect_fresh", collect),
                patch.object(
                    AcceptanceRunner,
                    "resources_guard",
                    lambda runner: runner.stop.wait(),
                ),
                patch("scripts.benchmark_suite.issue406_run._seal_acceptance_report"),
            ):
                result = await run(plan, Path(raw))
            self.assertIn("synthetic unknown failure", result["error"])
            self.assertEqual(result["retained_control"], {"marker": "kept"})
            self.assertEqual(result["known_failures"], ["kept"])
            self.assertTrue(result["settlement"]["settled"])
