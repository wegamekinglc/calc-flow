from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from scripts.benchmark_suite.three_arm_qualification import (
    NATIVE_DELAY_NS,
    PYTHON_DELAY_NS,
    DiskMonitor,
    collect_qualification,
    effect_diagnostics,
    native_injection_source,
    patch_python_wheel,
    python_injection_source,
    qualification_plan,
    qualification_verdict,
)


class ThreeArmQualificationTests(unittest.TestCase):
    def test_workflow_runs_one_bounded_three_arm_job(self):
        workflow = Path(".github/workflows/benchmarks.yml").read_text()
        job = workflow.split("  dal316-side-bias:\n", 1)[1].split(
            "  dal301-profile-build:\n", 1
        )[0]
        self.assertIn("dal316-three-arm", workflow)
        self.assertIn("timeout-minutes: ${{", job)
        for name in (
            "Pre-register three-arm schedule",
            "Build Python hot-path wheel",
            "Build native hot-path wheel",
            "Collect three-arm installed pairs",
            "Validate three-arm evidence",
        ):
            self.assertIn(name, job)

    def test_disk_monitor_records_runner_filesystem_peak_during_install(self):
        with TemporaryDirectory() as directory:
            usage = iter(
                (
                    SimpleNamespace(total=100, used=20, free=80),
                    SimpleNamespace(total=100, used=35, free=65),
                    SimpleNamespace(total=100, used=25, free=75),
                )
            )

            async def observe():
                with patch(
                    "scripts.benchmark_suite.three_arm_qualification.shutil.disk_usage",
                    side_effect=lambda _: next(
                        usage, SimpleNamespace(total=100, used=25, free=75)
                    ),
                ):
                    async with DiskMonitor(Path(directory), interval=0.001) as monitor:
                        await asyncio.sleep(0.005)
                return monitor.summary()

            result = asyncio.run(observe())
            self.assertEqual(result["peak_used_bytes"], 35)
            self.assertEqual(result["min_free_bytes"], 65)
            self.assertGreaterEqual(result["sample_count"], 3)

    def test_plan_preregisters_counterbalanced_arms_and_fixed_blocks(self):
        plan = qualification_plan("a" * 64)
        self.assertEqual(
            plan["round_arm_order"],
            [["aa", "python", "native"], ["native", "aa", "python"]],
        )
        self.assertEqual([len(round_) for round_ in plan["schedule"]], [10, 10])
        self.assertEqual(plan["python_delay_ns"], PYTHON_DELAY_NS)
        self.assertEqual(plan["native_delay_ns"], NATIVE_DELAY_NS)

    def test_python_injection_changes_source_binding_timed_data_path(self):
        original = (Path("python/calc_flow/runtime.py")).read_text()
        injected = python_injection_source(original)
        self.assertIn("_BENCH316_PYTHON_DATA_DELAY_NS", injected)
        self.assertIn("if isinstance(value, Data):", injected)
        self.assertIn("time.perf_counter_ns()", injected)
        self.assertEqual(injected.count("_BENCH316_PYTHON_DATA_DELAY_NS"), 2)
        self.assertEqual(original.count("value = await self.source.next()"), 1)

    def test_native_injection_changes_source_event_data_path(self):
        original = (Path("crates/calc-flow-python/src/continuous.rs")).read_text()
        injected = native_injection_source(original)
        self.assertIn("BENCH316_NATIVE_DATA_DELAY_NS", injected)
        self.assertIn("Some(calc_flow::SourceEvent::Data { .. })", injected)
        hot_path = injected.split("const BENCH316_NATIVE_DATA_DELAY_NS", 1)[1].split(
            "async fn close", 1
        )[0]
        self.assertIn("Duration::from_nanos(BENCH316_NATIVE_DATA_DELAY_NS)", hot_path)
        self.assertNotIn("std::time::Duration::from_nanos", hot_path)
        self.assertEqual(injected.count("BENCH316_NATIVE_DATA_DELAY_NS"), 2)

    def test_python_wheel_repack_seals_changed_file_and_keeps_native(self):
        with TemporaryDirectory() as directory:
            base = Path(directory) / "base.whl"
            variant = Path(directory) / "python.whl"
            with ZipFile(base, "w") as wheel:
                wheel.writestr(
                    "calc_flow/runtime.py",
                    Path("python/calc_flow/runtime.py").read_bytes(),
                )
                wheel.writestr("calc_flow/_native.abi3.so", b"native")
                wheel.writestr("calc_flow_python-1.dist-info/RECORD", b"")
            patch_python_wheel(base, variant)
            with ZipFile(variant) as wheel:
                self.assertIn(
                    "_BENCH316_PYTHON_DATA_DELAY_NS",
                    wheel.read("calc_flow/runtime.py").decode(),
                )
                self.assertEqual(wheel.read("calc_flow/_native.abi3.so"), b"native")
                record = wheel.read("calc_flow_python-1.dist-info/RECORD").decode()
                self.assertIn("calc_flow/runtime.py,sha256=", record)

    def test_qualification_requires_aa_clear_and_two_confirmed_regressions(self):
        accepted = {
            "aa": "no-confirmed-regression",
            "python": "regression",
            "native": "regression",
        }
        self.assertEqual(qualification_verdict(accepted), "qualified")
        for arm in accepted:
            changed = {**accepted, arm: "unresolved"}
            with self.subTest(arm=arm):
                self.assertEqual(qualification_verdict(changed), "blocked")

    def test_serial_or_slot_effect_blocks_exact_interval_claim(self):
        plan = qualification_plan("a" * 64)
        summary = {
            "block_changes": [[float(i) for i in range(10)], [0.0] * 10],
            "strata_medians": {
                "slot:A0": [0.0, 0.0],
                "slot:B1": [10.0, 0.0],
                "order:AB": [0.0, 0.0],
                "order:BA": [0.0, 0.0],
            },
        }
        observed = effect_diagnostics(summary, plan)
        self.assertTrue(observed["material"])
        self.assertGreater(observed["rounds"][0]["sequence_correlation"], 0.9)
        self.assertEqual(observed["rounds"][0]["slot_median_gap"], 10.0)


class ThreeArmCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_collect_uses_one_preregistered_schedule_for_all_three_arms(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = qualification_plan("a" * 64)
            (root / "plan.json").write_text(json.dumps(plan))
            observations = []
            base = {
                "wheel_sha256": "a" * 64,
                "native_sha256": "c" * 64,
                "wheel_path": "/sealed/base.whl",
                "git_sha": plan["base_git_sha"],
            }

            async def fake_side(_case, _release, role, site, index, output):
                self.assertFalse(site.exists())
                site.mkdir(parents=True)
                observations.append((output.parts[-3], role, str(site), index))
                return {"sample": {"seconds": 1.0}}

            with (
                patch(
                    "scripts.benchmark_suite.three_arm_qualification.load_release",
                    return_value=base,
                ),
                patch(
                    "scripts.benchmark_suite.three_arm_qualification.load_variant",
                    return_value={
                        **base,
                        "wheel_sha256": "b" * 64,
                        "package_files": {},
                    },
                ),
                patch(
                    "scripts.benchmark_suite.three_arm_qualification.wheel_package_files",
                    return_value={},
                ),
                patch(
                    "scripts.benchmark_suite.three_arm_qualification.get_shard",
                    return_value={},
                ),
                patch(
                    "scripts.benchmark_suite.three_arm_qualification.shard_cases",
                    return_value=[{"id": plan["case_id"]}],
                ),
                patch(
                    "scripts.benchmark_suite.three_arm_qualification.collect_side",
                    side_effect=fake_side,
                ),
                patch(
                    "scripts.benchmark_suite.three_arm_qualification.summarize_report",
                    return_value={"verdict": "no-confirmed-regression"},
                ),
                patch(
                    "scripts.benchmark_suite.three_arm_qualification.validate_report",
                    return_value={"verdict": "no-confirmed-regression"},
                ),
            ):
                await collect_qualification(
                    root / "plan.json", root / "base.json", root
                )
            self.assertEqual(len(observations), 120)
            self.assertEqual(observations[0][0], "aa")
            self.assertEqual(observations[20][0], "python")
            self.assertEqual(observations[40][0], "native")
            self.assertEqual(observations[60][0], "native")
            self.assertEqual(observations[80][0], "aa")
            self.assertEqual(observations[100][0], "python")
            self.assertEqual(len({item[2] for item in observations}), 2)


if __name__ == "__main__":
    unittest.main()
