from __future__ import annotations

import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def layout_control_observation():
    return {
        "schema": "calc-flow.join-layout-controls.v1",
        "check_only": True,
        "controls": [
            _layout_control_record(operator, layout)
            for operator in ("join", "asof")
            for layout in ("single_record", "multi_record", "unordered", "wide_key")
        ],
    }


def _layout_control_record(operator: str, layout: str) -> dict:
    geometry = [[7, 9, 8], [7, 9, 8]] if layout == "multi_record" else [[24], [24]]
    key_width = 1024 if layout == "wide_key" else 8
    unordered = layout == "unordered"
    return {
        "operator": operator,
        "layout": layout,
        "data_callbacks": 2,
        "input_rows": 48,
        "output_rows": 144 if unordered and operator == "join" else 24,
        "event_record_batch_rows": geometry,
        "event_record_batch_count": len(geometry[0]),
        "timestamp_descents_per_event": 23 if unordered else 0,
        "per_key_timestamp_descents": 20 if unordered else 0,
        "unique_keys": 4 if unordered else 24,
        "key_bytes_per_event": 24 * key_width,
        "max_key_bytes": key_width,
        "validated_all_rows": True,
        "input_unchanged": True,
        "owned_cleanup_complete": True,
    }


class MeasurementEvidenceTests(unittest.TestCase):
    def test_cpu_budget_uses_the_target_wall_instead_of_a_wider_envelope(self):
        from scripts.benchmark_suite.issue406_acceptance import validate_observation
        from scripts.test_issue406_report import record, selected

        case = selected(1)
        sample = record(case, 1.0)["rounds"][0][0]
        sample["cpu_budget"] = 2.0
        sample["cpu_evidence"] = {
            "schema": "calc-flow.phase-resources.v1",
            "status": "measured",
            "scope": "processing",
            "wall_seconds": 10.0,
            "process_cpu_seconds": 3.0,
            "threads": [{"tid": 1, "runtime_seconds": 1.0}],
            "departed_threads": [],
        }
        with self.assertRaisesRegex(ValueError, "CPU budget"):
            validate_observation(case, "candidate", sample)

    def test_layout_evidence_rejects_coalesced_messages_and_unexecuted_paths(self):
        from scripts.benchmark_suite.native_components import (
            validate_join_layout_controls,
        )

        observation = layout_control_observation()
        original = copy.deepcopy(observation)
        validate_join_layout_controls(observation)
        for field, value in (
            ("event_record_batch_rows", [[24], [24]]),
            ("validated_all_rows", False),
        ):
            bad = copy.deepcopy(observation)
            bad["controls"][1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_join_layout_controls(bad)
        self.assertEqual(observation, original)

    def test_thread_exit_does_not_remove_cpu_from_the_process_total(self):
        from scripts.benchmark_suite.issue406_measurement import PhaseClock

        with (
            patch(
                "scripts.benchmark_suite.issue406_measurement._thread_runtimes",
                side_effect=[
                    {"1/3": (10, "owner"), "2/4": (20, "worker")},
                    {"1/3": (30, "owner"), "3/5": (7, "new-worker")},
                ],
            ),
            patch(
                "scripts.benchmark_suite.issue406_measurement.time.process_time_ns",
                side_effect=[100, 200],
            ),
            patch(
                "scripts.benchmark_suite.issue406_measurement.time.perf_counter_ns",
                side_effect=[1_000, 2_000],
            ),
        ):
            evidence = PhaseClock("processing").finish()
        self.assertEqual(evidence["departed_threads"], ["2/4"])
        self.assertEqual(evidence["process_cpu_seconds"], 100 / 1e9)
        self.assertEqual(
            [item["runtime_seconds"] for item in evidence["threads"]],
            [20 / 1e9, 7 / 1e9],
        )

    def test_native_tick_resolution_cannot_certify_a_zero_tick_short_callback(self):
        from scripts.benchmark_suite.issue406_measurement import cpu_evidence_status

        evidence = {
            "schema": "calc-flow.phase-resources.v1",
            "status": "measured",
            "wall_seconds": 0.00005,
            "process_cpu_seconds": 0.0,
            "process_cpu_resolution_seconds": 0.01,
            "threads": [{"tid": 1, "runtime_seconds": 0.00004}],
            "departed_threads": [],
        }
        self.assertEqual(
            cpu_evidence_status(evidence, threads=1, budget=2.0), "inconclusive"
        )

    def test_event_geometry_counts_records_inside_each_data_message(self):
        import pyarrow as pa

        from scripts.benchmark_suite.issue406_measurement import event_layout

        table = pa.Table.from_batches(
            [pa.record_batch({"value": [1, 2]}), pa.record_batch({"value": [3]})]
        )
        batch = SimpleNamespace(num_rows=3, to_pyarrow=lambda: table)
        events = {"left": (SimpleNamespace(batch=batch), None)}
        result = event_layout(events)
        self.assertEqual(result["multi_record_events"], 1)
        self.assertEqual(result["data_events"][0]["record_batch_rows"], [2, 1])

    def test_cpu_budget_rejects_serial_and_requires_declared_parallel_budget(self):
        from scripts.benchmark_suite.issue406_measurement import validate_cpu_evidence

        evidence = {
            "schema": "calc-flow.phase-resources.v1",
            "status": "measured",
            "scope": "processing",
            "wall_seconds": 1.0,
            "process_cpu_seconds": 2.01,
            "threads": [{"tid": 1, "runtime_seconds": 1.0}],
            "departed_threads": [],
        }
        before = copy.deepcopy(evidence)
        with self.assertRaisesRegex(ValueError, "CPU budget"):
            validate_cpu_evidence(evidence, threads=1, budget=2.0)
        validate_cpu_evidence(evidence, threads=4, budget=4.0)
        for budget in (None, 0.0, float("nan"), 100.0):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                validate_cpu_evidence(evidence, threads=4, budget=budget)
        self.assertEqual(evidence, before)

    def test_cpu_clock_keeps_process_total_and_explicit_thread_coverage(self):
        from scripts.benchmark_suite.issue406_measurement import PhaseClock

        clock = PhaseClock("processing")
        sum(index * index for index in range(10_000))
        evidence = clock.finish()
        self.assertEqual(evidence["scope"], "processing")
        self.assertGreater(evidence["wall_seconds"], 0.0)
        self.assertGreater(evidence["process_cpu_seconds"], 0.0)
        if sys.platform == "linux":
            self.assertTrue(evidence["threads"])
        else:
            self.assertEqual(evidence["status"], "unmeasured")
        self.assertIn("departed_threads", evidence)

    def test_reports_retain_p99_and_reject_observed_excess_cpu(self):
        from scripts.benchmark_suite.issue406_report import summarize_acceptance
        from scripts.test_issue406_report import record, report, selected

        case = selected(1)
        observations = record(case, 1.0)
        for group in observations["rounds"]:
            for sample in group:
                sample["cpu_evidence"] = {
                    "schema": "calc-flow.phase-resources.v1",
                    "status": "measured",
                    "scope": "processing",
                    "wall_seconds": 1.0,
                    "process_cpu_seconds": 2.1,
                    "threads": [{"tid": 1, "runtime_seconds": 1.0}],
                    "departed_threads": [],
                }
                sample["cpu_budget"] = 2.0
        raw = report([observations])
        before = copy.deepcopy(raw)
        result = summarize_acceptance(raw)
        arm = next(item for item in result["cases"] if item["case"] == case.id)[
            "candidate"
        ]
        self.assertEqual(arm["status"], "invalid")
        self.assertIn("p99", arm["statistics"])
        self.assertEqual(len(result["targets"]), 8)
        self.assertEqual(raw, before)

    def test_serial_schedule_rejects_claimed_overlap_and_clock_tampering(self):
        from scripts.benchmark_suite.native_components import (
            validate_serial_compaction_schedule,
        )

        sample = {
            "queue_seconds": 8e-9,
            "combined_seconds": 25e-9,
            "schedule": {
                "owner": "single-exclusive-operator",
                "during_compaction_status": "unmeasured",
                "reason": "exclusive operator borrow through preparation",
                "preparation_first_poll_pending": True,
                "preparation_started_ns": 0,
                "handler_ready_ns": 2,
                "preparation_completed_ns": 9,
                "handler_started_ns": 10,
                "handler_completed_ns": 20,
            },
        }
        validate_serial_compaction_schedule(sample)
        original = copy.deepcopy(sample)
        for field, value in (
            ("handler_started_ns", 5),
            ("during_compaction_status", "measured"),
        ):
            invalid = {**sample, "schedule": {**sample["schedule"], field: value}}
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "schedule"),
            ):
                validate_serial_compaction_schedule(invalid)
        with self.assertRaisesRegex(ValueError, "clocks"):
            validate_serial_compaction_schedule({**sample, "queue_seconds": 1.0})
        self.assertEqual(sample, original)
