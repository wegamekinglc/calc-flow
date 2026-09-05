"""Tests for the SQL/DataFusion performance evidence contract."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.verify_sql_datafusion_performance import verify_p1, verify_report


def _engine(samples: list[float]) -> dict[str, object]:
    engine = {
        "parallelism_mode": "fixed",
        "configured_partitions": 16,
        "requested_partitions": 16,
        "effective_partitions": 16,
        "available_parallelism": 32,
        "max_partitions": 32,
        "min_rows_per_partition": 65_536,
        "small_rows_threshold": 10_001,
        "parallelism_decision_reused": False,
        "decision_input_rows": 1_000_000,
        "decision_active_entities": 64,
        "decision_active_entities_source": "batch_metadata",
        "partition_limit_reason": "configured_target_partitions",
        "batch_size": 8192,
        "input_logical_partitions": 1,
        "input_batch_rows": [8192] * 122 + [576],
        "normalized_plan_hash": "a" * 64,
        "bounded_window_agg_count": 1,
        "samples_ms": samples,
        "median_ms": 70.0 if samples[0] == 70.0 else 84.0,
        "p25_ms": 70.0 if samples[0] == 70.0 else 84.0,
        "p75_ms": 70.0 if samples[0] == 70.0 else 84.0,
        "mad_ms": 0.0,
        "cv": 0.0,
        "cpu_time_ms": 50.0,
        "peak_rss_bytes": 100_000_000,
        "spill_bytes": 0,
        "empty_partitions": 0,
        "partition_rows": [62_500] * 16,
        "partition_skew": 1.0,
        "window_compute_ms": 45.0,
        "repartition_sort_compute_ms": 5.0,
        "window_operator_count": 1,
        "repartition_operator_count": 1,
        "sort_operator_count": 1,
        "coalesce_operator_count": 1,
        "phase_medians_ms": {
            "runtime_acquire": 0.01,
            "session_state_create": 0.1,
            "input_adapter": 0.1,
            "table_register": 0.1,
            "sql_parse": 0.1,
            "logical_optimize": 0.1,
            "physical_plan": 0.1,
            "execution_to_first_batch": 1.0,
            "execution_remaining": 50.0,
            "collect_or_coalesce": 51.0,
            "output_arrow_wrap": 0.1,
            "audit": 0.01,
            "metrics_traversal": 0.01,
            "physical_plan_string": 0.01,
            "batch_envelope": 0.01,
            "run_result": 0.01,
            "run_session_envelope": 0.01,
        },
    }
    engine["phase_samples_ms"] = {
        name: [value] * len(samples)
        for name, value in engine["phase_medians_ms"].items()
    }
    return engine


def _report() -> dict[str, object]:
    raw_samples = [70.0] * 20
    calc_samples = [84.0] * 20
    return {
        "schema_version": 1,
        "git_sha": "1" * 40,
        "profile": "matched-adaptive",
        "environment": {
            "machine_fingerprint": "2" * 64,
            "dependency_fingerprint": "3" * 64,
            "workload_fingerprint": "4" * 64,
            "datafusion_version": "54.0.0",
            "arrow_version": "58.3.0",
            "build_profile": "release",
            "allocator": "system",
            "os": "linux",
            "arch": "x86_64",
            "cpu_model": "test cpu",
            "available_parallelism": 32,
            "rust_version": "rustc 1.88.0",
            "git_dirty": False,
        },
        "cases": [
            {
                "name": "sma_20",
                "rows": 1_000_000,
                "active_entities": 64,
                "window": 20,
                "warmups": 1,
                "rolling_rewrite_enabled": False,
                "sample_order": [
                    "ab" if index % 2 == 0 else "ba" for index in range(20)
                ],
                "calc_flow": _engine(calc_samples),
                "raw_datafusion": _engine(raw_samples),
                "paired_ratios": [
                    calc / raw
                    for calc, raw in zip(calc_samples, raw_samples, strict=True)
                ],
                "paired_ratio_median": 1.2,
                "paired_ratio_ci_low": 1.19,
                "paired_ratio_ci_high": 1.21,
                "correctness": {
                    "schema": True,
                    "rows": True,
                    "keys": True,
                    "order": True,
                    "null_nan_mask": True,
                    "values": True,
                    "rtol": 1e-10,
                    "atol": 1e-10,
                },
                "comparability": {
                    "comparable": True,
                    "mismatches": [],
                },
                "speedup_conclusion": "calc_flow_over_raw=1.200000x",
            }
        ],
    }


class TestSqlDataFusionEvidence(unittest.TestCase):
    def test_accepts_complete_comparable_twenty_pair_report(self) -> None:
        verify_report(_report(), minimum_samples=20, require_stable=True)

    def test_stable_evidence_rejects_a_dirty_worktree(self) -> None:
        report = _report()
        report["environment"]["git_dirty"] = True

        with self.assertRaisesRegex(ValueError, "clean Git worktree"):
            verify_report(report, minimum_samples=20, require_stable=True)

    def test_rejects_partition_or_plan_mismatch_without_speedup_suppression(
        self,
    ) -> None:
        report = _report()
        case = report["cases"][0]
        case["raw_datafusion"]["effective_partitions"] = 8
        case["raw_datafusion"]["partition_rows"] = [125_000] * 8
        case["comparability"] = {
            "comparable": False,
            "mismatches": ["effective_partitions"],
        }

        with self.assertRaisesRegex(ValueError, "speedup_conclusion"):
            verify_report(report, minimum_samples=20, require_stable=True)

    def test_rejects_mismatched_input_or_physical_plan(self) -> None:
        for field, value in (
            ("input_logical_partitions", 2),
            ("normalized_plan_hash", "b" * 64),
        ):
            with self.subTest(field=field):
                report = _report()
                report["cases"][0]["raw_datafusion"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    verify_report(report, minimum_samples=20, require_stable=True)

    def test_rejects_rewrite_noisy_or_incorrect_evidence(self) -> None:
        variants = []
        rewrite = _report()
        rewrite["cases"][0]["rolling_rewrite_enabled"] = True
        variants.append((rewrite, "rolling rewrite"))
        noisy = _report()
        noisy["cases"][0]["calc_flow"]["samples_ms"] = [50.0, 150.0] * 10
        noisy["cases"][0]["calc_flow"].update(
            {
                "median_ms": 100.0,
                "p25_ms": 50.0,
                "p75_ms": 150.0,
                "mad_ms": 50.0,
                "cv": 0.5,
            }
        )
        variants.append((noisy, "CV"))
        incorrect = _report()
        incorrect["cases"][0]["correctness"]["order"] = False
        variants.append((incorrect, "correctness"))

        for report, message in variants:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                verify_report(report, minimum_samples=20, require_stable=True)

    def test_repeat_report_must_share_fingerprints_and_median_within_ten_percent(
        self,
    ) -> None:
        first = _report()
        second = copy.deepcopy(first)
        second["cases"][0]["calc_flow"]["samples_ms"] = [100.0] * 20
        for field in ("median_ms", "p25_ms", "p75_ms"):
            second["cases"][0]["calc_flow"][field] = 100.0
        ratio = 100.0 / 70.0
        second["cases"][0]["paired_ratios"] = [ratio] * 20
        second["cases"][0]["paired_ratio_median"] = ratio
        second["cases"][0]["paired_ratio_ci_low"] = ratio
        second["cases"][0]["paired_ratio_ci_high"] = ratio

        with self.assertRaisesRegex(ValueError, "independent median"):
            verify_report(
                first,
                minimum_samples=20,
                require_stable=True,
                repeat=second,
            )

    def test_repeat_report_must_share_exact_git_sha(self) -> None:
        first = _report()
        second = copy.deepcopy(first)
        second["git_sha"] = "2" * 40

        with self.assertRaisesRegex(ValueError, "git_sha"):
            verify_report(first, minimum_samples=20, repeat=second)

    def test_repeat_report_must_share_plan_and_symmetric_rss(self) -> None:
        first = _report()
        second = copy.deepcopy(first)
        for engine in ("calc_flow", "raw_datafusion"):
            second["cases"][0][engine]["normalized_plan_hash"] = "b" * 64

        with self.assertRaisesRegex(ValueError, "normalized_plan_hash"):
            verify_report(first, minimum_samples=20, repeat=second)

        second = copy.deepcopy(first)
        first["cases"][0]["calc_flow"]["peak_rss_bytes"] = 160_000_000
        with self.assertRaisesRegex(ValueError, "RSS"):
            verify_report(first, minimum_samples=20, repeat=second)

    def test_p1_accepts_both_workloads_and_serial_memory_control(self) -> None:
        matched = _report()
        matched["cases"].append(copy.deepcopy(matched["cases"][0]))
        matched["cases"][1]["name"] = "dual_sma_spread"
        serial = copy.deepcopy(matched)
        serial["profile"] = "serial-control"
        for case in serial["cases"]:
            engine = case["calc_flow"]
            engine["configured_partitions"] = 1
            engine["requested_partitions"] = 1
            engine["effective_partitions"] = 1
            engine["partition_rows"] = [1_000_000]

        verify_p1(matched, serial)

    def test_p1_rejects_latency_ratio_or_memory_failure(self) -> None:
        for field, value, message in (
            ("median_ms", 91.0, "latency"),
            ("ratio", 1.31, "ratio"),
            ("peak_rss_bytes", 160_000_000, "RSS"),
        ):
            with self.subTest(field=field):
                matched = _report()
                matched["cases"].append(copy.deepcopy(matched["cases"][0]))
                matched["cases"][1]["name"] = "dual_sma_spread"
                serial = copy.deepcopy(matched)
                serial["profile"] = "serial-control"
                for case in serial["cases"]:
                    engine = case["calc_flow"]
                    engine["configured_partitions"] = 1
                    engine["requested_partitions"] = 1
                    engine["effective_partitions"] = 1
                    engine["partition_rows"] = [1_000_000]
                if field == "ratio":
                    matched["cases"][0]["paired_ratio_median"] = value
                else:
                    matched["cases"][0]["calc_flow"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    verify_p1(matched, serial)

    def test_schema_artifact_exists_and_parses(self) -> None:
        schema = Path("schemas/sql-datafusion-performance-v1.schema.json")
        self.assertEqual(
            json.loads(schema.read_text(encoding="utf-8"))["$id"], schema.name
        )

    def test_report_round_trips_from_disk(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw, "report.json")
            path.write_text(json.dumps(_report()), encoding="utf-8")
            verify_report(
                json.loads(path.read_text(encoding="utf-8")), minimum_samples=20
            )


if __name__ == "__main__":
    unittest.main()
