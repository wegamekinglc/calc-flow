"""Keep the bounded parallel SUM comparison and its independent fixture honest."""

from __future__ import annotations

import asyncio
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa

from scripts.benchmark_suite import sum_batch_parallel as parallel
from scripts.benchmark_suite import sum_state
from scripts.benchmark_suite import sum_state_profile as profile


def trade_batch(rows):
    schema = pa.schema(
        [
            pa.field("account", pa.string()),
            pa.field("stock", pa.string()),
            pa.field("quantity", pa.int64()),
            pa.field("side", pa.string()),
            pa.field("price", pa.float64()),
            pa.field("flow_type", pa.string()),
        ]
    )
    return pa.RecordBatch.from_pylist(rows, schema=schema)


def trade(account="account-000", quantity=0, flow="trade"):
    return {
        "account": account,
        "stock": "stock-000",
        "quantity": quantity,
        "side": "buy",
        "price": 10.0,
        "flow_type": flow,
    }


class FixtureTests(unittest.TestCase):
    def test_fixture_deadline_stops_and_removes_only_its_owned_partial_file(self):
        seed = trade_batch([trade()])
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("stock", "hotkey"))
            with (
                source.open("wb") as output,
                pa.ipc.new_stream(output, seed.schema) as writer,
            ):
                writer.write_batch(seed)
            with (
                patch.object(parallel.time, "monotonic", return_value=11),
                self.assertRaisesRegex(TimeoutError, "fixture"),
            ):
                parallel.hotkey_fixture(source, target, 10)
            self.assertFalse(target.exists())
            self.assertTrue(source.is_file())

    def test_reference_wraps_and_filters_without_mutating_prior_state(self):
        previous = {("account-000", "stock-000"): (1 << 63) - 1}
        batch = trade_batch([trade(quantity=1), trade(quantity=100, flow="order")])
        updated = parallel.reference_batch(previous, batch)
        self.assertEqual(updated[("account-000", "stock-000")], -(1 << 63))
        self.assertEqual(previous[("account-000", "stock-000")], (1 << 63) - 1)

    def test_common_harness_null_boundary_is_rejected_explicitly(self):
        batch = trade_batch([trade(quantity=None)])
        with self.assertRaisesRegex(ValueError, "non-null"):
            parallel.reference_batch({}, batch)

    def test_hotkey_preserves_seed_tail_and_every_snapshot_independently(self):
        seed = trade_batch([trade(), trade("tail"), trade("nul\0雪")])
        business = trade_batch([trade("tail", 1) for _ in range(100)])
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("stock", "hotkey"))
            with (
                source.open("wb") as output,
                pa.ipc.new_stream(output, seed.schema) as writer,
            ):
                writer.write_batch(seed)
                for _ in range(20):
                    writer.write_batch(business)
            before = source.read_bytes()
            manifest = parallel.hotkey_fixture(source, target)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(manifest["batch_rows"], [3] + [100] * 20)
            self.assertEqual(manifest["snapshot_rows"], [3] * 21)
            expected = []
            for ordinal in range(21):
                rows = [
                    ["account-000", "stock-000", ordinal * 90],
                    ["nul\0雪", "stock-000", 0],
                    ["tail", "stock-000", ordinal * 10],
                ]
                expected.append(parallel.json_hash(rows))
            self.assertEqual(manifest["snapshot_sha256"], expected)
            with pa.ipc.open_stream(target) as reader:
                self.assertTrue(next(reader).equals(seed))
            with self.assertRaises(FileExistsError):
                parallel.hotkey_fixture(source, source)


def protocol():
    return {
        "slug": parallel.SLUG,
        "budget_seconds": 600,
        "rounds": 2,
        "pairs_per_round": 10,
        "warmups_per_arm_per_round": 1,
        "cases": list(parallel.CASES),
        "candidate_revision": "a" * 40,
        "sql_phase_frames": [list(rule) for rule in parallel.SQL_PHASE_FRAMES],
    }


class ProtocolTests(unittest.TestCase):
    def test_fixed_five_cases_have_one_partition_and_physical_cpu_policy(self):
        parallel.validate_protocol(protocol())
        self.assertEqual(
            [case["cpus"] for case in parallel.CASES],
            [[0], [0, 2, 4, 6], list(range(8)), [0], list(range(8))],
        )
        self.assertTrue(all(case["partitions"] == 1 for case in parallel.CASES))

    def test_rejects_case_sample_budget_or_unrelated_worker_frame_drift(self):
        for changed in (
            {"budget_seconds": 601},
            {"pairs_per_round": 11},
            {"cases": list(parallel.CASES[:-1])},
            {"sql_phase_frames": [["sql_local_update", "tokio::worker"]]},
            {"sql_phase_frames": [["sql_local_update", "calc_flow::operator::sql::"]]},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                parallel.validate_protocol({**protocol(), **changed})

    def test_actual_af400_baseline_and_new_common_harness_product_are_required(self):
        builds = {
            "baseline": {
                "revision": parallel.BASELINE,
                "binary_sha256": parallel.BASELINE_BINARY,
                "harness_sha256": parallel.COMMON_HARNESS,
            },
            "candidate": {
                "revision": "a" * 40,
                "binary_sha256": "new",
                "harness_sha256": parallel.COMMON_HARNESS,
                "product_recompiled": True,
                "example_recompiled": True,
            },
        }
        parallel.validate_products(builds, protocol())
        with self.assertRaisesRegex(ValueError, "baseline"):
            parallel.validate_products(
                {
                    **builds,
                    "baseline": {
                        **builds["baseline"],
                        "revision": parallel.MERGED_BASE,
                    },
                },
                protocol(),
            )
        with self.assertRaisesRegex(ValueError, "fresh"):
            parallel.validate_products(
                {
                    **builds,
                    "candidate": {**builds["candidate"], "product_recompiled": False},
                },
                protocol(),
            )

    def test_corrupt_merged_base_equivalence_is_rejected_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.rs"
            path.write_text("hello\n")
            build = {"source_root": directory, "source_sha256": {"source.rs": "sealed"}}
            with patch.object(
                parallel,
                "command_output",
                return_value="ce013625030ba8dba906f756967f9e9ca394464a source.rs",
            ):
                parallel.baseline_equivalence(build, 10**10)
                path.write_text("different\n")
                with self.assertRaisesRegex(ValueError, "differs from public"):
                    parallel.baseline_equivalence(build, 10**10)

    def test_wrong_candidate_source_is_rejected_even_with_matching_revision_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol()))
            args = SimpleNamespace(protocol=path)
            collection = sum_state.CollectionContext(
                args, {"baseline": {}, "candidate": {}}, {}, 0, 600
            )
            with (
                patch.object(
                    parallel.maintained, "collection_context", return_value=collection
                ),
                patch.object(parallel, "validate_products"),
                patch.object(parallel, "baseline_equivalence"),
                patch.object(
                    parallel,
                    "source_equivalence",
                    side_effect=ValueError("wrong candidate source"),
                ),
                self.assertRaisesRegex(ValueError, "wrong candidate source"),
            ):
                parallel.prepare_context(args, 0)

    def test_only_dense_target_improvement_can_accept_the_complete_inventory(self):
        cases = [
            {"id": case["id"], "statistics": {"verdict": "no-confirmed-regression"}}
            for case in parallel.CASES
        ]
        cases[3]["statistics"]["verdict"] = "improved"
        self.assertFalse(parallel.performance_acceptance(cases))
        cases[1]["statistics"]["verdict"] = "improved"
        self.assertTrue(parallel.performance_acceptance(cases))
        self.assertFalse(parallel.performance_acceptance(cases[:-1]))
        cases[4]["statistics"]["verdict"] = "regression"
        self.assertFalse(parallel.performance_acceptance(cases))


class WorkerEvidenceTests(unittest.TestCase):
    def test_async_merge_bodies_are_exact_coordinator_frames_without_worker_credit(
        self,
    ):
        prefix = "calc_flow::operator::sql::incremental::grouped_parallel::"
        for name in ("merge_fragment", "resolve_ranks", "merge_states"):
            symbol = prefix + name + "::{{closure}}"
            with self.subTest(name=name):
                self.assertEqual(
                    profile._phase((symbol + " [binary]",), parallel.SQL_PHASE_FRAMES),
                    "sql_coordinator_merge",
                )
                self.assertEqual(
                    profile._phase(
                        (symbol + "_other [binary]",), parallel.SQL_PHASE_FRAMES
                    ),
                    "other",
                )
                counts = profile.SampleCounts(
                    [0], phase_frames=parallel.SQL_PHASE_FRAMES
                )
                counts.observe(
                    (symbol + " [binary]",), {"tid": 11, "name": "coordinator"}, 1, 0
                )
                self.assertEqual(counts.worker_distribution(), [])

    def test_worker_prefix_collision_and_unrelated_sql_are_not_worker_samples(self):
        needle = "calc_flow::operator::sql::Work::run"
        rules = (("sql_local_update", needle),)
        self.assertEqual(profile._phase((needle + "away [binary]",), rules), "other")
        self.assertEqual(
            profile._phase(("calc_flow::operator::sql::Work::new [binary]",), rules),
            "other",
        )
        merge = parallel.SQL_PHASE_FRAMES[-1]
        self.assertEqual(
            profile._phase((merge[1] + " [binary]",), (merge,)), "sql_coordinator_merge"
        )

    def test_compiled_symbol_admission_rejects_gather_only_or_prefix_collisions(self):
        worker = parallel.SQL_PHASE_FRAMES[0][1]
        for names in ({"calc_flow::runtime::gather::run"}, {worker + "_other"}):
            with (
                self.subTest(names=names),
                self.assertRaisesRegex(ValueError, "exact SQL worker"),
            ):
                parallel.admitted_symbols(names)
        self.assertEqual(
            parallel.admitted_symbols({worker})["sql_local_update"], [worker]
        )

    def test_segments_of_one_os_tid_do_not_prove_multiple_workers(self):
        records = [
            {"side": "baseline"},
            {
                "side": "candidate",
                "windows": [{}] * 5,
                "sql_worker_samples_by_window": [
                    {"window": i, "tid": tid, "samples": 1}
                    for i in range(1, 6)
                    for tid in ("11", "11.1")
                ],
            },
        ]
        with self.assertRaisesRegex(ValueError, "multiple threads"):
            parallel.validate_worker_profiles(records)

    def test_worker_attribution_is_sql_specific_per_window_and_thread(self):
        counts = profile.SampleCounts(
            [0, 0], phase_frames=protocol()["sql_phase_frames"]
        )
        worker = protocol()["sql_phase_frames"][0][1]
        counts.observe(
            (worker + " [binary]",),
            {"tid": 11, "name": "worker"},
            2,
            0,
        )
        counts.observe(
            (worker + " [binary]",),
            {"tid": 12, "name": "worker"},
            3,
            1,
        )
        counts.observe(
            ("ExpressionOperator::process_data",), {"tid": 13, "name": "worker"}, 5, 0
        )
        self.assertEqual(counts.phases["sql_local_update"], 5)
        self.assertEqual(
            counts.worker_distribution(),
            [
                {"window": 1, "tid": "11", "name": "worker", "samples": 2},
                {"window": 2, "tid": "12", "name": "worker", "samples": 3},
            ],
        )

    def test_missing_or_single_thread_worker_evidence_cannot_pass_profile_gate(self):
        base = {"side": "baseline"}
        candidate = {
            "side": "candidate",
            "windows": [{}] * 5,
            "sql_worker_samples_by_window": [
                {"window": i, "tid": "1", "samples": 2} for i in range(1, 6)
            ],
        }
        for records in ([], [base, candidate]):
            with (
                self.subTest(records=records),
                self.assertRaisesRegex(ValueError, "worker"),
            ):
                parallel.validate_worker_profiles(records)
        candidate["sql_worker_samples_by_window"] += [
            {"window": i, "tid": "2", "samples": 2} for i in range(1, 6)
        ]
        parallel.validate_worker_profiles([base, candidate])


class LifecycleTests(unittest.TestCase):
    def test_partial_report_does_not_claim_accepted_complete_profiles(self):
        summary = {
            "status": "stopped",
            "failure": "invalid profile",
            "cases": [],
            "profiles": [],
        }
        text = parallel.report(summary)
        self.assertNotIn("Both 8CPU profiles contain", text)
        self.assertIn("Accepted profiles: 0", text)

    def test_stopped_after_good_timings_cannot_claim_performance_acceptance(self):
        cases = [
            {"id": case["id"], "statistics": {"verdict": "improved"}}
            for case in parallel.CASES
        ]
        summary = {"status": "stopped", "failure": "source drift", "cases": cases}
        with patch.object(
            parallel.publication, "rendered_rows", return_value=([], [], [])
        ):
            self.assertIn("Performance acceptance: **False**", parallel.report(summary))

    def test_report_keeps_artifact_link_destinations_intact(self):
        summary = {"status": "stopped", "failure": "partial", "cases": []}
        text = parallel.report(summary)
        for destination in ("complete-evidence.json.gz", "profile-targets.json"):
            self.assertIn(f"]({destination})", text)

    def test_malformed_build_receipt_never_starts_a_native_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen, build = root / "protocol.json", root / "build.json"
            frozen.write_text(json.dumps(protocol()))
            build.write_text("{}")
            args = SimpleNamespace(
                protocol=frozen,
                baseline_build=build,
                candidate_build=build,
                output=root / "output",
                publish_directory=root / "publish",
            )
            with patch.object(parallel.maintained, "invoke") as invoke:
                with self.assertRaisesRegex(ValueError, "incomplete build receipt"):
                    parallel.prepare_context(args, 0)
                invoke.assert_not_called()

    def test_invalid_profile_stops_before_any_formal_case(self):
        context = SimpleNamespace(
            collection=SimpleNamespace(scope={"cases": parallel.CASES}, deadline=600)
        )
        with (
            patch.object(parallel, "prepare_fixture"),
            patch.object(parallel, "collect_profiles"),
            patch.object(parallel.followup, "accepted_profiles", return_value=[]),
            patch.object(parallel.maintained, "collect_case") as case,
        ):
            with self.assertRaisesRegex(ValueError, "worker"):
                parallel.collect(context, {"profiles": []})
            case.assert_not_called()

    def test_profiles_and_worker_gate_precede_all_fixed_cases(self):
        context = SimpleNamespace(
            collection=SimpleNamespace(scope={"cases": parallel.CASES}, deadline=600)
        )
        order = []
        with (
            patch.object(
                parallel,
                "prepare_fixture",
                side_effect=lambda *_: order.append("fixture"),
            ),
            patch.object(
                parallel,
                "collect_profiles",
                side_effect=lambda *_: order.append("profiles+decode"),
            ),
            patch.object(parallel.followup, "accepted_profiles", return_value=[]),
            patch.object(
                parallel,
                "validate_worker_profiles",
                side_effect=lambda *_: order.append("worker gate"),
            ),
            patch.object(
                parallel.maintained,
                "collect_case",
                side_effect=lambda _, case, __: order.append(case["id"]),
            ),
            patch.object(parallel.time, "monotonic", return_value=1),
        ):
            acquisition = {"status": "running"}
            parallel.collect(context, acquisition)
            self.assertEqual(
                order,
                [
                    "fixture",
                    "profiles+decode",
                    "worker gate",
                    *[case["id"] for case in parallel.CASES],
                ],
            )
            self.assertEqual(acquisition["status"], "completed")

    def test_cancellation_keeps_partial_failure_and_runs_owned_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            collection = sum_state.CollectionContext(args, {}, {}, 0, 600)
            context = SimpleNamespace(collection=collection)
            with (
                patch.object(parallel, "prepare_context", return_value=context),
                patch.object(
                    parallel.followup,
                    "plan_and_machine",
                    return_value=({"followup": {}}, {"helper_sha256": {}}),
                ),
                patch.object(parallel, "collect", side_effect=asyncio.CancelledError()),
                patch.object(parallel.maintained, "finalize_acquisition") as finalize,
                patch.object(parallel, "publish") as publish,
                self.assertRaises(asyncio.CancelledError),
            ):
                parallel.run(args)
            finalize.assert_called_once()
            acquisition = publish.call_args.args[1]
            self.assertEqual(acquisition["status"], "stopped")
            self.assertIn("CancelledError", acquisition["failure"])

    def test_late_publication_durably_demotes_every_authoritative_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, destination = root / "output", root / "publication"
            output.mkdir()
            destination.mkdir()
            timestamps = output / "profiles/baseline/sample-timestamps.txt"
            timestamps.parent.mkdir(parents=True)
            timestamps.write_text("11/12 1.000000000:\n")
            args = SimpleNamespace(
                output=output,
                publish_directory=destination,
                prior_measurement_seconds=0,
            )
            collection = sum_state.CollectionContext(
                args, {"candidate": {"binary_sha256": "candidate"}}, {}, 0, 10
            )
            context = parallel.followup.Followup(
                collection, protocol(), "protocol", "batch-parallel", None
            )
            acquisition = {
                "status": "completed",
                "failure": None,
                "cases": [],
                "profiles": [{"path": "profiles/baseline"}],
                "started_monotonic": 0,
            }
            summary = {
                "status": "completed",
                "failure": None,
                "cases": [],
                "profiles": [],
            }
            raw = {"acquisition": acquisition}
            ticks = iter((1, 1, 11))
            with (
                patch.object(parallel.publication, "validate_inventory"),
                patch.object(
                    parallel.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    parallel.time, "monotonic", side_effect=lambda: next(ticks, 12)
                ),
                self.assertRaisesRegex(TimeoutError, "publication deadline"),
            ):
                parallel.publish(context, acquisition, {}, {})
            for path in (
                output / "acquisition.json",
                output / "summary.json",
                output / "phase-stopped.json",
                destination / "summary.json",
                destination / "evidence-manifest.json",
            ):
                record = json.loads(path.read_text())
                self.assertEqual(record["status"], "stopped")
                self.assertIn("deadline", record["failure"])
            complete = json.loads(
                gzip.decompress(
                    (destination / "complete-evidence.json.gz").read_bytes()
                )
            )
            self.assertEqual(complete["acquisition"]["status"], "stopped")
            self.assertEqual(
                complete["profile_clock_evidence"][
                    "profiles/baseline/sample-timestamps.txt"
                ]["text"],
                timestamps.read_text(),
            )


if __name__ == "__main__":
    unittest.main()
