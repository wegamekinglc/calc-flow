"""Preserve the current baseline, fixed 1T/4T contract and complete evidence."""

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

from scripts.benchmark_suite import sum_columnar_keys as keys
from scripts.benchmark_suite import sum_state


def protocol():
    return {
        "slug": keys.SLUG,
        "baseline_revision": keys.BASELINE,
        "candidate_revision": "a" * 40,
        "budget_seconds": 600,
        "rounds": 2,
        "pairs_per_round": 10,
        "warmups_per_arm_per_round": 1,
        "cases": list(keys.CASES),
        "control_fixtures": keys.CONTROL_FIXTURES,
        "profile_route_anchors": list(keys.PROFILE_ANCHORS),
        "tool_source_sha256": keys.helper_seals(),
    }


def builds():
    return {
        side: {
            "revision": revision,
            "binary_sha256": side,
            "harness_sha256": keys.COMMON_HARNESS,
            "product_recompiled": True,
            "example_recompiled": True,
            "compiled_columnar_anchors": [],
        }
        for side, revision in (("baseline", keys.BASELINE), ("candidate", "a" * 40))
    }


def profiles():
    return [
        {
            "side": side,
            "binary_sha256": side,
            "windows": [{}] * 5,
            "samples_per_window": [2] * 5,
            "clock_alignment": {"matched_sample_count": 10, "max_error_ms": 0},
        }
        for side in ("baseline", "candidate")
    ]


def completed_cases():
    return [
        {
            **case,
            "observations": [{}] * 40,
            "baseline": [[0.1] * 10] * 2,
            "candidate": [[0.08] * 10] * 2,
            "statistics": {"verdict": "improved", "samples": 20},
        }
        for case in keys.CASES
    ]


def publication_input(directory, records):
    root = Path(directory)
    output, destination = root / "output", root / "publish"
    output.mkdir()
    destination.mkdir()
    args = SimpleNamespace(
        output=output, publish_directory=destination, prior_measurement_seconds=0
    )
    context = keys.followup.Followup(
        sum_state.CollectionContext(args, builds(), {}, 0, 600),
        protocol(),
        "protocol",
        "columnar-keys",
        None,
    )
    acquisition = {
        "status": "completed",
        "failure": None,
        "cases": completed_cases(),
        "profiles": [],
        "started_monotonic": 0,
    }
    summary = {
        "status": "completed",
        "failure": None,
        "cases": completed_cases(),
        "profiles": records,
    }
    return (
        context,
        acquisition,
        summary,
        {"acquisition": acquisition, "profiles": records},
    )


class ColumnarContractTests(unittest.TestCase):
    def test_frozen_python_helper_drift_is_rejected_before_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            frozen = protocol()
            frozen["tool_source_sha256"][keys.TOOL_FILES[0]] = "wrong"
            path.write_text(json.dumps(frozen))
            args = SimpleNamespace(protocol=path)
            collection = sum_state.CollectionContext(args, {}, {}, 0, 600)
            with (
                patch.object(keys, "validate_receipt_seals"),
                patch.object(
                    keys.maintained, "collection_context", return_value=collection
                ),
                patch.object(keys, "validate_products"),
                self.assertRaisesRegex(ValueError, "helper"),
            ):
                keys.prepare_context(args, 0)

    def test_current_multithread_control_uses_four_physical_cores(self):
        self.assertEqual(
            [case["id"] for case in keys.CASES],
            [
                "stock-stream-1cpu",
                "stock-stream-4cpu",
                "sparse-stream-1cpu",
                "distinct-stream-1cpu",
                "edge-stream-1cpu",
            ],
        )
        self.assertEqual(keys.CASES[1]["id"], "stock-stream-4cpu")
        self.assertEqual(keys.CASES[1]["cpus"], [0, 2, 4, 6])
        self.assertTrue(all(case["partitions"] == 1 for case in keys.CASES))

    def test_one_profile_cannot_satisfy_the_required_two_product_inventory(self):
        with self.assertRaisesRegex(ValueError, "profiles"):
            keys.validate_profiles([{"side": "baseline"}], {})

    def test_protocol_cannot_expand_inventory_budget_or_samples(self):
        keys.validate_protocol(protocol())
        for changes in (
            {"budget_seconds": 601},
            {"pairs_per_round": 11},
            {"cases": list(keys.CASES) + [keys.CASES[0]]},
            {"baseline_revision": "74a1eaf2183adfd04fcb1aeecd655c6df73ed16f"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                keys.validate_protocol({**protocol(), **changes})

    def test_old_P74_and_stale_native_receipts_are_rejected(self):
        keys.validate_products(builds(), protocol())
        for side, changes in (
            ("baseline", {"revision": "74a1eaf2183adfd04fcb1aeecd655c6df73ed16f"}),
            ("candidate", {"product_recompiled": False}),
            ("baseline", {"harness_sha256": "other"}),
        ):
            altered = {**builds(), side: {**builds()[side], **changes}}
            with (
                self.subTest(side=side, changes=changes),
                self.assertRaises(ValueError),
            ):
                keys.validate_products(altered, protocol())

    def test_validated_profiles_bind_both_binaries_and_all_five_clock_windows(self):
        keys.validate_profiles(profiles(), builds())
        for change in (
            {"binary_sha256": "wrong"},
            {"samples_per_window": [2, 2, 0, 2, 2]},
            {"clock_alignment": {"matched_sample_count": 10, "max_error_ms": 1}},
        ):
            records = [profiles()[0], {**profiles()[1], **change}]
            with self.subTest(change=change), self.assertRaises(ValueError):
                keys.validate_profiles(records, builds())

    def test_dense_improvement_with_no_regression_is_required(self):
        cases = [
            {**case, "statistics": {"verdict": "inconclusive"}} for case in keys.CASES
        ]
        cases[2]["statistics"]["verdict"] = "improved"
        self.assertFalse(keys.performance_acceptance("completed", cases))
        cases[1]["statistics"]["verdict"] = "improved"
        self.assertTrue(keys.performance_acceptance("completed", cases))
        self.assertFalse(keys.performance_acceptance("stopped", cases))
        cases[0]["statistics"]["verdict"] = "regression"
        self.assertFalse(keys.performance_acceptance("completed", cases))

    def test_build_receipt_drift_fails_before_a_native_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.json"
            path.write_text("{}")
            args = SimpleNamespace(baseline_build=path, candidate_build=path)
            with patch.object(keys.maintained, "invoke") as invoke:
                with self.assertRaisesRegex(ValueError, "receipt"):
                    keys.validate_receipt_seals(
                        args,
                        {
                            "build_receipt_sha256": {
                                "baseline": "wrong",
                                "candidate": "wrong",
                            }
                        },
                    )
                invoke.assert_not_called()


class ColumnarLifecycleTests(unittest.TestCase):
    def test_two_invalid_profile_records_are_retained_but_never_accepted(self):
        invalid = [profiles()[0], {**profiles()[1], "binary_sha256": "wrong"}]
        with tempfile.TemporaryDirectory() as directory:
            context, acquisition, summary, raw = publication_input(directory, invalid)
            with (
                patch.object(
                    keys.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    keys.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(keys.time, "monotonic", return_value=1),
                patch("builtins.print"),
            ):
                keys.publish(context, acquisition, {}, {})
            destination = context.collection.args.publish_directory
            published = json.loads((destination / "summary.json").read_text())
            self.assertEqual(published["status"], "stopped")
            self.assertFalse(published["profile_analysis_complete"])
            self.assertNotEqual(published["profile_disposition"], "accepted")
            self.assertEqual(published["accepted_profile_count"], 0)
            manifest = json.loads((destination / "evidence-manifest.json").read_text())
            self.assertEqual(manifest["current_profiles"], 0)
            self.assertEqual(manifest["profile_records"], 2)
            self.assertIn(
                "Accepted profiles: 0", (destination / "README.md").read_text()
            )
            complete = json.loads(
                gzip.decompress(
                    (destination / "complete-evidence.json.gz").read_bytes()
                )
            )
            self.assertEqual(complete["profiles"], invalid)

    def test_postpublication_helper_drift_revokes_every_completed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            context, acquisition, summary, raw = publication_input(
                directory, profiles()
            )
            with (
                patch.object(
                    keys.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    keys.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(keys.time, "monotonic", return_value=1),
                patch.object(
                    keys,
                    "validate_tool_seals",
                    side_effect=[None, ValueError("measurement helper drift")],
                ),
                patch("builtins.print"),
            ):
                keys.publish(context, acquisition, {}, {})
            args = context.collection.args
            for path in (
                args.output / "acquisition.json",
                args.output / "summary.json",
                args.output / "phase-stopped.json",
                args.publish_directory / "summary.json",
                args.publish_directory / "evidence-manifest.json",
            ):
                record = json.loads(path.read_text())
                self.assertEqual(record["status"], "stopped")
                self.assertIn("helper drift", record["failure"])
            manifest = json.loads(
                (args.publish_directory / "evidence-manifest.json").read_text()
            )
            self.assertFalse(manifest["performance_acceptance"])
            for name, seal in manifest["files"].items():
                self.assertEqual(
                    keys.maintained.sha(args.publish_directory / name), seal["sha256"]
                )

    def test_invalid_profiles_stop_before_any_formal_case(self):
        context = SimpleNamespace(
            collection=SimpleNamespace(
                builds=builds(), scope={"cases": keys.CASES}, deadline=600
            )
        )
        with (
            patch.object(keys, "prepare_fixtures"),
            patch.object(keys.maintained, "collect_profiles"),
            patch.object(keys.followup, "accepted_profiles", return_value=[]),
            patch.object(keys.maintained, "collect_case") as case,
        ):
            with self.assertRaisesRegex(ValueError, "profiles"):
                keys.collect(context, {"profiles": []})
            case.assert_not_called()

    def test_profiles_precede_the_five_fixed_cases(self):
        context = SimpleNamespace(
            collection=SimpleNamespace(
                builds=builds(), scope={"cases": keys.CASES}, deadline=600
            )
        )
        order = []
        with (
            patch.object(keys, "prepare_fixtures"),
            patch.object(
                keys.maintained,
                "collect_profiles",
                side_effect=lambda *_: order.append("profiles+decode"),
            ),
            patch.object(keys.followup, "accepted_profiles", return_value=profiles()),
            patch.object(
                keys.maintained,
                "collect_case",
                side_effect=lambda _, case, __: order.append(case["id"]),
            ),
            patch.object(keys.time, "monotonic", return_value=1),
        ):
            acquisition = {"status": "running"}
            keys.collect(context, acquisition)
            self.assertEqual(
                order, ["profiles+decode", *[case["id"] for case in keys.CASES]]
            )
            self.assertEqual(acquisition["status"], "completed")

    def test_cancellation_still_finalizes_and_publishes_partial_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            collection = sum_state.CollectionContext(args, builds(), {}, 0, 600)
            context = SimpleNamespace(collection=collection, protocol=protocol())
            with (
                patch.object(keys, "prepare_context", return_value=context),
                patch.object(keys, "plan_and_machine", return_value=({}, {})),
                patch.object(keys, "collect", side_effect=asyncio.CancelledError()),
                patch.object(keys.maintained, "finalize_acquisition") as finalize,
                patch.object(keys, "publish") as publish,
                self.assertRaises(asyncio.CancelledError),
            ):
                keys.run(args)
            finalize.assert_called_once()
            self.assertEqual(publish.call_args.args[1]["status"], "stopped")
            self.assertIn("CancelledError", publish.call_args.args[1]["failure"])

    def test_late_publication_rewrites_all_statuses_and_keeps_raw_clock_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, destination = root / "output", root / "publish"
            output.mkdir()
            destination.mkdir()
            stamp = output / "profiles/baseline/sample-timestamps.txt"
            stamp.parent.mkdir(parents=True)
            stamp.write_text("11/12 1.000000000:\n")
            args = SimpleNamespace(
                output=output,
                publish_directory=destination,
                prior_measurement_seconds=0,
            )
            context = keys.followup.Followup(
                sum_state.CollectionContext(args, builds(), {}, 0, 10),
                protocol(),
                "protocol",
                "columnar-keys",
                None,
            )
            acquisition = {
                "status": "completed",
                "failure": None,
                "cases": completed_cases(),
                "profiles": [{"path": "profiles/baseline"}],
                "started_monotonic": 0,
            }
            summary = {
                "status": "completed",
                "failure": None,
                "cases": completed_cases(),
                "profiles": profiles(),
            }
            raw = {"acquisition": acquisition}
            ticks = iter((1, 1, 11))
            with (
                patch.object(
                    keys.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    keys.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(
                    keys.time, "monotonic", side_effect=lambda: next(ticks, 12)
                ),
                self.assertRaisesRegex(TimeoutError, "publication deadline"),
            ):
                keys.publish(context, acquisition, {}, {})
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
                stamp.read_text(),
            )
            self.assertIn(
                "Performance acceptance: **False**",
                (destination / "README.md").read_text(),
            )


KEYS = (("", "a|b"), ("a|", "b"), ("雪", ""), ("x", "y"))


def trade_side(index, cycle):
    return "buy" if cycle % 2 == 0 or index % 4 == 3 else "sell"


def trade_columns(rows, cycles, seed):
    return [
        pa.array([key[0] for key in rows]),
        pa.array([key[1] for key in rows]),
        pa.array([0 if seed else (1 << 63) - 1] * len(rows), pa.int64()),
        pa.array([trade_side(index, cycle) for index, cycle in enumerate(cycles)]),
        pa.array([1.0] * len(rows)),
        pa.array(["trade" if seed else "order"] * len(rows)),
    ]


def trade_record(seed=False):
    rows = list(KEYS) if seed else list(KEYS) * 10
    cycles = [0] * 4 if seed else [i // 4 for i in range(40)]
    return pa.record_batch(
        trade_columns(rows, cycles, seed),
        names=["account", "stock", "quantity", "side", "price", "flow_type"],
    )


def write_source(path):
    seed, business = trade_record(True), trade_record()
    with pa.ipc.new_stream(path, seed.schema) as writer:
        writer.write_batch(seed)
        for _ in range(20):
            writer.write_batch(business)


class ColumnarFixtureTests(unittest.TestCase):
    def test_distinct_control_and_first_edge_keep_full_key_sets(self):
        record = trade_record()
        for copies in (1, 2):
            selected = keys.control_record(record, 9, 4, copies)
            pairs = list(
                zip(
                    selected.column(0).to_pylist(),
                    selected.column(1).to_pylist(),
                    strict=True,
                )
            )
            self.assertEqual(selected.num_rows, 4 * copies)
            self.assertEqual(set(pairs), set(KEYS))
            self.assertTrue(all(pairs.count(key) == copies for key in KEYS))
            self.assertEqual(
                selected.column("flow_type").to_pylist(), ["trade"] * (4 * copies)
            )
            self.assertEqual(selected.column("side")[0].as_py(), "sell")
            self.assertEqual(record.column("flow_type")[0].as_py(), "order")

    def test_all_twenty_one_oracles_are_independent_and_wrap_integers(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stock.arrow"
            write_source(source)
            original = source.read_bytes()
            for copies in (1, 2):
                target = Path(directory) / f"control-{copies}.arrow"
                manifest = keys.control_fixture(source, target, copies)
                self.assertEqual(manifest["batch_rows"], [4] + [4 * copies] * 20)
                self.assertEqual(manifest["snapshot_rows"], [4] * 21)
                self.assertEqual(manifest["input_rows"], 4 + 80 * copies)
                for index, digest in enumerate(manifest["snapshot_sha256"]):
                    value = ((1 << 63) - 1) if index % 2 and copies == 1 else 0
                    wrapped = (index * copies * ((1 << 63) - 1) + (1 << 63)) % (
                        1 << 64
                    ) - (1 << 63)
                    expected = [
                        [
                            account,
                            stock,
                            wrapped if (account, stock) == KEYS[3] else value,
                        ]
                        for account, stock in sorted(KEYS)
                    ]
                    self.assertEqual(digest, keys.evidence.json_hash(expected))
                self.assertEqual(source.read_bytes(), original)

    def test_fixture_failure_unlinks_owned_partial_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (
                Path(directory) / "source.arrow",
                Path(directory) / "out.arrow",
            )
            write_source(source)
            original = source.read_bytes()
            with (
                patch.object(
                    keys.evidence,
                    "ensure_remaining",
                    side_effect=TimeoutError("fixture"),
                ),
                self.assertRaises(TimeoutError),
            ):
                keys.control_fixture(source, target, 1)
            self.assertFalse(target.exists())
            self.assertEqual(source.read_bytes(), original)

    def test_existing_output_and_same_input_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.arrow"
            write_source(source)
            original = source.read_bytes()
            with self.assertRaises(FileExistsError):
                keys.control_fixture(source, source, 1)
            self.assertEqual(source.read_bytes(), original)

    def test_malformed_cycles_copies_and_nulls_are_rejected(self):
        record = trade_record()
        invalid = record.set_column(
            2, record.schema.field(2), pa.array([None] * 40, pa.int64())
        )
        for source, copies in ((record.slice(0, 39), 1), (record, 3), (invalid, 1)):
            with self.subTest(copies=copies), self.assertRaises(ValueError):
                keys.control_record(source, 0, 4, copies)

    def test_missing_snapshot_batch_rejects_and_deletes_incomplete_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (
                Path(directory) / "source.arrow",
                Path(directory) / "out.arrow",
            )
            with pa.ipc.new_stream(source, trade_record(True).schema) as writer:
                writer.write_batch(trade_record(True))
                writer.write_batch(trade_record())
            with self.assertRaisesRegex(ValueError, "twenty"):
                keys.control_fixture(source, target, 1)
            self.assertFalse(target.exists())

    def test_control_selector_dimensions_are_frozen(self):
        self.assertEqual(keys.CONTROL_FIXTURES["distinct"]["business_sql_rows"], 10_000)
        self.assertEqual(keys.CONTROL_FIXTURES["edge"]["business_sql_rows"], 20_000)
        changed = {**protocol(), "control_fixtures": {}}
        with self.assertRaisesRegex(ValueError, "protocol"):
            keys.validate_protocol(changed)


class ColumnarRouteTests(unittest.TestCase):
    def test_prefix_collision_never_receives_exact_anchor_credit(self):
        anchor = keys.PROFILE_ANCHORS[0]
        data = {
            "frames": [anchor + "_other [binary]", anchor + " [binary]"],
            "stacks": [{"frames": [0], "samples": 7}, {"frames": [1], "samples": 3}],
        }
        self.assertEqual(keys.anchor_samples(data, [anchor]), 3)

    def test_hash_and_binary_annotations_canonicalize_to_one_exact_symbol(self):
        anchor = keys.PROFILE_ANCHORS[1]
        self.assertEqual(
            keys.canonical_symbol(anchor + "::h0123456789abcdef [binary]"), anchor
        )
        self.assertNotEqual(keys.canonical_symbol(anchor + "_other [binary]"), anchor)

    def test_baseline_presence_and_noncanonical_receipt_anchors_are_rejected(self):
        keys.validate_symbol_anchors(builds())
        for side, anchors in (
            ("baseline", [keys.PROFILE_ANCHORS[0]]),
            ("candidate", [keys.PROFILE_ANCHORS[0] + "_other"]),
        ):
            invalid = {
                **builds(),
                side: {**builds()[side], "compiled_columnar_anchors": anchors},
            }
            with self.subTest(side=side), self.assertRaises(ValueError):
                keys.validate_symbol_anchors(invalid)

    def test_inlined_or_unsampled_anchors_remain_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stacks.gz"
            self.assertIsNone(keys.profile_route(path, [])["inclusive_samples"])
            path.write_bytes(
                gzip.compress(
                    json.dumps(
                        {"frames": ["other"], "stacks": [{"frames": [0], "samples": 2}]}
                    ).encode()
                )
            )
            record = keys.profile_route(path, [keys.PROFILE_ANCHORS[0]])
            self.assertEqual(record["status"], "unresolved_by_sampling")
            self.assertEqual(record["inclusive_samples"], 0)

    def test_dispatcher_only_samples_never_prove_active_interner_work(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stacks.gz"
            anchor = keys.PROFILE_ANCHORS[1]
            path.write_bytes(
                gzip.compress(
                    json.dumps(
                        {"frames": [anchor], "stacks": [{"frames": [0], "samples": 7}]}
                    ).encode()
                )
            )
            self.assertNotEqual(
                keys.profile_route(path, [anchor])["status"], "observed"
            )
            path.write_bytes(
                gzip.compress(
                    json.dumps(
                        {
                            "frames": [anchor, keys.PROFILE_ANCHORS[0]],
                            "stacks": [
                                {"frames": [0], "samples": 7},
                                {"frames": [1], "samples": 3},
                            ],
                        }
                    ).encode()
                )
            )
            record = keys.profile_route(path, [anchor, keys.PROFILE_ANCHORS[0]])
            self.assertEqual(record["status"], "observed")
            self.assertEqual(record["active_work_samples"], 3)
            self.assertEqual(record["inclusive_samples"], 10)


if __name__ == "__main__":
    unittest.main()
