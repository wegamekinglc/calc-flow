from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite import measure
from scripts.benchmark_suite.legacy import observe_workers


class BenchmarkMeasureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_round_keeps_partial_samples_and_primary_error(self):
        case = {"id": "recovery", "family": "warm", "backend": "calc-flow"}
        release = {"native_sha256": "a" * 64}
        sample = {"seconds": 0.01, "correctness": {"passed": True}}
        environment = {**release, "polars_threads": 32, "tokio_worker_threads": "32"}
        workers = {}
        for side in ("baseline", "candidate"):
            requests = [environment, {"case": case, "warmup": sample}, sample]
            requests += (
                [TimeoutError("original terminal wait")]
                if side == "baseline"
                else [sample]
            )
            workers[side] = SimpleNamespace(
                request=AsyncMock(side_effect=requests),
                close=AsyncMock(side_effect=OSError("cleanup failure")),
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                measure.Worker, "start", AsyncMock(side_effect=workers.values())
            ):
                result = await measure.measure_case(
                    case,
                    "interleaved",
                    {side: (root, root) for side in workers},
                    dict.fromkeys(workers, release),
                    root,
                )
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error"], "TimeoutError: original terminal wait")
            self.assertEqual(result["evidence"], [])
            partial = result["partial_evidence"][0]
            self.assertFalse(partial["complete"])
            self.assertEqual(
                {side: len(values) for side, values in partial["samples"].items()},
                {"baseline": 1, "candidate": 2},
            )
            self.assertEqual(partial["contexts"]["baseline"]["release"], release)
            self.assertIn("cleanup failure", result["error_notes"][0])
        for worker in workers.values():
            worker.close.assert_awaited_once()

    def test_polars_collection_engine_is_required_for_polars_references(self):
        release = {"native_sha256": "a" * 64}
        for backend, threads in (("polars", 32), ("polars-1t", 1)):
            environment = {
                **release,
                "polars_threads": threads,
                "tokio_worker_threads": "32",
            }
            for engine in (None, "auto", "in-memory"):
                with (
                    self.subTest(backend=backend, engine=engine),
                    self.assertRaisesRegex(ValueError, "Polars collection engine"),
                ):
                    measure.validate_environment(
                        {**environment, "polars_engine": engine},
                        release,
                        polars_threads=threads,
                        backend=backend,
                    )
            valid = {**environment, "polars_engine": "streaming"}
            self.assertEqual(
                measure.validate_environment(
                    valid, release, polars_threads=threads, backend=backend
                ),
                {
                    "polars_threads": threads,
                    "tokio_worker_threads": "32",
                    "polars_engine": "streaming",
                },
            )
            self.assertEqual(valid["native_sha256"], release["native_sha256"])

    async def test_calc_flow_pair_still_rejects_changed_execution_environment(self):
        release = {"native_sha256": "a" * 64}
        environment = {
            **release,
            "polars_threads": 32,
            "tokio_worker_threads": "32",
            "packages": {"polars": "2.0.0", "numpy": "2.4.4"},
            "platform": "same-host",
        }
        for changed in (
            {"native_sha256": "b" * 64},
            {"polars_threads": 1},
            {"tokio_worker_threads": "1"},
            {"packages": {"polars": "2.0.0", "numpy": "other"}},
            {"platform": "other-host"},
        ):
            with self.subTest(changed=changed):
                workers = {
                    side: SimpleNamespace(
                        request=AsyncMock(return_value=identity), close=AsyncMock()
                    )
                    for side, identity in (
                        ("baseline", environment),
                        (
                            "candidate",
                            {**environment, "polars_engine": "streaming", **changed},
                        ),
                    )
                }
                roots = {side: Path(side) for side in workers}
                with (
                    patch.object(
                        measure.Worker, "start", AsyncMock(side_effect=workers.values())
                    ),
                    self.assertRaises(ValueError),
                ):
                    await observe_workers(
                        roots, dict.fromkeys(workers, release), roots, Path("runs")
                    )
                for worker in workers.values():
                    worker.close.assert_awaited_once()

    async def test_warm_pair_accepts_release_before_polars_engine_metadata(self):
        case = {"family": "warm", "backend": "calc-flow", "scenario": "sma20"}
        releases = {
            "baseline": {"native_sha256": "a" * 64},
            "candidate": {"native_sha256": "b" * 64},
        }
        environment = {
            "polars_threads": 32,
            "tokio_worker_threads": "32",
            "packages": {"polars": "2.0.0", "numpy": "2.4.4"},
        }
        workers = {}
        for side, release in releases.items():
            identity = {**environment, **release}
            if side == "candidate":
                identity = {**identity, "polars_engine": "streaming"}
            sample = {
                "seconds": 0.01,
                "correctness": {"passed": True},
                "start_row": 64,
            }
            workers[side] = SimpleNamespace(
                request=AsyncMock(
                    side_effect=[
                        identity,
                        {"case": case, "warmup": sample},
                        *[sample] * 10,
                        {"state": "completed"},
                    ]
                ),
                close=AsyncMock(),
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                measure.Worker, "start", AsyncMock(side_effect=workers.values())
            ):
                result = await measure._round(
                    case,
                    {side: (root / side, root / side) for side in releases},
                    releases,
                    root,
                )
        self.assertEqual(result["environment"], environment)
        for side, worker in workers.items():
            self.assertEqual(len(result["samples"][side]), 10)
            worker.close.assert_awaited_once()

    async def test_legacy_pair_accepts_release_before_polars_engine_metadata(self):
        environment = {"polars_threads": 32, "tokio_worker_threads": "32"}
        releases = {
            "baseline": {"native_sha256": "a" * 64},
            "candidate": {"native_sha256": "b" * 64},
        }
        workers = [
            SimpleNamespace(
                request=AsyncMock(return_value={**environment, **releases["baseline"]}),
                close=AsyncMock(),
            ),
            SimpleNamespace(
                request=AsyncMock(
                    return_value={
                        **environment,
                        **releases["candidate"],
                        "polars_engine": "streaming",
                    }
                ),
                close=AsyncMock(),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            roots = {side: Path(directory) / side for side in releases}
            with patch.object(measure.Worker, "start", AsyncMock(side_effect=workers)):
                observed = await observe_workers(
                    roots, releases, roots, Path(directory)
                )
        self.assertEqual(observed, environment)
        for worker in workers:
            worker.close.assert_awaited_once()

    async def test_single_thread_reference_launches_and_validates_its_own_pool(self):
        case = {"family": "engines", "backend": "polars-1t"}
        release = {"native_sha256": "a" * 64}
        environment = {
            **release,
            "polars_threads": 1,
            "polars_engine": "streaming",
            "tokio_worker_threads": "32",
        }
        sample = {"seconds": 0.01, "correctness": {"passed": True}}

        async def request(**message):
            return {
                "hello": environment,
                "prepare": {"case": case, "warmup": sample},
                "sample": sample,
                "finish": {"state": "completed"},
            }[message["operation"]]

        worker = SimpleNamespace(request=request, close=AsyncMock())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                measure.Worker, "start", AsyncMock(return_value=worker)
            ) as start:
                result = await measure._round(
                    case,
                    {"candidate": (root / "site", root / "source")},
                    {"candidate": release},
                    root / "runs",
                )
            self.assertEqual(start.await_args.kwargs["polars_threads"], 1)
            self.assertEqual(result["environment"]["polars_threads"], 1)
            self.assertEqual(len(result["samples"]["candidate"]), 10)
            worker.close.assert_awaited_once()

    async def test_engine_versions_share_fixture_and_validate_release_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            common = root / "common"
            case = {
                "id": "case",
                "family": "engines",
                "backend": "calc-flow-stream",
            }
            releases = {"baseline": {}, "candidate": {}}
            sites = {side: root / side / "site" for side in releases}
            with (
                patch.object(measure, "ROOT", common),
                patch(
                    "scripts.benchmark_suite.legacy.validate_sources", AsyncMock()
                ) as validate,
                patch.object(measure, "install", AsyncMock(side_effect=sites.values())),
                patch.object(measure, "baseline_case_ids", return_value=frozenset()),
                patch.object(measure, "shard_cases", return_value=[case]),
                patch.object(measure, "_case_order", return_value=[0]),
                patch.object(
                    measure, "measure_case", AsyncMock(return_value={})
                ) as run,
            ):
                await measure.measure_shard(
                    {"id": "engines-10", "family": "engines"},
                    releases,
                    root / "results",
                    baseline,
                )
            validate.assert_awaited_once_with(
                {"baseline": baseline.resolve(), "candidate": common}, releases
            )
            self.assertEqual(
                run.await_args.args[2],
                {side: (site, common) for side, site in sites.items()},
            )
