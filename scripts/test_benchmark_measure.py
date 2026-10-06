from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite import measure


class BenchmarkMeasureTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_thread_reference_launches_and_validates_its_own_pool(self):
        case = {"family": "engines", "backend": "polars-1t"}
        release = {"native_sha256": "a" * 64}
        environment = {
            **release,
            "polars_threads": 1,
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
