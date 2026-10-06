from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite import measure


class BenchmarkMeasureTests(unittest.IsolatedAsyncioTestCase):
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
