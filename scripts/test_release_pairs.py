"""The release gate consumes real adjacent observations, never zipped suite blocks."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.toolkit import fingerprint_json


def identity() -> dict:
    values = {
        "machine": {"cpu": "fixture"},
        "dependency": {"python": "3.13"},
        "workload": {
            "scenario": "case",
            "scope": "pytest-native-boundary",
            "backend": "calc-flow",
            "scale": "overhead",
            "input_rows": 1000,
        },
    }
    return {
        key: value
        for name, raw in values.items()
        for key, value in (
            (f"{name}_identity", raw),
            (f"{name}_fingerprint", fingerprint_json(raw)),
        )
    }


def observation(side, seconds=1.0, worker="worker"):
    return {
        "samples": [seconds] * 10,
        "metadata": identity(),
        "binary_sha256": ("a" if side == "baseline" else "b") * 64,
        "worker": worker,
        "correctness": True,
    }


def case(ratios=(1.0, 1.0)):
    return {
        "id": "python/case",
        "rounds": [
            [
                {
                    "pair": p,
                    "order": ["baseline", "candidate"]
                    if p % 2 == 0
                    else ["candidate", "baseline"],
                    "observations": {
                        side: observation(
                            side,
                            ratio if side == "candidate" else 1.0,
                            f"{r}/{p}/{side}",
                        )
                        for side in ("baseline", "candidate")
                    },
                }
                for p in range(10)
            ]
            for r, ratio in enumerate(ratios)
        ],
    }


class ReleasePairTests(unittest.TestCase):
    def test_release_uses_the_reviewed_pr_sampling_plan(self):
        from scripts.benchmark_suite.release_pairs import (
            RELEASE_ROUNDS,
            RELEASE_SAMPLES,
        )
        from scripts.benchmark_suite.report import ROUNDS, SAMPLES

        self.assertEqual((RELEASE_ROUNDS, RELEASE_SAMPLES), (ROUNDS, SAMPLES))

    def evaluate(self, row):
        from scripts.benchmark_suite.release_pairs import evaluate_case

        return evaluate_case(row, {"baseline": "a" * 64, "candidate": "b" * 64})

    def test_two_rounds_use_existing_paired_median_verdict(self):
        for ratios, expected in (
            ((1.1, 1.1), "regression"),
            ((1.1, 1.0), "inconclusive"),
            ((1.0, 1.0), "no-confirmed-regression"),
            ((1.05, 1.05), "no-confirmed-regression"),
            ((0.9, 0.9), "improved"),
        ):
            with self.subTest(ratios=ratios):
                raw = case(ratios)
                before = copy.deepcopy(raw)
                self.assertEqual(self.evaluate(raw)["verdict"], expected)
                self.assertEqual(raw, before)

    def test_corrupt_pairing_or_identity_never_receives_verdict(self):
        mutations = [
            lambda c: c["rounds"].pop(),
            lambda c: c["rounds"][0].pop(),
            lambda c: c["rounds"][0][1].update(pair=0),
            lambda c: c["rounds"][0][1].update(order=["baseline", "candidate"]),
            lambda c: c["rounds"][1][0]["observations"]["baseline"].update(
                worker="0/0/baseline"
            ),
            lambda c: c["rounds"][0][0]["observations"]["candidate"].update(
                binary_sha256="c" * 64
            ),
            lambda c: c["rounds"][0][0]["observations"]["baseline"].update(
                samples=[float("nan")]
            ),
            lambda c: c["rounds"][0][0]["observations"]["baseline"].update(metadata={}),
            lambda c: c["rounds"][0][0]["observations"]["baseline"].update(
                correctness=False
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                raw = case()
                mutate(raw)
                with self.assertRaises(ValueError):
                    self.evaluate(raw)


class ReleaseCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_collector_executes_adjacent_alternating_pairs_and_persists_raw(self):
        from scripts.benchmark_suite.release_pairs import collect_case

        calls = []

        async def measure(side, destination):
            calls.append(side)
            return observation(side, worker=str(destination))

        with TemporaryDirectory() as raw:
            path = Path(raw)
            result = await collect_case("python/case", measure, path)
            expected = [
                side
                for _ in range(2)
                for pair in range(10)
                for side in (
                    ("baseline", "candidate")
                    if pair % 2 == 0
                    else ("candidate", "baseline")
                )
            ]
            self.assertEqual(calls, expected)
            self.assertEqual(len(list(path.rglob("observation.json"))), 40)
            self.assertEqual(result, json.loads((path / "pairs.json").read_text()))

    async def test_failed_measurement_preserves_partial_evidence(self):
        from scripts.benchmark_suite.release_pairs import collect_case

        async def measure(side, destination):
            (destination / "raw.json").write_text('{"started": true}')
            if side == "candidate":
                raise RuntimeError("measurement exited 9")
            return observation(side, worker=str(destination))

        with TemporaryDirectory() as raw:
            path = Path(raw)
            with self.assertRaisesRegex(RuntimeError, "exited 9"):
                await collect_case("python/case", measure, path)
            self.assertEqual(len(list(path.rglob("raw.json"))), 2)
            failure = json.loads((path / "failure.json").read_text())
            self.assertIn("exited 9", failure["error"])
            self.assertEqual(failure["side"], "candidate")
            self.assertTrue((path / "pairs.json").is_file())


class ReleasePairReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_pair_evidence_cannot_be_overwritten_by_a_retry(self):
        from unittest.mock import AsyncMock

        from scripts.benchmark_suite.release_pairs import collect_case

        with TemporaryDirectory() as raw:
            root = Path(raw)
            evidence = root / "pairs.json"
            evidence.write_text('{"original": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fresh"):
                await collect_case(
                    "case", AsyncMock(return_value=observation("baseline")), root
                )
            self.assertEqual(json.loads(evidence.read_text()), {"original": True})
