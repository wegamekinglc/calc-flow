"""Collect adjacent case invocations and apply the unified paired-median gate."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from scripts.benchmark_suite.identity import compare_identity
from scripts.benchmark_suite.report import ROUNDS, SAMPLES, comparison
from scripts.toolkit import write_json

SIDES = ("baseline", "candidate")
RELEASE_ROUNDS = ROUNDS
RELEASE_SAMPLES = SAMPLES


def pair_order(index: int) -> list[str]:
    return list(SIDES if index % 2 == 0 else reversed(SIDES))


async def collect_case(name: str, measure, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("paired collection requires a fresh evidence directory")
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"id": name, "rounds": []}
    for round_index in range(RELEASE_ROUNDS):
        pairs = []
        evidence["rounds"].append(pairs)
        for index in range(RELEASE_SAMPLES):
            pair = {"pair": index, "order": pair_order(index), "observations": {}}
            pairs.append(pair)
            for side in pair["order"]:
                destination = output / f"round-{round_index}" / f"pair-{index}" / side
                destination.mkdir(parents=True, exist_ok=True)
                try:
                    observation = await measure(side, destination)
                    write_json(destination / "observation.json", observation)
                    pair["observations"][side] = observation
                except Exception as error:
                    write_json(
                        output / "failure.json",
                        {
                            "case": name,
                            "round": round_index,
                            "pair": index,
                            "side": side,
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
                    raise
                finally:
                    write_json(output / "pairs.json", evidence)
    return evidence


def _observation(value: dict, seal: str, reference: dict) -> float:
    if value.get("binary_sha256") != seal:
        raise ValueError("incomparable observation: wrong sealed binary/native SHA")
    if value.get("correctness") is not True:
        raise ValueError("observation correctness was not established")
    compare_identity(reference, value.get("metadata", {}))
    return _sample_median(value.get("samples"))


def _sample_median(samples: object) -> float:
    if (
        not isinstance(samples, list)
        or not samples
        or any(
            type(sample) not in (int, float) or not math.isfinite(sample) or sample <= 0
            for sample in samples
        )
    ):
        raise ValueError("raw samples must be finite and positive")
    return statistics.median(samples)


def _pair_observations(pair: dict, index: int) -> dict:
    if pair.get("pair") != index or pair.get("order") != pair_order(index):
        raise ValueError("duplicate pair or incorrect AB/BA execution order")
    observations = pair.get("observations", {})
    if set(observations) != set(SIDES):
        raise ValueError("incomplete paired observation")
    return observations


def _unique_worker(value: dict, workers: set[str]) -> str:
    worker = value.get("worker")
    if not isinstance(worker, str) or not worker or worker in workers:
        raise ValueError("each invocation requires a fresh isolated worker")
    return worker


def evaluate_case(case: dict, seals: dict) -> dict:
    rounds = case.get("rounds", [])
    if len(rounds) != RELEASE_ROUNDS or any(
        len(pairs) != RELEASE_SAMPLES for pairs in rounds
    ):
        raise ValueError(
            f"release evidence requires {RELEASE_ROUNDS} rounds of "
            f"{RELEASE_SAMPLES} actual pairs"
        )
    values = {side: [[], []] for side in SIDES}
    reference = (
        rounds[0][0].get("observations", {}).get("baseline", {}).get("metadata", {})
    )
    workers = set()
    for round_index, pairs in enumerate(rounds):
        for index, pair in enumerate(pairs):
            observations = _pair_observations(pair, index)
            for side in SIDES:
                value = observations[side]
                workers.add(_unique_worker(value, workers))
                values[side][round_index].append(
                    _observation(value, seals[side], reference)
                )
    return comparison(
        {"id": case["id"], "comparison": "interleaved", "correctness": True, **values},
        minimum_samples=RELEASE_SAMPLES,
    )
