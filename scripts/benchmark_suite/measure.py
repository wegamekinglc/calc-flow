"""Same-host, release-bound, alternating version measurements."""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.benchmark_suite.catalog import (
    CONTRACT,
    STREAM_EVIDENCE_FIELDS,
    THREADS,
    baseline_case_ids,
    comparison_kind,
    polars_thread_count,
    shard_cases,
)
from scripts.benchmark_suite.process import ROOT, Worker, install
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.report import ROUNDS, SAMPLES, comparison


def validate_environment(
    environment: dict, release: dict, *, polars_threads: int = THREADS
) -> dict:
    if environment["native_sha256"] != release["native_sha256"]:
        raise ValueError(
            "worker loaded a different native module than the release wheel"
        )
    if environment["polars_threads"] != polars_threads or environment[
        "tokio_worker_threads"
    ] != str(THREADS):
        raise ValueError("worker thread configuration does not match the catalog")
    return {key: value for key, value in environment.items() if key != "native_sha256"}


def validate_sample(sample: dict) -> float:
    seconds = sample["seconds"]
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or seconds <= 0
        or sample["correctness"]["passed"] is not True
    ):
        raise ValueError("invalid timing or failed correctness result")
    return seconds


def validate_stream_sample(case: dict, sample: dict) -> None:
    """Require original adapter evidence for every replay-backed dimension."""

    if case.get("backend") != "calc-flow-stream" or "batch_rows" not in case:
        return
    evidence = sample.get("stream_evidence")
    if not isinstance(evidence, dict) or any(
        key not in evidence or evidence[key] != case[key]
        for key in STREAM_EVIDENCE_FIELDS
    ):
        raise ValueError("stream evidence dimensions differ from the prepared case")
    if case["checkpoint_interval_millis"] is not None:
        epochs = evidence.get("nonterminal_epochs")
        rows = evidence.get("rows_before_checkpoint")
        if (
            not isinstance(epochs, list)
            or not epochs
            or any(type(epoch) is not int or epoch < 1 for epoch in epochs)
            or type(rows) is not int
            or not 0 < rows < case["rows"]
            or evidence.get("recovery") != "verified"
            or sample["seconds"] < case["checkpoint_interval_millis"] / 1000
        ):
            raise ValueError(
                "checkpoint evidence lacks durable nonterminal recovery proof"
            )


async def _prepare(workers: dict, releases: dict, case: dict) -> dict:
    identities = {}
    for side, worker in workers.items():
        identities[side] = validate_environment(
            await worker.request(operation="hello"),
            releases[side],
            polars_threads=polars_thread_count(case),
        )
        response = await worker.request(operation="prepare", case=case)
        if response["case"] != case:
            raise ValueError("worker prepared a different workload")
        validate_sample(response["warmup"])
        validate_stream_sample(case, response["warmup"])
    if "baseline" in identities and identities["baseline"] != identities["candidate"]:
        raise ValueError("base/head machine, dependency or thread fingerprints differ")
    return identities["candidate"]


async def _samples(workers: dict, case: dict) -> dict:
    collected = {side: [] for side in workers}
    for index in range(SAMPLES):
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        for side in order:
            if side not in workers:
                continue
            sample = await workers[side].request(operation="sample")
            validate_sample(sample)
            validate_stream_sample(case, sample)
            collected[side].append(sample)
        if len(workers) == 2:
            _check_latest_cursors(collected)
    return collected


def _check_latest_cursors(collected: dict) -> None:
    starts = [samples[-1].get("start_row") for samples in collected.values()]
    if starts[0] != starts[1]:
        raise ValueError("warm base/head sample cursors differ")


async def _round(case: dict, workers_by_side: dict, releases: dict, root: Path) -> dict:
    workers = {}
    try:
        for side, (site, source) in workers_by_side.items():
            workers[side] = await Worker.start(
                site,
                root / side,
                source=source,
                polars_threads=polars_thread_count(case),
            )
        environment = await _prepare(workers, releases, case)
        samples = await _samples(workers, case)
        completion = {}
        for side, worker in workers.items():
            result = await worker.request(operation="finish")
            if result["state"] != "completed":
                raise ValueError("benchmark worker did not complete")
            completion[side] = result
        return {
            "environment": environment,
            "samples": samples,
            "completion": completion,
            "native_sha256": {
                side: releases[side]["native_sha256"] for side in workers
            },
        }
    finally:
        for worker in workers.values():
            await worker.close()


async def measure_case(
    case: dict, kind: str, workers_by_side: dict, releases: dict, root: Path
) -> dict:
    selected = (
        workers_by_side
        if kind == "interleaved"
        else {"candidate": workers_by_side["candidate"]}
    )
    evidence = []
    try:
        for index in range(ROUNDS):
            evidence.append(
                await _round(case, selected, releases, root / f"round-{index}")
            )
        if evidence[0]["environment"] != evidence[1]["environment"]:
            raise ValueError("confirmation-round environment changed")
        row = _measured_row(case, evidence, kind)
        return {**row, "result": comparison(row)}
    except Exception as error:
        return {
            **case,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "evidence": evidence,
        }


def _sample_seconds(evidence: list[dict], side: str) -> list[list[float]]:
    return [
        [sample["seconds"] for sample in round_["samples"][side]] for round_ in evidence
    ]


def _measured_row(case: dict, evidence: list[dict], kind: str) -> dict:
    return {
        **case,
        "status": "ok",
        "correctness": True,
        "comparison": kind,
        "baseline": (
            _sample_seconds(evidence, "baseline") if kind == "interleaved" else []
        ),
        "candidate": _sample_seconds(evidence, "candidate"),
        "evidence": evidence,
    }


def _case_order(count: int) -> list[int]:
    import numpy as np

    return np.random.default_rng(20260905).permutation(count).tolist()


async def measure_shard(
    shard: dict, releases: dict, root: Path, baseline_source: Path | None
) -> dict:
    from scripts.benchmark_suite.legacy import validate_sources

    if baseline_source is None:
        raise ValueError("paired comparison requires a baseline source checkout")
    sources = {"baseline": baseline_source.resolve(), "candidate": ROOT}
    await validate_sources(sources, releases)

    root.mkdir(parents=True, exist_ok=True)
    sites = {
        side: await install(release, root / side) for side, release in releases.items()
    }
    workers_by_side = {
        side: (site, ROOT if shard["family"] == "engines" else sources[side])
        for side, site in sites.items()
    }
    baseline_ids = baseline_case_ids(baseline_source, shard)
    cases = shard_cases(shard)
    report = {
        "contract": CONTRACT,
        "harness_sha256": harness_sha256(),
        "shard": shard,
        "releases": releases,
        "baseline_case_ids": None if baseline_ids is None else sorted(baseline_ids),
        "cases": [],
        "errors": [],
    }
    order = _case_order(len(cases))
    for index in order:
        case = cases[int(index)]
        print(f"Measuring {case['id']}", flush=True)
        kind = comparison_kind(case, baseline_ids)
        row = await measure_case(
            case, kind, workers_by_side, releases, root / f"case-{index}"
        )
        report["cases"].append(row)
        (root / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report
