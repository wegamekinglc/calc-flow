"""Same-host, release-bound, alternating version measurements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.benchmark_suite.catalog import CONTRACT, THREADS, shard_cases
from scripts.benchmark_suite.process import Worker, install
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.report import ROUNDS, SAMPLES, comparison


def validate_environment(environment: dict, release: dict) -> dict:
    if environment["native_sha256"] != release["native_sha256"]:
        raise ValueError(
            "worker loaded a different native module than the release wheel"
        )
    if environment["polars_threads"] != THREADS or environment[
        "tokio_worker_threads"
    ] != str(THREADS):
        raise ValueError("worker thread configuration does not match the catalog")
    return {key: value for key, value in environment.items() if key != "native_sha256"}


def validate_sample(sample: dict) -> float:
    seconds = sample["seconds"]
    if (
        type(seconds) not in (int, float)
        or not np.isfinite(seconds)
        or seconds <= 0
        or sample["correctness"]["passed"] is not True
    ):
        raise ValueError("invalid timing or failed correctness result")
    return seconds


async def _prepare(workers: dict, releases: dict, case: dict) -> dict:
    identities = {}
    for side, worker in workers.items():
        identities[side] = validate_environment(
            await worker.request(operation="hello"), releases[side]
        )
        response = await worker.request(operation="prepare", case=case)
        if response["case"] != case:
            raise ValueError("worker prepared a different workload")
        validate_sample(response["warmup"])
    if "baseline" in identities and identities["baseline"] != identities["candidate"]:
        raise ValueError("base/head machine, dependency or thread fingerprints differ")
    return identities["candidate"]


async def _samples(workers: dict) -> dict:
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
            collected[side].append(sample)
        if len(workers) == 2:
            _check_latest_cursors(collected)
    return collected


def _check_latest_cursors(collected: dict) -> None:
    starts = [samples[-1].get("start_row") for samples in collected.values()]
    if starts[0] != starts[1]:
        raise ValueError("warm base/head sample cursors differ")


async def _round(case: dict, sites: dict, releases: dict, root: Path) -> dict:
    workers = {}
    try:
        for side, site in sites.items():
            workers[side] = await Worker.start(site, root / side)
        environment = await _prepare(workers, releases, case)
        samples = await _samples(workers)
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


async def measure_case(case: dict, sites: dict, releases: dict, root: Path) -> dict:
    external = not case["backend"].startswith("calc-flow")
    selected = {"candidate": sites["candidate"]} if external else sites
    evidence = []
    try:
        for index in range(ROUNDS):
            evidence.append(
                await _round(case, selected, releases, root / f"round-{index}")
            )
        if evidence[0]["environment"] != evidence[1]["environment"]:
            raise ValueError("confirmation-round environment changed")
        row = _measured_row(case, evidence, external)
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


def _measured_row(case: dict, evidence: list[dict], external: bool) -> dict:
    return {
        **case,
        "status": "ok",
        "correctness": True,
        "comparison": "external" if external else "interleaved",
        "baseline": [] if external else _sample_seconds(evidence, "baseline"),
        "candidate": _sample_seconds(evidence, "candidate"),
        "evidence": evidence,
    }


async def measure_shard(shard: dict, releases: dict, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    sites = {
        side: await install(release, root / side) for side, release in releases.items()
    }
    cases = shard_cases(shard)
    report = {
        "contract": CONTRACT,
        "harness_sha256": harness_sha256(),
        "shard": shard,
        "releases": releases,
        "cases": [],
        "errors": [],
    }
    order = np.random.default_rng(20260905).permutation(len(cases))
    for index in order:
        case = cases[int(index)]
        print(f"Measuring {case['id']}", flush=True)
        row = await measure_case(case, sites, releases, root / f"case-{index}")
        report["cases"].append(row)
        (root / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report
