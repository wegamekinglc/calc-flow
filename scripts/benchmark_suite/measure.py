"""Same-host, release-bound, alternating version measurements."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_suite.catalog import (
    CONTRACT,
    THREADS,
    comparison_kind,
    shard_cases,
)
from scripts.benchmark_suite.process import Worker, install
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.report import ROUNDS, SAMPLES, comparison


def _baseline_catalog_constants(
    catalog_path: Path,
) -> dict[str, tuple[str, ...]] | None:
    """Read the baseline catalog's declarative tuples without executing code.

    Only literal string-tuple assignments are accepted; anything else fails
    closed so an unparseable baseline keeps every paired case gated.
    """

    tree = ast.parse(catalog_path.read_text(encoding="utf-8"))
    constants: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _declarative_tuple(node.value, constants)
        if value is not None:
            constants[target.id] = value
    required = ("ROW_SCALES", "SQL_CASES", "ROLLING_CASES")
    if any(name not in constants for name in required):
        return None
    return constants


def _declarative_tuple(
    node: ast.expr, constants: dict[str, tuple[str, ...]]
) -> tuple[str, ...] | None:
    """Accept a literal string tuple or the catalog's derived forms."""

    try:
        literal = ast.literal_eval(node)
    except ValueError:
        literal = None
    if isinstance(literal, tuple) and all(isinstance(item, str) for item in literal):
        return literal
    sliced = _sliced_tuple(node, constants)
    if sliced is not None:
        return sliced
    return _powers_of_ten_tuple(node)


def _sliced_tuple(
    node: ast.expr, constants: dict[str, tuple[str, ...]]
) -> tuple[str, ...] | None:
    """Match ``KNOWN_TUPLE[lower:upper]`` over already-parsed constants."""

    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    source = constants.get(node.value.id)
    if source is None:
        return None
    part = node.slice
    if not isinstance(part, ast.Slice):
        return None
    bounds = []
    for component in (part.lower, part.upper, part.step):
        if component is None:
            bounds.append(None)
            continue
        try:
            bounds.append(ast.literal_eval(component))
        except ValueError:
            return None
    lower, upper, step = bounds
    return tuple(source[lower:upper:step])


def _powers_of_ten_tuple(node: ast.expr) -> tuple[str, ...] | None:
    """Match ``tuple(10**power for power in range(start, stop))`` exactly."""

    generator = _generator_argument(node)
    if generator is None:
        return None
    power = generator.elt
    if not isinstance(power, ast.BinOp) or not isinstance(power.op, ast.Pow):
        return None
    if not isinstance(power.left, ast.Constant) or power.left.value != 10:
        return None
    if not isinstance(power.right, ast.Name):
        return None
    comprehension = generator.generators[0]
    if power.right.id != comprehension.target.id:
        return None
    bounds = _constant_range_bounds(comprehension.iter)
    if bounds is None:
        return None
    start, stop = bounds
    return tuple(str(10**exponent) for exponent in range(start, stop))


def _generator_argument(node: ast.expr) -> ast.GeneratorExp | None:
    """Return the sole generator argument of a ``tuple(...)`` call."""

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "tuple" or len(node.args) != 1 or node.keywords:
        return None
    generator = node.args[0]
    if not isinstance(generator, ast.GeneratorExp) or len(generator.generators) != 1:
        return None
    if generator.generators[0].ifs or generator.generators[0].is_async:
        return None
    return generator


def _constant_range_bounds(node: ast.expr) -> tuple[int, int] | None:
    """Match ``range(constant, constant)`` exactly."""

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "range" or node.keywords or len(node.args) != 2:
        return None
    try:
        start, stop = (ast.literal_eval(argument) for argument in node.args)
    except ValueError:
        return None
    if isinstance(start, int) and isinstance(stop, int):
        return (start, stop)
    return None


def _baseline_engine_ids(constants: dict[str, tuple[str, ...]]) -> frozenset[str]:
    sql, rolling = constants["SQL_CASES"], constants["ROLLING_CASES"]
    stream = constants.get("STREAM_CASES", rolling)
    columns = (
        ("calc-flow-sql", sql),
        ("datafusion", sql),
        ("polars", sql),
        ("calc-flow-stream", stream),
        ("ta-lib", rolling),
    )
    return frozenset(
        f"engines/{rows}/{backend}/{scenario}"
        for rows in constants["ROW_SCALES"]
        for backend, scenarios in columns
        for scenario in scenarios
    )


def _baseline_warm_ids(constants: dict[str, tuple[str, ...]]) -> frozenset[str]:
    scales = constants["ROW_SCALES"]
    dense = "1000000" in scales
    ids = set()
    for scale in scales:
        appends = (
            (1, 4, 16, 64, 640, 6_400, 64_000)
            if dense and scale == "1000000"
            else (64,)
        )
        for append in appends:
            for scenario in constants["ROLLING_CASES"]:
                ids.add(f"warm/{scale}/{append}/{scenario}")
    return frozenset(ids)


def baseline_case_ids(
    baseline_source: Path | None, shard: dict
) -> frozenset[str] | None:
    """Resolve the baseline catalog's case ids for one shard family."""

    if baseline_source is None:
        return None
    catalog_path = baseline_source / "scripts" / "benchmark_suite" / "catalog.py"
    if not catalog_path.is_file():
        return None
    constants = _baseline_catalog_constants(catalog_path)
    if constants is None:
        return None
    if shard["family"] == "engines":
        return _baseline_engine_ids(constants)
    if shard["family"] == "warm":
        return _baseline_warm_ids(constants)
    return None


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


async def measure_case(
    case: dict, kind: str, sites: dict, releases: dict, root: Path
) -> dict:
    selected = sites if kind == "interleaved" else {"candidate": sites["candidate"]}
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


async def measure_shard(
    shard: dict, releases: dict, root: Path, baseline_source: Path | None
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    sites = {
        side: await install(release, root / side) for side, release in releases.items()
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
    order = np.random.default_rng(20260905).permutation(len(cases))
    for index in order:
        case = cases[int(index)]
        print(f"Measuring {case['id']}", flush=True)
        kind = comparison_kind(case, baseline_ids)
        row = await measure_case(case, kind, sites, releases, root / f"case-{index}")
        report["cases"].append(row)
        (root / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report
