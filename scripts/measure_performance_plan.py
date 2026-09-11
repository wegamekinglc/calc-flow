"""Exact-wheel paired evidence for the seven-priority performance implementation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from benchmarks.performance_diagnostics import (
    FALLBACK_VARIANTS,
    _fallback_batch_size,
    diagnostic_sql,
)
from scripts.benchmark_suite.catalog import THREADS
from scripts.benchmark_suite.measure import validate_sample
from scripts.benchmark_suite.process import ROOT, Worker
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.report import comparison
from scripts.benchmark_suite.statistics import paired_round
from scripts.profile_warm_stream import _wheel_native_sha256


@dataclass(frozen=True, slots=True)
class ReleasePair:
    """Matched release manifests and their worker import locations."""

    sites: dict
    releases: dict


@dataclass(frozen=True, slots=True)
class SampleValidation:
    """The fixed workload and identity checks shared by a round's samples."""

    case: dict
    revision_comparison: bool
    declaration: dict | None


def _compare_numeric(
    first: pa.ChunkedArray, second: pa.ChunkedArray, name: str
) -> float:
    a, b = first.to_numpy(), second.to_numpy()
    for classify in (np.isnan, np.isposinf, np.isneginf):
        if not np.array_equal(classify(a), classify(b)):
            raise ValueError(f"revision numeric classification changed: {name}")
    finite = np.isfinite(a)
    np.testing.assert_allclose(a[finite], b[finite], rtol=1e-10, atol=1e-10)
    return float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0


def _compare_column(
    first: pa.ChunkedArray, second: pa.ChunkedArray, name: str
) -> float:
    if not first.is_null().equals(second.is_null()):
        raise ValueError(f"revision output validity changed: {name}")
    numeric = name in ("value", "moving_average", "dual_sma_spread")
    if numeric and pa.types.is_floating(first.type):
        return _compare_numeric(first, second, name)
    if not first.equals(second):
        raise ValueError(f"revision output payload or delivered order changed: {name}")
    return 0.0


def compare_outputs(left_path: str, right_path: str) -> dict:
    """Compare revisions directly, retaining validity and special-value distinctions."""
    with (
        pa.memory_map(left_path, "r") as left_file,
        pa.memory_map(right_path, "r") as right_file,
    ):
        left = pa.ipc.open_file(left_file).read_all()
        right = pa.ipc.open_file(right_file).read_all()
        if not left.schema.equals(right.schema, check_metadata=True):
            raise ValueError("revision output schema or metadata changed")
        if left.num_rows != right.num_rows:
            raise ValueError("revision output row count changed")
        maximum = 0.0
        for name in left.column_names:
            maximum = max(maximum, _compare_column(left[name], right[name], name))
        return {"passed": True, "rows": left.num_rows, "max_abs_error": maximum}


def _engine_case(
    backend: str, scenario: str, rows: int, *, diagnostic: bool = False
) -> dict:
    sql = backend == "calc-flow-sql"
    return {
        "id": f"{backend}/{scenario}/{rows}",
        "family": "sql-diagnostic" if diagnostic else "engine-diagnostic",
        "backend": backend,
        "scenario": scenario,
        "rows": rows,
        "scope": "execute-to-arrow" if sql else "ready-enqueue-to-arrow",
    }


def _warm_case(
    history: int,
    append: int,
    entities: int,
    batch: int,
    window: int = 20,
) -> dict:
    config = {
        "history_rows": history,
        "append_rows": append,
        "entities": entities,
        "history_segment_rows": batch,
        "window": window,
        "fast_window": min(5, window),
        "indicator": "rolling_mean",
        "append_entities": None,
    }
    return {
        "id": (f"warm/h{history}/a{append}/e{entities}/b{batch}/w{window}/activeNone"),
        "family": "native-diagnostic",
        "backend": "calc-flow-stream",
        "scenario": "sma20" if window == 20 else f"mean{window}",
        "rows": append,
        "scope": "warm-enqueue-to-arrow-partial-window",
        "config": config,
    }


def _sparse_warm_case(history: int, append: int) -> dict:
    case = _warm_case(history, append, 64, 64_000)
    return {
        **case,
        "id": case["id"].removesuffix("None") + "1",
        "config": {**case["config"], "append_entities": 1},
    }


def _core_inventory() -> list[dict]:
    return [
        *(
            _engine_case("calc-flow-sql", scenario, rows, diagnostic=True)
            for rows in (10, 1_000, 1_000_000, 10_000_000)
            for scenario in ("filter", "filter_uint64_modulo", "sma20", "dual_sma")
        ),
        *(
            _engine_case("calc-flow-sql", scenario, rows)
            for rows in (1_000_000, 10_000_000)
            for scenario in ("projection", "group_by", "join")
        ),
        *(
            _engine_case("calc-flow-stream", scenario, rows)
            for rows in (10, 1_000, 100_000, 1_000_000, 10_000_000)
            for scenario in ("sma20", "dual_sma")
        ),
    ]


def inventory() -> dict[str, list[dict]]:
    tail = [
        _warm_case(64_000, 1, 1, 64_000),
        _warm_case(1_024_000, 64, 64, 64_000),
    ]
    sensitivity = [
        _warm_case(64_000, 1_024, 1, 1_024),
        _warm_case(64_000, 8_192, 64, 8_192),
        _warm_case(64_000, 64_000, 1_000, 64_000),
        _warm_case(1_024_000, 256_000, 64, 256_000),
        *(_warm_case(64_000, 64_000, 64, 64_000, window) for window in (1, 5, 20, 512)),
        *(
            _sparse_warm_case(history, append)
            for history in (64_000, 1_024_000)
            for append in (1, 64)
        ),
        _warm_case(1_024_000, 64_000, 64, 64_000),
        *tail,
    ]
    return {"core": _core_inventory(), "sensitivity": sensitivity, "tail": tail}


def entity_parallel_inventory() -> list[dict]:
    """Declare the dual-mean target and single-mean control outside older groups."""
    base = _warm_case(64_000, 64_000, 64, 64_000)
    return [
        {
            **base,
            "id": f"entity-parallel/{scenario}/h64000/a64000/e64/b64000/w{windows}",
            "scenario": scenario,
            "config": {**base["config"], "indicator": indicator},
        }
        for scenario, indicator, windows in (
            ("dual_sma", "dual_sma_spread", "5-20"),
            ("sma20", "rolling_mean", "20"),
        )
    ]


def fallback_inventory() -> list[dict]:
    """Keep supplemental SQL costs outside the original all/core/P7 inventory."""
    return [
        {
            **_engine_case("calc-flow-sql", scenario, rows, diagnostic=True),
            "id": f"calc-flow-sql/fallback-{variant}/{scenario}/{rows}",
            "diagnostic_variant": variant,
            "enable_rolling_rewrite": variant != "rewrite_disabled",
            **(
                {"input_layout": "single_batch", "batch_size": max(8192, rows)}
                if variant == "map_partition_single_batch"
                else {}
            ),
        }
        for variant in FALLBACK_VARIANTS
        for scenario in ("sma20", "dual_sma")
        for rows in (1_000, 1_000_000)
    ]


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_release(path: Path) -> tuple[dict, Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["profile"] != "release" or manifest["tracked_source_clean"] is not True:
        raise ValueError(
            "performance comparisons require a clean, provenance-bound release build"
        )
    if any(
        not re.fullmatch("[0-9a-f]{40}", manifest.get(field, ""))
        for field in ("source_sha", "source_tree")
    ):
        raise ValueError("release requires exact source commit and tree identities")
    for field in ("wheel", "native"):
        if _sha256(Path(manifest[field])) != manifest[f"{field}_sha256"]:
            raise ValueError(f"release {field} hash differs from its build manifest")
    if _wheel_native_sha256(Path(manifest["wheel"])) != manifest["native_sha256"]:
        raise ValueError("extracted native module does not match the recorded wheel")
    return manifest, Path(manifest["native"]).parent.parent


def _compare_latest(collected: dict) -> dict:
    left, right = (collected[side][-1] for side in ("baseline", "candidate"))
    if left.get("start_row") != right.get("start_row"):
        raise ValueError("revision workers advanced different warm cursors")
    result = compare_outputs(left["comparison_output"], right["comparison_output"])
    for sample in (left, right):
        Path(sample.pop("comparison_output")).unlink()
        sample["revision_equivalence"] = result
    return result


def _record(path: Path, **event) -> None:
    with path.open("a", encoding="utf-8") as journal:
        journal.write(json.dumps(event, allow_nan=False) + "\n")


def _validate_callback_counts(callback: dict) -> None:
    if any(type(value) is not int or value < 0 for value in callback.values()):
        raise ValueError("rolling callback metrics require nonnegative integers")
    if callback["started"] != callback["succeeded"] or any(
        callback[outcome] for outcome in ("failed", "cancelled", "interrupted")
    ):
        raise ValueError("completed rolling callback outcomes are inconsistent")


def _validate_callback(callback: dict) -> None:
    _validate_callback_counts(callback)
    stages = (
        "input_validation",
        "ordering_proof",
        "entity_resolution",
        "state_preparation",
        "numeric_update",
        "history_maintenance",
        "arrow_output",
        "budget_preparation",
        "send_wait",
        "other",
    )
    if callback["callback_duration_ns"] != sum(
        callback[f"{stage}_duration_ns"] for stage in stages
    ):
        raise ValueError("rolling callback stages do not cover its duration")


def _validate_completion(completion: dict) -> None:
    if completion["state"] != "completed":
        raise ValueError("diagnostic job did not complete")
    status = completion.get("after_status", {})
    if status.get("metrics_overflowed", False) is not False:
        raise ValueError("terminal metrics overflowed")
    for node in status.get("rolling_metrics", {}).values():
        if node["overflowed"] is not False:
            raise ValueError("terminal rolling metrics overflowed")
        for name in ("data", "watermark", "end"):
            _validate_callback(node[name])


def _fallback_path(query: dict) -> dict:
    return {
        **{
            name: query.get(name)
            for name in (
                "configured_batch_size",
                "configured_target_partitions",
                "requested_target_partitions",
            )
        },
        **{
            name: query[name]
            for name in (
                "rolling_rewrite_enabled",
                "rolling_candidate_windows",
                "rolling_rewritten_windows",
                "rolling_fallback_reasons",
            )
        },
        "window_nodes": sorted(
            re.findall(
                r"\b(?:BoundedWindowAggExec|WindowAggExec)\b", query["physical_plan"]
            )
        ),
    }


def _validate_fallback_config(case: dict, query: dict, side: str) -> None:
    expected_config = {
        "configured_batch_size": _fallback_batch_size(case),
        "configured_target_partitions": 32,
        "requested_target_partitions": 32,
    }
    if any(
        type(query.get(name)) is not int or query[name] != value
        for name, value in expected_config.items()
    ):
        raise ValueError(f"{side} SQL fallback configuration metric changed or missing")


def _fallback_reasons_valid(variant: str, side: str, reasons: list, count: int) -> bool:
    if variant == "rewrite_disabled":
        return reasons == []
    if side == "baseline":
        return bool(reasons) and set(reasons) == {"window_aggregate_is_not_avg"}
    if variant == "unsupported_count":
        return reasons == ["count_filter_distinct_or_null_treatment_is_not_supported"]
    return len(reasons) == 1 and any(
        reasons[0] == f"physical_window_shape_not_supported:{attempted}_of_{count}"
        for attempted in range(count)
    )


def _fallback_rewrite_metrics_valid(case: dict, path: dict, count: int) -> bool:
    enabled = case["diagnostic_variant"] != "rewrite_disabled"
    if case.get("enable_rolling_rewrite") is not enabled:
        return False
    if path["rolling_rewrite_enabled"] is not enabled:
        return False
    expected = {
        "rolling_candidate_windows": count if enabled else 0,
        "rolling_rewritten_windows": 0,
    }
    return all(
        type(path[name]) is int and path[name] == value
        for name, value in expected.items()
    )


def _validate_fallback_path(case: dict, query: dict, side: str) -> None:
    variant = case["diagnostic_variant"]
    if variant not in FALLBACK_VARIANTS or case["scenario"] not in (
        "sma20",
        "dual_sma",
    ):
        raise ValueError("unsupported SQL fallback variant")
    _validate_fallback_config(case, query, side)
    count = (2 if case["scenario"] == "sma20" else 3) + (variant == "unsupported_count")
    path = _fallback_path(query)
    valid = _fallback_reasons_valid(
        variant, side, path["rolling_fallback_reasons"], count
    )
    if not _fallback_rewrite_metrics_valid(case, path, count):
        valid = False
    if not path["window_nodes"] or "CalcFlowRollingExec" in query["physical_plan"]:
        valid = False
    if not valid:
        raise ValueError(
            f"{side} SQL fallback did not use its declared DataFusion path"
        )


def _fallback_batch_rows(rows: int, single_batch: bool) -> list[int]:
    if single_batch:
        return [rows]
    return [64_000] * (rows // 64_000) + ([rows % 64_000] if rows % 64_000 else [])


def _fixture_dimensions_valid(
    fixture: dict, rows: int, entities: int, full: int
) -> bool:
    expected = {"rows": rows, "entities": entities, "full_window_rows": full}
    if any(fixture.get(name) != value for name, value in expected.items()):
        return False
    return all(
        re.fullmatch("[0-9a-f]{64}", fixture.get(name, ""))
        for name in ("input_sha256", "schema_sha256")
    )


def _fixture_layout_valid(fixture: dict, rows: int, single_batch: bool) -> bool:
    batch_rows = _fallback_batch_rows(rows, single_batch)
    expected = {
        "input_layout": "single_batch" if single_batch else "engine_batches",
        "batch_rows": batch_rows,
        "native_batch_rows": batch_rows,
    }
    if any(fixture.get(name) != value for name, value in expected.items()):
        return False
    if any(type(size) is not int or size <= 0 for size in fixture["batch_rows"]):
        return False
    return sum(fixture["batch_rows"]) == rows


def _fallback_fixture(case: dict, sample: dict) -> dict:
    fixture = sample.get("fixture", {})
    rows = case["rows"]
    single_batch = case["diagnostic_variant"] == "map_partition_single_batch"
    entities = min(64, max(1, rows // 40))
    full = rows - 19 * entities
    if sample.get("query") != diagnostic_sql(
        case["scenario"], case["diagnostic_variant"]
    ):
        raise ValueError("SQL fallback query differs from the closed variant")
    if (
        not _fixture_dimensions_valid(fixture, rows, entities, full)
        or not _fixture_layout_valid(fixture, rows, single_batch)
        or sample.get("full_window_validation")
        != {"passed": True, "full_window_rows": full, "null_rows": rows - full}
    ):
        raise ValueError("SQL fallback fixture/full-window evidence is incomplete")
    return {"query": sample["query"], "fixture": fixture}


def _validate_fallback_declaration(
    case: dict, sample: dict, side: str, declaration: dict
) -> None:
    actual = _fallback_fixture(case, sample)
    if any(actual[name] != declaration.get(name) for name in ("query", "fixture")):
        raise ValueError(
            "SQL fallback fixture/query differs from the frozen declaration"
        )
    if _fallback_path(sample["datafusion_metrics"][0]) != declaration["paths"].get(
        side
    ):
        raise ValueError(
            f"{side} SQL fallback path differs from the frozen declaration"
        )


def _validate_sma_path(query: dict, scenario: str) -> None:
    expected = 2 if scenario == "sma20" else 3
    if (
        query["rolling_rewritten_windows"] != expected
        or query["rolling_fallback_reasons"]
        or "CalcFlowRollingExec" not in query["physical_plan"]
    ):
        raise ValueError(
            "candidate full-window SQL did not use the required rolling path"
        )


def _validate_optimized_path(
    case: dict, sample: dict, *, side: str = "candidate"
) -> None:
    if case.get("family") != "sql-diagnostic":
        return
    metrics = sample["datafusion_metrics"]
    if len(metrics) != 1:
        raise ValueError("SQL diagnostic requires exactly one query metric")
    query = metrics[0]
    if "diagnostic_variant" in case:
        _validate_fallback_path(case, query, side)
        return
    scenario = case["scenario"]
    if scenario in ("sma20", "dual_sma"):
        _validate_sma_path(query, scenario)
    elif scenario in ("filter", "filter_uint64_modulo") and (
        "Decimal128" in query["physical_plan"] or "%" not in query["physical_plan"]
    ):
        raise ValueError("candidate filter did not retain a UInt64 modulo predicate")


def _validate_fallback_sample(
    case: dict, sample: dict, side: str, declaration: dict | None, *, warmup: bool
) -> None:
    if warmup:
        _fallback_fixture(case, sample)
    if not warmup or declaration is not None:
        _validate_fallback_declaration(case, sample, side, declaration)


def _validate_case_sample(
    validation: SampleValidation,
    sample: dict,
    side: str,
    *,
    warmup: bool = False,
) -> None:
    validate_sample(sample)
    case = validation.case
    fallback = "diagnostic_variant" in case
    if fallback or (side == "candidate" and validation.revision_comparison):
        _validate_optimized_path(case, sample, side=side)
    if fallback:
        _validate_fallback_sample(
            case, sample, side, validation.declaration, warmup=warmup
        )


async def _prepare_workers(
    workers: dict,
    pair: ReleasePair,
    validation: SampleValidation,
    journal: Path,
) -> tuple[dict, dict]:
    identities, warmups = {}, {}
    for side, worker in workers.items():
        identity = await worker.request(operation="hello", scope="core")
        _record(journal, operation="hello", side=side, response=identity)
        if identity.pop("native_sha256") != pair.releases[side]["native_sha256"]:
            raise ValueError("worker loaded a different native binary")
        if identity["tokio_worker_threads"] != str(THREADS):
            raise ValueError("worker thread configuration differs from the catalog")
        identities[side] = identity
        prepared = await worker.request(operation="prepare", case=validation.case)
        _record(journal, operation="prepare", side=side, response=prepared)
        if prepared["case"] != validation.case:
            raise ValueError("worker prepared a different workload")
        _validate_case_sample(validation, prepared["warmup"], side, warmup=True)
        warmups[side] = [prepared["warmup"]]
    return identities, warmups


def _validate_environments(identities: dict, declaration: dict | None) -> None:
    if identities["baseline"] != identities["candidate"]:
        raise ValueError("revision machine/dependency configurations differ")
    if declaration is not None and identities["candidate"] != declaration.get(
        "environment"
    ):
        raise ValueError("SQL fallback environment differs from the frozen declaration")


def _freeze_fallback(case: dict, warmups: dict, environment: dict) -> dict:
    if "diagnostic_variant" not in case:
        return {}
    first = _fallback_fixture(case, warmups["baseline"][0])
    if first != _fallback_fixture(case, warmups["candidate"][0]):
        raise ValueError("revision SQL fallback fixture/query differs")
    return {
        "fallback_preflight": {
            **first,
            "environment": environment,
            "paths": {
                side: _fallback_path(samples[0]["datafusion_metrics"][0])
                for side, samples in warmups.items()
            },
        }
    }


async def _sample_pairs(
    workers: dict,
    validation: SampleValidation,
    count: int,
    journal: Path,
) -> dict:
    collected = {side: [] for side in workers}
    for index in range(count):
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        for side in order:
            sample = await workers[side].request(operation="sample")
            _record(journal, operation="sample", side=side, pair=index, response=sample)
            _validate_case_sample(validation, sample, side)
            collected[side].append(sample)
        _record(
            journal,
            operation="equivalence",
            pair=index,
            result=_compare_latest(collected),
        )
    return collected


async def _finish_workers(workers: dict, journal: Path) -> dict:
    completion = {}
    for side, worker in workers.items():
        completion[side] = await worker.request(operation="finish")
        _record(journal, operation="finish", side=side, response=completion[side])
        _validate_completion(completion[side])
    return completion


async def measure_round(
    case: dict,
    pair: ReleasePair,
    root: Path,
    count: int,
    *,
    fallback_declaration: dict | None = None,
) -> dict:
    sites, releases = pair.sites, pair.releases
    root.mkdir(parents=True, exist_ok=True)
    journal = root / "raw.jsonl"
    with journal.open("x", encoding="utf-8"):
        pass
    _record(journal, operation="begin", case=case, pairs=count)
    workers = {}
    validation = SampleValidation(
        case,
        releases["baseline"]["native_sha256"] != releases["candidate"]["native_sha256"],
        fallback_declaration,
    )
    try:
        if "diagnostic_variant" in case and count > 0 and fallback_declaration is None:
            raise ValueError("SQL fallback samples require a frozen declaration")
        for side, site in sites.items():
            workers[side] = await Worker.start(site, root / side)
        identities, warmups = await _prepare_workers(workers, pair, validation, journal)
        _validate_environments(identities, fallback_declaration)
        frozen = _freeze_fallback(case, warmups, identities["candidate"])
        _record(
            journal, operation="warmup-equivalence", result=_compare_latest(warmups)
        )
        collected = await _sample_pairs(workers, validation, count, journal)
        completion = await _finish_workers(workers, journal)
        return {
            "environment": identities["candidate"],
            "samples": collected,
            "completion": completion,
            **frozen,
        }
    except BaseException as error:
        _record(journal, operation="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for worker in workers.values():
            await worker.close()


def _quantiles(values: list[float]) -> dict:
    return dict(
        zip(
            ("p50", "p95", "p99"),
            np.quantile(values, (0.5, 0.95, 0.99), method="linear").tolist(),
            strict=True,
        )
    )


def _round_observations(evidence: list[dict], sides: dict) -> dict:
    return {
        side: [
            [sample["seconds"] for sample in item["samples"][side]] for item in evidence
        ]
        for side in sides
    }


def _latency_quantiles(observations: dict) -> dict:
    return {
        side: _quantiles([value for values in rounds for value in values])
        for side, rounds in observations.items()
    }


def _first_sink_seconds(sample: dict) -> float:
    return sum(
        sample["phases_seconds"][phase]
        for phase in (
            "enqueue_to_source_data",
            "source_data_to_source_watermark",
            "source_watermark_to_sink",
        )
    )


def _sink_quantiles(evidence: list[dict], sides: dict) -> dict:
    return {
        side: _quantiles(
            [
                _first_sink_seconds(sample)
                for item in evidence
                for sample in item["samples"][side]
            ]
        )
        for side in sides
    }


def _tail_result(
    observations: dict, evidence: list[dict], count: int, rounds: int
) -> dict:
    return {
        "verdict": "descriptive-tail-evidence",
        "pairs_per_round": count,
        "fresh_worker_pairs": rounds,
        "quantile_method": "linear interpolation, NumPy method=linear",
        "intervals": [
            paired_round(a, b)
            for a, b in zip(
                observations["baseline"], observations["candidate"], strict=True
            )
        ],
        "latency_seconds": _latency_quantiles(observations),
        "first_sink_seconds": _sink_quantiles(evidence, observations),
    }


async def measure_case(
    case: dict,
    pair: ReleasePair,
    root: Path,
    *,
    tail: bool,
    fallback_declaration: dict | None = None,
) -> dict:
    sites = pair.sites
    rounds, count = (3, 1_000) if tail else (2, 10)
    evidence = []
    for index in range(rounds):
        round_root = root / f"round-{index}"
        result = await measure_round(
            case, pair, round_root, count, fallback_declaration=fallback_declaration
        )
        (round_root / "round.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        evidence.append(result)
    if any(item["environment"] != evidence[0]["environment"] for item in evidence):
        raise ValueError("confirmation-round environments differ")
    observations = _round_observations(evidence, sites)
    row = {
        **case,
        "status": "ok",
        "correctness": True,
        "comparison": "interleaved",
        **observations,
        "evidence": evidence,
        "sample_count_per_revision": rounds * count,
        "quantile_method": "linear interpolation, NumPy method=linear",
        "latency_seconds": _latency_quantiles(observations),
    }
    row["result"] = (
        _tail_result(observations, evidence, count, rounds) if tail else comparison(row)
    )
    return row


def _diagnostic_hashes() -> dict:
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in (Path(__file__), ROOT / "benchmarks/performance_diagnostics.py")
    }


def _validate_declaration_identity(declaration: dict, report: dict) -> None:
    expected = {
        "contract": "calc-flow-sql-fallback-preflight-v2",
        "status": "complete",
        "inventory": [[group, case] for group, case in report["inventory"]],
        **{
            key: report[key]
            for key in (
                "releases",
                "common_harness_sha256",
                "diagnostic_harness_sha256",
            )
        },
    }
    if any(declaration.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "SQL fallback declaration has different identity, cases or instrument"
        )


def _declaration_rows(declaration: dict, report: dict) -> list[dict]:
    rows = declaration.get("cases", [])
    expected_ids = [case["id"] for _, case in report["inventory"]]
    if [row.get("id") for row in rows] != expected_ids or any(
        row.get("status") != "ok" or "fallback_preflight" not in row for row in rows
    ):
        raise ValueError("SQL fallback declaration is incomplete")
    return rows


def _validate_frozen_path(case: dict, structured: dict, side: str) -> None:
    if not isinstance(structured, dict):
        raise ValueError("missing structured path")
    nodes = structured["window_nodes"]
    if (
        not isinstance(nodes, list)
        or not nodes
        or any(node not in ("BoundedWindowAggExec", "WindowAggExec") for node in nodes)
    ):
        raise ValueError("missing DataFusion window nodes")
    # Validate frozen fields, without treating a reconstructed string as observed
    # evidence. Actual plans still come from each sample.
    _validate_fallback_path(
        case, {**structured, "physical_plan": "\n".join(nodes)}, side
    )


def _validate_frozen_case(case: dict, frozen: dict) -> None:
    if not isinstance(frozen, dict) or any(
        not isinstance(frozen.get(key), kind) or not frozen[key]
        for key, kind in (
            ("query", str),
            ("fixture", dict),
            ("environment", dict),
            ("paths", dict),
        )
    ):
        raise ValueError("missing case facts")
    entities = min(64, max(1, case["rows"] // 40))
    _fallback_fixture(
        case,
        {
            **frozen,
            "full_window_validation": {
                "passed": True,
                "full_window_rows": case["rows"] - 19 * entities,
                "null_rows": 19 * entities,
            },
        },
    )
    if set(frozen["paths"]) != {"baseline", "candidate"}:
        raise ValueError("missing revision path")
    for side, structured in frozen["paths"].items():
        _validate_frozen_path(case, structured, side)


def _load_fallback_declarations(path: Path, report: dict) -> dict[str, dict]:
    declaration = json.loads(path.read_text(encoding="utf-8"))
    _validate_declaration_identity(declaration, report)
    rows = _declaration_rows(declaration, report)
    for row, (_, case) in zip(rows, report["inventory"], strict=True):
        try:
            _validate_frozen_case(case, row["fallback_preflight"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "SQL fallback declaration has invalid case facts"
            ) from error
    return {row["id"]: row["fallback_preflight"] for row in rows}


def _selected_groups(group: str) -> dict:
    selected = inventory()
    if group == "fallback-cost":
        return {group: fallback_inventory()}
    if group in ("entity-parallel", "entity-parallel-tail"):
        return {group: entity_parallel_inventory()}
    if group == "all":
        return selected
    return {group: selected[group]}


def _selected_cases(args) -> list[tuple[str, dict]]:
    cases = [
        (group, case)
        for group, group_cases in _selected_groups(args.group).items()
        for case in group_cases
    ]
    if not args.case:
        return cases
    wanted = set(args.case)
    if wanted - {case["id"] for _, case in cases}:
        raise ValueError("requested case is absent from the selected inventory")
    return [(group, case) for group, case in cases if case["id"] in wanted]


def _validate_fallback_mode(
    cases: list[tuple[str, dict]],
    fallback: bool,
    preflight: bool,
    declaration_path: Path | None,
) -> None:
    if not fallback:
        if preflight or declaration_path is not None:
            raise ValueError(
                "preflight/declaration options require the fallback-cost group"
            )
        return
    if [case for _, case in cases] != fallback_inventory():
        raise ValueError("SQL fallback requires all 12 predeclared cases")
    if bool(preflight) == (declaration_path is not None):
        raise ValueError("SQL fallback requires preflight or its frozen declaration")


def _load_builds(args) -> tuple[dict, dict]:
    if args.baseline_build is None or args.candidate_build is None or args.root is None:
        raise ValueError(
            "both release build manifests and an evidence root are required"
        )
    releases, sites = {}, {}
    for side, path in {
        "baseline": args.baseline_build,
        "candidate": args.candidate_build,
    }.items():
        releases[side], sites[side] = load_release(path)
    for key in ("cargo_lock_sha256", "rustc_verbose", "features", "profile"):
        if releases["baseline"][key] != releases["candidate"][key]:
            raise ValueError(f"revision build configurations differ: {key}")
    return releases, sites


def _new_report(
    root: Path, cases: list, releases: dict, preflight: bool
) -> tuple[dict, Path]:
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / ("preflight.json" if preflight else "results.json")
    if report_path.exists():
        raise ValueError("use a new evidence directory; prior results are immutable")
    report = {
        "contract": "calc-flow-sql-fallback-preflight-v2"
        if preflight
        else "calc-flow-performance-plan-paired-v1",
        "comparison_kind": (
            "harness-self-check"
            if releases["baseline"]["native_sha256"]
            == releases["candidate"]["native_sha256"]
            else "source-revision-comparison"
        ),
        "releases": releases,
        "common_harness_sha256": harness_sha256(),
        "diagnostic_harness_sha256": _diagnostic_hashes(),
        "inventory": cases,
        "method": (
            "prepare/warmup only; no measured pairs or performance verdict"
            if preflight
            else "sequential AB/BA; every materialized Arrow output checked directly "
            "outside timing; no samples removed"
        ),
        "cases": [],
    }
    return report, report_path


def _prepare_declarations(
    report: dict, fallback: bool, preflight: bool, declaration_path: Path | None
) -> tuple[dict, dict]:
    if preflight:
        return {}, {"status": "incomplete"}
    if not fallback:
        return {}, {}
    declarations = _load_fallback_declarations(declaration_path, report)
    return declarations, {
        "fallback_declaration": {
            "path": str(declaration_path),
            "sha256": _sha256(declaration_path),
        }
    }


def _save_report(report_path: Path, report: dict) -> None:
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


async def _measure_selected_case(
    selected: tuple[str, dict],
    pair: ReleasePair,
    root: Path,
    preflight: bool,
    declaration: dict | None,
) -> dict:
    group, case = selected
    try:
        if preflight:
            result = await measure_round(case, pair, root, 0)
            return {
                **case,
                "status": "ok",
                "fallback_preflight": result["fallback_preflight"],
            }
        return await measure_case(
            case,
            pair,
            root,
            tail=group in ("tail", "entity-parallel-tail"),
            fallback_declaration=declaration,
        )
    except Exception as error:
        return {
            **case,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "evidence_directory": str(root),
        }


async def run(args) -> None:
    cases = _selected_cases(args)
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    fallback = args.group == "fallback-cost"
    preflight = getattr(args, "preflight", False)
    declaration_path = getattr(args, "fallback_declaration", None)
    _validate_fallback_mode(cases, fallback, preflight, declaration_path)
    releases, sites = _load_builds(args)
    pair = ReleasePair(sites, releases)
    report, report_path = _new_report(args.root, cases, releases, preflight)
    declarations, updates = _prepare_declarations(
        report, fallback, preflight, declaration_path
    )
    report = {**report, **updates}
    _save_report(report_path, report)
    for index, (group, case) in enumerate(cases):
        print(
            f"{'Preflight' if preflight else 'Measuring'} {group}: {case['id']}",
            flush=True,
        )
        row = await _measure_selected_case(
            (group, case),
            pair,
            args.root / f"case-{index}",
            preflight,
            declarations.get(case["id"]),
        )
        report["cases"].append({"group": group, **row})
        _save_report(report_path, report)
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "status": row["status"],
                    "result": row.get("result"),
                    "error": row.get("error"),
                }
            ),
            flush=True,
        )
    if any(case["status"] == "error" for case in report["cases"]):
        raise RuntimeError(
            "performance evidence is incomplete; see recorded case errors"
        )
    if preflight:
        report["status"] = "complete"
        _save_report(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", type=Path)
    parser.add_argument("--candidate-build", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--group",
        choices=(
            "core",
            "sensitivity",
            "tail",
            "all",
            "fallback-cost",
            "entity-parallel",
            "entity-parallel-tail",
        ),
        default="core",
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--fallback-declaration", type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
