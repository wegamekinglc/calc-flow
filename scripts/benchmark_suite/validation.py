"""Validate release identities and original per-sample evidence before reporting."""

from __future__ import annotations

import re

from scripts.benchmark_suite.catalog import CONTRACT, THREADS, get_shard, shard_cases
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.report import SAMPLES, comparison, validate_shards


def validate_shape(report: dict) -> dict:
    if not isinstance(report, dict):
        raise ValueError("benchmark artifact must be an object")
    shard = get_shard(report["shard"]["id"])
    if not isinstance(report["cases"], list) or not isinstance(report["errors"], list):
        raise ValueError("cases and errors must be lists")
    if any(not isinstance(error, str) for error in report["errors"]):
        raise ValueError("errors must be strings")
    for case in report["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("each result must have a string case id")
        if not isinstance(case.get("scope"), str) or case.get("status") not in (
            "ok",
            "error",
        ):
            raise ValueError("each result must declare its status and timing scope")
        if case.get("family") != shard["family"]:
            raise ValueError("case family differs from its shard")
    return shard


def validate_releases(releases: dict, base: str, head: str) -> None:
    for side, expected in (("baseline", base), ("candidate", head)):
        release = releases[side]
        if (
            not re.fullmatch("[0-9a-f]{40}", expected)
            or release["git_sha"] != expected
            or release["git_clean"] is not True
            or release["build_profile"] != "release"
            or not re.fullmatch("[0-9a-f]{64}", release["native_sha256"])
        ):
            raise ValueError(f"{side}: not a clean release build at the expected SHA")


def _validate_round(case: dict, evidence: dict, releases: dict) -> None:
    sides = (
        {"candidate"} if case["comparison"] == "external" else {"baseline", "candidate"}
    )
    if set(evidence["samples"]) != sides:
        raise ValueError("measured worker roles differ from the comparison contract")
    if set(evidence["completion"]) != sides or any(
        evidence["completion"][side]["state"] != "completed" for side in sides
    ):
        raise ValueError("worker completion was not established")
    if evidence["native_sha256"] != {
        side: releases[side]["native_sha256"] for side in sides
    }:
        raise ValueError("raw worker native hashes differ from release manifests")
    for samples in evidence["samples"].values():
        if len(samples) != SAMPLES:
            raise ValueError("confirmation round has an incomplete sample inventory")
        if any(sample["correctness"]["passed"] is not True for sample in samples):
            raise ValueError("a measured output failed correctness")
    if case["family"] == "warm":
        _validate_cursors(case, evidence)


def _validate_cursors(case: dict, evidence: dict) -> None:
    expected = [
        case["history_rows"] + case["rows"] * (index + 1) for index in range(SAMPLES)
    ]
    for samples in evidence["samples"].values():
        if [sample["start_row"] for sample in samples] != expected:
            raise ValueError("warm base/head sample cursors differ")


def _validate_evidence(case: dict, releases: dict) -> None:
    rounds = case["evidence"]
    if len(rounds) != 2 or rounds[0]["environment"] != rounds[1]["environment"]:
        raise ValueError("missing confirmation or changed environment")
    environment = rounds[0]["environment"]
    if environment["polars_threads"] != THREADS or environment[
        "tokio_worker_threads"
    ] != str(THREADS):
        raise ValueError("unexpected worker thread configuration")
    for side in ("baseline", "candidate"):
        actual = [
            [sample["seconds"] for sample in evidence["samples"].get(side, [])]
            for evidence in rounds
        ]
        if side == "baseline" and case["comparison"] == "external":
            actual = []
        if actual != case[side]:
            raise ValueError("summary samples differ from raw worker evidence")
    for evidence in rounds:
        _validate_round(case, evidence, releases)
    comparison(case)


def _validate_catalog_cases(report: dict, shard: dict) -> None:
    expected = {case["id"]: case for case in shard_cases(shard)}
    for case in report["cases"]:
        if any(case.get(key) != value for key, value in expected[case["id"]].items()):
            raise ValueError(f"{case['id']}: workload dimensions changed")
        if case["status"] != "ok":
            continue
        kind = "interleaved" if case["backend"].startswith("calc-flow") else "external"
        if case["comparison"] != kind:
            raise ValueError("comparison kind differs from the backend contract")
        _validate_evidence(case, report["releases"])


def validate_fragment(report: dict, base: str, head: str) -> None:
    shard = validate_shape(report)
    if report["contract"] != CONTRACT or shard != report["shard"]:
        raise ValueError("report contract or shard differs from the current catalog")
    validate_releases(report["releases"], base, head)
    if report["harness_sha256"] != harness_sha256():
        raise ValueError(
            "measurement harness or dependency lock differs from the candidate"
        )
    expected = (
        [case["id"] for case in shard_cases(shard)]
        if shard["family"] in ("engines", "warm")
        else report["expected_case_ids"]
    )
    if not expected:
        raise ValueError("benchmark inventory is empty")
    validate_shards(expected, [case["id"] for case in report["cases"]])
    if shard["family"] in ("engines", "warm"):
        _validate_catalog_cases(report, shard)
