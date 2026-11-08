"""Fixed C1 invocation contract, distinct from scheduled twenty-sample reports."""

from __future__ import annotations

import hashlib
import math
import struct
from functools import lru_cache
from pathlib import Path

from scripts.benchmark_suite.normalize import read_json
from scripts.toolkit import fingerprint_json

SCHEMA = "calc-flow.sql-snapshot-cache-invocation.v1"
SCOPE = "warm-native-sql-operator-append-full-snapshots"
QUERY = "SELECT account, stock, SUM(value) AS total FROM events GROUP BY account, stock"
MODES = ("sparse", "dense")
DEFAULT_METADATA = {"source": "", "sequence": 0, "attributes": {}}
THREAD_ENVIRONMENT = {
    "TOKIO_WORKER_THREADS": "4",
    "POLARS_MAX_THREADS": "4",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}
PHASES = (
    "warm_seconds",
    "process_seconds",
    "validation_seconds",
    "checkpoint_seconds",
    "recovery_seconds",
    "cleanup_seconds",
)


def slots(mode: str) -> tuple[int, ...]:
    if mode not in MODES:
        raise ValueError("invalid snapshot-cache selection")
    return (
        tuple(range(9000, 10000))
        if mode == "sparse"
        else tuple(sorted((i * 37) % 10000 for i in range(1000)))
    )


def row_digest(indices: tuple[int, ...], values: tuple[int, ...]) -> str:
    digest = hashlib.sha256(b"calc-flow.c1-rows.v1\0")
    digest.update(struct.pack("<Q", len(indices)))
    for slot, value in zip(indices, values, strict=True):
        account = None if slot == 0 else f"account-{slot // 100:03}"
        stock = f"stock-{slot % 100:03}"
        for text in (account, stock):
            digest.update(bytes([text is not None]))
            if text is not None:
                encoded = text.encode()
                digest.update(struct.pack("<Q", len(encoded)))
                digest.update(encoded)
        digest.update(b"\x01" + struct.pack("<q", value))
    return digest.hexdigest()


@lru_cache(maxsize=2)
def expected_digests(mode: str) -> tuple:
    selected = slots(mode)
    all_slots = tuple(range(10000))
    values = [1] * 10000
    warm = row_digest(all_slots, tuple(values))
    inputs, outputs = [], []
    for index in range(20):
        increment = index + 1
        inputs.append(row_digest(selected, (increment,) * 1000))
        for slot in selected:
            values[slot] += increment
        outputs.append(row_digest(all_slots, tuple(values)))
    recovery_first = outputs[-1]
    values[0] += 1
    recovery_second = row_digest(all_slots, tuple(values))
    return warm, tuple(inputs), tuple(outputs), (recovery_first, recovery_second)


def counts() -> dict:
    return {
        "warm": {"input_rows": 10000, "output_rows": 10000, "snapshots": 1},
        "append": {"input_rows": 20000, "output_rows": 200000, "snapshots": 20},
        "recovery": {"input_rows": 1, "output_rows": 20000, "snapshots": 2},
        "total": {"input_rows": 30001, "output_rows": 230000, "snapshots": 23},
        "held_warm_validation_rows": 200000,
    }


def selected_snapshot_row(path: Path) -> dict:
    report = read_json(path)
    mode = report.get("selection")
    validate_contract(report, mode)
    validate_counts(report.get("counts"))
    require(
        report.get("metadata") == DEFAULT_METADATA
        and type(report["metadata"]["sequence"]) is int,
        "batch metadata changed",
    )
    validate_oracles(report, mode)
    validate_timing(report)
    validate_checkpoint(report["checkpoint"])
    validate_resources(report)
    return {
        "samples": [report["process_seconds"]],
        "rows": counts()["append"]["input_rows"],
        "scope": SCOPE,
        "metadata": invocation_metadata(report),
    }


def invocation_metadata(report: dict) -> dict:
    workload = {
        "scenario": f"sql_snapshot_cache/{report['selection']}",
        "scope": SCOPE,
        "backend": "rust-native-sql-operator",
        "scale": "fixed",
        "input_rows": 20000,
        "output_rows": 200000,
        "groups": 10000,
        "appends": 20,
        "append_rows": 1000,
        "query": QUERY,
        "counts_all_phases": counts(),
        "metadata": DEFAULT_METADATA,
        "row_digest_format": "calc-flow.c1-rows.v1",
        "checkpoint_sha256": report["checkpoint"]["sha256"],
    }
    return {
        "input_rows": 20000,
        "output_rows": 200000,
        "scope": SCOPE,
        "workload_identity": workload,
        "workload_fingerprint": fingerprint_json(workload),
        "selection": report["selection"],
        "report": report,
    }


def validate_contract(report: dict, mode: str) -> None:
    require(
        report.get("schema") == SCHEMA
        and report.get("scope") == SCOPE
        and report.get("query") == QUERY
        and mode in MODES,
        "invalid selected SQL invocation contract",
    )
    require(
        "samples" not in report and "cases" not in report, "one invocation required"
    )


def validate_counts(observed: object) -> None:
    require(observed == counts(), "invalid all-phase row/count descriptor")
    for value in observed.values():
        values = value.values() if isinstance(value, dict) else [value]
        require(
            all(type(count) is int for count in values),
            "row/count fields must be integers",
        )


def validate_resources(report: dict) -> None:
    require(report.get("cleanup") is True, "owned native cleanup incomplete")
    require(
        report.get("engine_limit_basis") == "default-runtime-known-infinite",
        "wrong engine eligibility",
    )
    require(
        report.get("engine_reserved") == {"status": "unmeasured"},
        "unsupported reservation claim",
    )
    resources = report.get("resources", {})
    require(
        resources.get("schema") == "calc-flow.phase-resources.v2"
        and resources.get("status") == "measured"
        and resources.get("scope") == "c1-invocation-before-json",
        "native process resources unavailable",
    )
    require(finite(resources.get("process_cpu_seconds")), "invalid process CPU")
    resolution = resources.get("process_cpu_resolution_seconds")
    require(finite(resolution) and resolution > 0, "invalid CPU clock resolution")
    require(
        report.get("observed_cpu_affinity") == [0, 2], "fixed native affinity changed"
    )
    require(
        report.get("observed_thread_environment") == THREAD_ENVIRONMENT,
        "native thread environment changed",
    )
    pid = report.get("process_id")
    require(type(pid) is int and pid > 0, "native process identity missing")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_timing(report: dict) -> None:
    require(
        all(finite(report.get(name)) for name in (*PHASES, "total_seconds")),
        "invalid phase timing",
    )
    appends = [sample["seconds"] for sample in report["appends"]]
    require(
        all(finite(value) and value > 0 for value in appends), "invalid append timing"
    )
    require(
        report["process_seconds"] > 0
        and math.isclose(
            sum(appends), report["process_seconds"], rel_tol=1e-12, abs_tol=1e-12
        ),
        "primary must sum twenty complete append observations",
    )
    require(
        sum(report[name] for name in PHASES) <= report["total_seconds"] * 1.01,
        "overlapping phase timings",
    )


def validate_oracles(report: dict, mode: str) -> None:
    warm, inputs, outputs, recovery = expected_digests(mode)
    require(
        report.get("warm")
        == {"input_sha256": warm, "sha256": warm, "chunks": [8192, 1808]},
        "wrong warm snapshot",
    )
    appends = report.get("appends")
    require(
        isinstance(appends, list) and len(appends) == 20,
        "twenty business appends required",
    )
    for index, (sample, incoming, output) in enumerate(
        zip(appends, inputs, outputs, strict=True)
    ):
        require(
            sample.get("index") == index and type(sample.get("index")) is int,
            "append sequence differs",
        )
        require(
            sample.get("input_sha256") == incoming
            and sample.get("output_sha256") == output,
            "full typed row oracle differs",
        )
        require(sample.get("held_warm_sha256") == warm, "held warm snapshot changed")
    require(
        report.get("recovery")
        == {
            "input_sha256": [row_digest((), ()), row_digest((0,), (1,))],
            "snapshot_sha256": list(recovery),
            "validated": True,
        }
        and report["recovery"]["validated"] is True,
        "cold restore/continuation differs",
    )


def validate_checkpoint(state: dict) -> None:
    expected = {"layout": 3, "groups": 10000, "columns": 3, "logical_rows": 30000}
    require(
        all(
            type(state.get(key)) is int and state[key] == value
            for key, value in expected.items()
        ),
        "checkpoint state descriptor differs",
    )
    require(
        type(state.get("logical_bytes")) is int and state["logical_bytes"] > 0,
        "invalid logical bytes",
    )
    digest = state.get("sha256")
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        "invalid checkpoint digest",
    )
