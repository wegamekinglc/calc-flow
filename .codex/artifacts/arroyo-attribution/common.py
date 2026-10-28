"""Fixed workloads and immutable identities for the DAL-417 follow-up."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CF_REF = "b9a10d4fa4b109b810d2f5839951912b92e597f8"
ARROYO_REF = "4ef15f6336a596651f10cf27c1ee29c29bf81577"
ARROYO_SHA = "558b28b4c2aa3153500308684fd121dea2f040b5599d9f3342830834c24a319f"
ROWS = 32768
HISTORY = ROOT / ".codex/artifacts/arroyo-comparison/measure.py"
SPEC = importlib.util.spec_from_file_location("dal417_legacy", HISTORY)
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    batch_rows: int
    transport: str
    window: bool = False


@dataclass(frozen=True, slots=True)
class Transport:
    endpoint: str
    profile: bool


CASES = (
    Case("live-512", 512, "live"),
    Case("live-8192", 8192, "live"),
    Case("arrow-full", ROWS, "arrow"),
    Case("arrow-512", 512, "arrow"),
    Case("window-live-512", 512, "live", True),
)


def arms(case: Case) -> tuple[str, ...]:
    if case.window:
        return "streaming", "arroyo"
    return "streaming", "sql", "arroyo" if case.transport == "live" else "sql_plan"


def case_named(name: str) -> Case:
    return next(case for case in CASES if case.name == name)


def prepared(case: Case) -> tuple[list[bytes], list[dict], list[dict]]:
    frames, records = legacy.fixture(ROWS)
    stride = case.batch_rows // 512
    grouped = [b"".join(frames[i : i + stride]) for i in range(0, 64, stride)]
    return (
        [*grouped, frames[-1]],
        records,
        legacy.oracle("window" if case.window else "stateless", records),
    )


def schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.int64()),
            ("k", pa.int64()),
            ("ts", pa.timestamp("us", tz="UTC")),
            ("x", pa.int64()),
        ]
    )


def declaration(engine: str, window: bool = False):
    import calc_flow as cf

    source = cf.table_input("events", schema=schema())
    accepted = source.filter(source["id"] >= 0)
    if window:
        output = cf.window.tumbling(
            accepted,
            event_time="ts",
            size_micros=4_000_000,
            group_by=("k",),
            aggregates=(
                cf.window.sum("x", output="total"),
                cf.window.count("id", output="cnt"),
            ),
        )
    else:
        selected = accepted.filter(accepted["k"] < 8)
        output = selected.select("id", y=selected["x"] * 2 + 1)
    return cf.Program("dal417-attribution", engine=engine, outputs={"result": output})


def normalize(table, window: bool):
    import pyarrow as pa
    import pyarrow.compute as pc

    if not window:
        return table
    return pa.table(
        {
            "ws": pc.cast(table["window_start"], pa.int64()),
            "k": table["k"],
            "total": table["total"],
            "cnt": table["cnt"],
        }
    )


def check(actual: list[dict], expected: list[dict]) -> None:
    def key(row):
        return tuple(sorted(row.items()))

    if sorted(actual, key=key) != sorted(expected, key=key):
        raise ValueError(
            f"result mismatch: actual={len(actual)}, expected={len(expected)}"
        )


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def environment() -> dict[str, str]:
    return {
        "TOKIO_WORKER_THREADS": "4",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "RUST_LOG": "warn",
        "ARROYO__DISABLE_TELEMETRY": "true",
        "ARROYO__DEFAULT_CHECKPOINT_INTERVAL": "1h",
        "ARROYO__PIPELINE__SOURCE_BATCH_LINGER": "100ms",
        "ARROYO__PIPELINE__ALLOWED_RESTARTS": "0",
        "ARROYO__ADMIN__HTTP_PORT": "0",
        **{
            f"ARROYO__{name}__BIND_ADDRESS": "127.0.0.1"
            for name in ("ADMIN", "API", "CONTROLLER", "WORKER")
        },
    }
