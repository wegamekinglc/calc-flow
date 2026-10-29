"""Arrow-only fixtures, immutable scenario identities and exact oracles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

REF = "b5ae227e48ac9bb8a81583a23485e2c6064cee52"
NATIVE = "a0708fcf5fa84a1cf2c5347a66b73680d7426a3313746a1ef99519d5b2c7bfb8"
ROWS = 32768
BASE = 1767225600000000


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    batch_rows: int
    window: bool = False

    @property
    def arms(self):
        return ("streaming",) if self.window else ("streaming", "sql", "sql_plan")


CASES = (
    Case("transform-full", ROWS),
    Case("transform-512", 512),
    Case("window-full", ROWS, True),
    Case("window-512", 512, True),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def fixture(case):
    ids = np.arange(ROWS, dtype=np.int64)
    table = pa.table(
        {
            "id": ids,
            "k": ids % 16,
            "ts": pa.array(BASE + ids // 512 * 2000000, pa.timestamp("us", tz="UTC")),
            "x": ids * 17 % 101 - 50,
        }
    )
    batches = table.to_batches(max_chunksize=case.batch_rows)
    if case.window:
        marker = pa.table(
            {
                "id": np.full(512, -1, np.int64),
                "k": np.full(512, -1, np.int64),
                "ts": pa.array(np.full(512, BASE + 130000000), table["ts"].type),
                "x": np.zeros(512, np.int64),
            }
        )
        batches.extend(marker.to_batches())
    return table, batches


def encode(batches):
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_stream(buffer, batches[0].schema) as writer:
        for batch in batches:
            writer.write_batch(batch)
    return buffer.getvalue().to_pybytes()


def oracle(case):
    ids = np.arange(ROWS, dtype=np.int64)
    if not case.window:
        selected = ids[ids % 16 < 8]
        return pa.table({"id": selected, "y": (selected * 17 % 101 - 50) * 2 + 1})
    return pa.table(
        {
            "ws": np.repeat(BASE + np.arange(32, dtype=np.int64) * 4000000, 16),
            "k": np.tile(np.arange(16, dtype=np.int64), 32),
            "total": (ids * 17 % 101 - 50).reshape(32, 64, 16).sum(axis=1).ravel(),
            "cnt": np.full(512, 64, np.int64),
        }
    )


def validate(case, table):
    expected = oracle(case)
    if case.window:
        actual = pa.table(
            {
                "ws": pc.cast(table["window_start"], pa.int64()),
                "k": table["k"],
                "total": table["total"],
                "cnt": table["cnt"],
            }
        ).sort_by([("ws", "ascending"), ("k", "ascending")])
    else:
        actual = table.select(["id", "y"]).sort_by("id")
    if actual.num_rows != expected.num_rows or any(
        not actual[name].cast(expected[name].type).equals(expected[name])
        for name in expected.column_names
    ):
        raise ValueError("Arrow result differs from every-row integer oracle")


def declaration(case, arm):
    import calc_flow as cf

    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("k", pa.int64()),
            ("ts", pa.timestamp("us", tz="UTC")),
            ("x", pa.int64()),
        ]
    )
    source = cf.table_input("events", schema=schema)
    accepted = source.filter(source["id"] >= 0)
    if case.window:
        output = cf.window.tumbling(
            accepted,
            event_time="ts",
            size_micros=4000000,
            group_by=("k",),
            aggregates=(
                cf.window.sum("x", output="total"),
                cf.window.count("id", output="cnt"),
            ),
        )
    else:
        selected = accepted.filter(accepted["k"] < 8)
        output = selected.select("id", y=selected["x"] * 2 + 1)
    return cf.Program(
        "dal417-arrow-ipc",
        engine="streaming" if arm == "streaming" else "sql",
        outputs={"result": output},
    )
