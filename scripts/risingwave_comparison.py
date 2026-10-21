"""DAL-416: bounded single-node application-boundary comparison, not a gate.

Run as ``python -m scripts.risingwave_comparison --help`` from the repo root.
Uses the maintained warm-stream source/sink and validates every result against
an integer oracle. Owns and settles its RisingWave child before returning.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import sys
import time
from contextlib import redirect_stdout, suppress
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import psutil
import psycopg
import pyarrow as pa
from psycopg import sql

import calc_flow as cf
from benchmarks.warm_stream import _InteractiveSource, _QueueSink
from calc_flow import window
from scripts.toolkit import sha256_file, wheel_native_sha256

BASE = datetime(2026, 1, 1, tzinfo=UTC)
BASE_US = int(BASE.timestamp()) * 1_000_000
CF_COMMIT = "dab98a559314bd46d866ed7a6489abcdd3810afe"
RW_COMMIT = "45baf6460ea942e4ff57a5aa057d5b8db7f3b563"
SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("k", pa.int64()),
        ("v", pa.int64()),
        ("ts", pa.timestamp("us", tz="UTC")),
    ]
)
CASES = (("projection", 640), ("grouped", 640), ("grouped", 64_000), ("window", 640))
ROUNDS, WARMUP, SAMPLES = 2, 2, 10
QUERY = "SELECT k, SUM(v) AS total, COUNT(v) AS n FROM input GROUP BY k"
_CASE_NAMES = tuple(
    f"bench_{kind}_{rows}_{rid}" for rid in range(ROUNDS) for kind, rows in CASES
) + ("window_semantics",)
# These unquoted tokens are built exclusively from the fixed case catalog.
# Lookup rejects arbitrary identifiers while preserving the measured SQL bytes.
_RELATIONS = {
    name: sql.SQL(name) for base in _CASE_NAMES for name in (base, base + "_mv")
}
_VIEW_QUERIES = {
    "projection": sql.SQL("SELECT id, v * 2 + 1 AS value FROM {} WHERE v >= 0"),
    "grouped": sql.SQL("SELECT k, SUM(v) AS total, COUNT(v) AS n FROM {} GROUP BY k"),
    "window": sql.SQL(
        "SELECT window_start, window_end, k, SUM(v) AS total, COUNT(v) AS n "
        "FROM TUMBLE({}, ts, INTERVAL '10 seconds') WHERE k >= 0 "
        "GROUP BY window_start, window_end, k EMIT ON WINDOW CLOSE"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def fixture(batch: int, rows: int, *, progress: bool) -> pa.Table:
    """Ten event-time seconds per request; key 63 is deliberately all NULL."""
    start = batch * rows
    ids = list(range(start, start + rows))
    keys = [i % 64 for i in ids]
    values = [
        None if i % 13 == 0 or k == 63 else i % 257 - 128
        for i, k in zip(ids, keys, strict=True)
    ]
    times = [BASE + timedelta(seconds=batch * 10 + i * 10 // rows) for i in range(rows)]
    if progress:
        ids.append(-(batch + 1))
        keys.append(0)
        values.append(None)
        times.append(BASE + timedelta(seconds=(batch + 1) * 10))
    return pa.table({"id": ids, "k": keys, "v": values, "ts": times}, schema=SCHEMA)


def _relation(name: str) -> sql.SQL:
    require(name in _RELATIONS, "unknown benchmark relation")
    return _RELATIONS[name]


def _integer_literal(value: int | None) -> sql.SQL:
    require(value is None or type(value) is int, "expected an integer or NULL")
    # Psycopg guards negative literals with a leading space. Each value here
    # has its own comma/parenthesis delimiter, so trim only that guard space
    # to retain the original wire text; no string payload enters this path.
    return sql.SQL(sql.Literal(value).as_string().lstrip())


def _insert_row(row: dict) -> sql.Composed:
    return sql.SQL("({},{},{},CAST({} AS TIMESTAMPTZ))").format(
        *(_integer_literal(row[field]) for field in ("id", "k", "v")),
        sql.Literal(row["ts"].isoformat()),
    )


def insert_sql(table: pa.Table, name: str) -> str:
    relation = _relation(name)
    require(table.schema == SCHEMA, "benchmark insert schema differs")
    values = sql.SQL(",").join(_insert_row(row) for row in table.to_pylist())
    return sql.SQL("INSERT INTO {} VALUES {}").format(relation, values).as_string()


def _select_view(name: str) -> sql.Composed:
    return sql.SQL("SELECT * FROM {}").format(_relation(name + "_mv"))


def _sample_selection(name: str, kind: str, index: int, rows: int) -> str:
    selection = _select_view(name)
    if kind == "projection":
        selection += sql.SQL(" WHERE id >= {} AND id < {}").format(
            _integer_literal(index * rows), _integer_literal((index + 1) * rows)
        )
    elif kind == "window":
        start = (BASE + timedelta(seconds=index * 10)).isoformat()
        selection += sql.SQL(" WHERE window_start = CAST({} AS TIMESTAMPTZ)").format(
            sql.Literal(start)
        )
    return selection.as_string()


def _oracle_rows(kind: str, tables: list[pa.Table]) -> list[dict]:
    columns = SCHEMA.names if kind == "window" else ["id", "k", "v"]
    return [
        r
        for t in tables
        for r in t.select(columns).to_pylist()
        if r["k"] >= 0 and r["id"] >= 0
    ]


def _project_rows(rows: list[dict]) -> list[list]:
    return [
        [row["id"], row["v"] * 2 + 1]
        for row in rows
        if row["v"] is not None and row["v"] >= 0
    ]


def _window_key(row: dict) -> tuple:
    micros = int((row["ts"] - BASE).total_seconds() * 1_000_000)
    start = BASE_US + micros // 10_000_000 * 10_000_000
    return start, start + 10_000_000, row["k"]


def _aggregate_result(key, aggregate: tuple) -> list:
    total, count = aggregate
    return [
        *(key if isinstance(key, tuple) else (key,)),
        total if count else None,
        count,
    ]


def _aggregate_rows(rows: list[dict], key_for) -> list[list]:
    groups = {}
    for row in rows:
        key = key_for(row)
        total, count = groups.get(key, (0, 0))
        value = row["v"]
        groups[key] = (total + (value or 0), count + (value is not None))
    return [
        _aggregate_result(key, aggregate) for key, aggregate in sorted(groups.items())
    ]


def expected(kind: str, tables: list[pa.Table]) -> list[list]:
    rows = _oracle_rows(kind, tables)
    if kind == "projection":
        return _project_rows(rows)
    key_for = _window_key if kind == "window" else lambda row: row["k"]
    return _aggregate_rows(rows, key_for)


def normalize(rows: list | tuple) -> list[list]:
    return sorted(
        [
            [
                int(v.timestamp()) * 1_000_000 + v.microsecond
                if isinstance(v, datetime)
                else int(v)
                if v is not None
                else None
                for v in row
            ]
            for row in rows
        ],
        key=lambda row: tuple((v is not None, v or 0) for v in row),
    )


def program(kind: str) -> cf.Program:
    source = cf.table_input("events", schema=SCHEMA)
    if kind == "window":
        output = window.tumbling(
            source.filter(source["k"] >= 0),
            event_time="ts",
            size_micros=10_000_000,
            group_by=["k"],
            aggregates=[window.sum("v", output="total"), window.count("v", output="n")],
        )
    else:
        query = (
            "SELECT id, v * 2 + 1 AS value FROM input WHERE v >= 0"
            if kind == "projection"
            else QUERY
        )
        output = source.sql(query)
    return cf.Program("dal416", engine="streaming", outputs={"result": output})


class CalcStream:
    """Own the existing runner/source/sink lifecycle for one advancing case."""

    def __init__(self, kind: str, rows: int, root: Path):
        self.kind = kind
        self.source = _InteractiveSource(max_batch_rows=rows + 1)
        self.sink = _QueueSink()
        self.plan = program(kind).compile_stream(cf.Runtime())
        self.runner = cf.StreamingRunner(
            self.plan,
            {
                self.plan.source_binding_ids[0]: cf.SourceBinding(
                    self.source, watermark_policy=cf.SourceProvidedWatermarks()
                )
            },
            {
                self.plan.sink_binding_ids[0]: [
                    cf.SinkBinding.ordinary("collector", self.sink)
                ]
            },
            None,
            config=cf.StreamRuntimeConfig(
                checkpointing=False,
                sql_state_budget=cf.StateBudget(10_000_000, 8 << 30),
                edge_budget=cf.EdgeBudget(max_rows=rows + 1, max_bytes=64 << 20),
            ),
        )
        self.job = None

    async def start(self):
        self.job = await self.runner.start_async()
        await asyncio.wait_for(self.source.opened.wait(), timeout=30)
        return self

    async def append(self, data: cf.Data, watermark: cf.Watermark | None):
        started = time.perf_counter()
        await self.source.push(data)
        if watermark is not None:
            await self.source.push(watermark)
        try:
            table = await self.sink.receive(timeout=30)
        except TimeoutError as error:
            raise RuntimeError(
                f"Calc Flow output timeout: {self.job.status()}"
            ) from error
        columns = (
            ["window_start", "window_end", "k", "total", "n"]
            if self.kind == "window"
            else table.column_names
        )
        actual = normalize(
            list(zip(*(table[c].to_pylist() for c in columns), strict=True))
        )
        elapsed = time.perf_counter() - started
        return actual, {"seconds": elapsed, "delivery_seconds": elapsed}

    async def close(self):
        if self.job is not None:
            await self.job.cancel_async()


class RisingWave:
    """Own one official binary and local persistent store, including restart."""

    def __init__(self, args, output: Path):
        self.args, self.output = args, output
        self.process = None
        self.connection = None
        self.log = None
        self.starts = 0

    async def start(self):
        self.starts += 1
        self.log = (self.output / f"risingwave-{self.starts}.log").open("wb")
        env = dict(
            os.environ,
            LD_LIBRARY_PATH=str(self.args.python312_lib),
            RUST_LOG="warn",
            RW_TELEMETRY_ENABLED="false",
        )
        self.process = await asyncio.create_subprocess_exec(
            str(self.args.risingwave),
            "single_node",
            "--config-path",
            str(self.output / "risingwave.toml"),
            "--store-directory",
            str(self.output / "rw-store"),
            "--listen-addr",
            "127.0.0.1:14566",
            "--parallelism",
            "4",
            "--total-memory-bytes",
            str(8 << 30),
            "--compaction-worker-threads-number",
            "1",
            stdout=self.log,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        for _ in range(300):
            require(self.process.returncode is None, "RisingWave exited; inspect log")
            try:
                self.connection = await psycopg.AsyncConnection.connect(
                    "host=127.0.0.1 port=14566 user=root dbname=dev connect_timeout=1",
                    autocommit=True,
                )
                await self.execute("SET TIME ZONE 'UTC'")
                await self.execute("SET statement_timeout = '30s'")
                await self.execute("FLUSH")
                return self
            except psycopg.Error:
                if self.connection is not None:
                    await self.connection.close()
                    self.connection = None
                await asyncio.sleep(0.1)
        raise TimeoutError("RisingWave SQL readiness exceeded 30 seconds")

    async def execute(self, sql: str):
        cursor = await self.connection.execute(sql, prepare=False)
        return await cursor.fetchall() if cursor.description else []

    async def stop(self, *, crash: bool = False):
        if self.connection is not None:
            await self.connection.close()
            self.connection = None
        if self.process is not None and self.process.returncode is None:
            if crash:
                self.process.kill()
            else:
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=20)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.log is not None:
            self.log.close()

    async def setup(self, name: str, kind: str):
        await self.execute(
            "SET streaming_parallelism = 1"
            if kind == "window"
            else "SET streaming_parallelism = 4"
        )
        watermark = sql.SQL(", WATERMARK FOR ts AS ts" if kind == "window" else "")
        await self.execute(
            sql.SQL(
                "CREATE TABLE {}(id BIGINT, k BIGINT, v BIGINT, ts TIMESTAMPTZ"
                "{}) APPEND ONLY"
            )
            .format(_relation(name), watermark)
            .as_string()
        )
        query = _VIEW_QUERIES[kind].format(_relation(name))
        await self.execute(
            sql.SQL("CREATE MATERIALIZED VIEW {} AS {}")
            .format(_relation(name + "_mv"), query)
            .as_string()
        )
        await self.execute("FLUSH")

    async def drop(self, name: str, *, if_exists: bool):
        optional = sql.SQL("IF EXISTS " if if_exists else "")
        await self.execute(
            sql.SQL("DROP MATERIALIZED VIEW {}{}")
            .format(optional, _relation(name + "_mv"))
            .as_string()
        )
        await self.execute(
            sql.SQL("DROP TABLE {}{}").format(optional, _relation(name)).as_string()
        )

    async def append(self, statement: str, selection: str):
        started = time.perf_counter()
        await self.execute(statement)
        inserted = time.perf_counter()
        await self.execute("FLUSH")
        flushed = time.perf_counter()
        rows = await self.execute(selection)
        actual = normalize(rows)
        finished = time.perf_counter()
        return actual, {
            "seconds": finished - started,
            "insert_seconds": inserted - started,
            "flush_seconds": flushed - inserted,
            "select_seconds": finished - flushed,
        }


def process_usage(pid: int) -> dict:
    process = psutil.Process(pid)
    cpu = process.cpu_times()
    io = process.io_counters()
    return {
        "cpu_seconds": cpu.user + cpu.system,
        "rss_bytes": process.memory_info().rss,
        "write_bytes": io.write_bytes,
        "read_bytes": io.read_bytes,
        "threads": process.num_threads(),
    }


async def observe_peak(pids: list[int], stop: asyncio.Event, record: dict):
    while not stop.is_set():
        values = [process_usage(pid) for pid in pids]
        record["rss_peak_bytes"] = max(
            record.get("rss_peak_bytes", 0), sum(v["rss_bytes"] for v in values)
        )
        record["engine_rss_peak_bytes"] = max(
            record.get("engine_rss_peak_bytes", 0), values[-1]["rss_bytes"]
        )
        record["threads_peak"] = max(
            record.get("threads_peak", 0), sum(v["threads"] for v in values)
        )
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=0.005)


async def measured_call(call, pids: list[int]):
    before = [process_usage(pid) for pid in pids]
    peak = {
        "rss_peak_bytes": sum(p["rss_bytes"] for p in before),
        "threads_peak": 0,
        "engine_rss_peak_bytes": before[-1]["rss_bytes"],
    }
    stop = asyncio.Event()
    monitor = asyncio.create_task(observe_peak(pids, stop, peak))
    try:
        actual, sample = await call()
    finally:
        stop.set()
        await monitor
    after = [process_usage(pid) for pid in pids]
    return actual, {
        **sample,
        **peak,
        **{
            field: sum(a[field] - b[field] for a, b in zip(after, before, strict=True))
            for field in ("cpu_seconds", "write_bytes", "read_bytes")
        },
        "process_usage_before": before,
        "process_usage_after": after,
        "rss_after_bytes": sum(p["rss_bytes"] for p in after),
    }


def _prepared_tables(kind: str, rows: int) -> list[pa.Table]:
    return [
        fixture(i, rows, progress=kind == "window") for i in range(WARMUP + SAMPLES)
    ]


def _case_inputs(name: str, kind: str, rows: int) -> tuple:
    prepared = _prepared_tables(kind, rows)
    statements = [insert_sql(table, name) for table in prepared]
    events = [
        cf.Data(
            cf.Batch.from_pyarrow(table),
            cf.Cursor((i + 1).to_bytes(8, "big"), {"i": i + 1}),
        )
        for i, table in enumerate(prepared)
    ]
    oracles = [
        expected(kind, prepared[: i + 1] if kind == "grouped" else [table])
        for i, table in enumerate(prepared)
    ]
    return prepared, statements, events, oracles


async def _engine_sample(rw, stream, engine, event, watermark, statement, selection):
    if engine == "calc_flow":
        return await measured_call(
            lambda e=event, w=watermark: stream.append(e, w), [os.getpid()]
        )
    return await measured_call(
        lambda s=statement, q=selection: rw.append(s, q),
        [os.getpid(), rw.process.pid],
    )


async def _sample_pair(rw, stream, kind, rows, round_id, index, inputs) -> list[dict]:
    table, event, statement, oracle = inputs
    name = f"bench_{kind}_{rows}_{round_id}"
    watermark = (
        cf.Watermark(BASE + timedelta(seconds=(index + 1) * 10, microseconds=1))
        if kind == "window"
        else None
    )
    selection = _sample_selection(name, kind, index, rows)
    order = (
        ("calc_flow", "risingwave")
        if (index + round_id) % 2 == 0
        else ("risingwave", "calc_flow")
    )
    samples = []
    for engine in order:
        actual, sample = await _engine_sample(
            rw, stream, engine, event, watermark, statement, selection
        )
        require(
            actual == oracle,
            f"{name}/{index}/{engine} differs: {actual[:4]} != {oracle[:4]}",
        )
        samples.append(
            {
                **sample,
                "engine": engine,
                "index": index,
                "warmup": index < WARMUP,
                "input_rows": table.num_rows,
                "payload_rows": rows,
                "output_rows": len(actual),
                "oracle_sha256": digest(oracle),
                "actual_sha256": digest(actual),
                "order": list(order),
            }
        )
    return samples


async def case(rw: RisingWave, kind: str, rows: int, round_id: int, output: Path):
    name = f"bench_{kind}_{rows}_{round_id}"
    # Preparing SQL, Arrow events and oracles remains outside every sample timer.
    prepared, statements, events, oracles = _case_inputs(name, kind, rows)
    record = {
        "case": f"{kind}_{rows}",
        "round": round_id,
        "batch_rows": rows,
        "input_sha256": digest([table.to_pydict() for table in prepared]),
        "samples": [],
    }
    stream = CalcStream(kind, rows, output / f"cf-{name}")
    try:
        await rw.setup(name, kind)
        await stream.start()
        record["rss_ready"] = {
            "driver": process_usage(os.getpid()),
            "risingwave": process_usage(rw.process.pid),
        }
        for index, inputs in enumerate(
            zip(prepared, events, statements, oracles, strict=True)
        ):
            record["samples"].extend(
                await _sample_pair(rw, stream, kind, rows, round_id, index, inputs)
            )
            # Preserve each complete pair before advancing state.
            write_json(output / f"{name}.json", record)
        record["calc_flow_status"] = stream.job.status()
        write_json(output / f"{name}.json", record)
        return record
    finally:
        await stream.close()
        await rw.drop(name, if_exists=True)


async def window_semantics(rw: RisingWave, output: Path) -> dict:
    """Probe the boundary, then a late row after progress; never time this probe."""
    name = "window_semantics"
    await rw.setup(name, "window")
    stream = await CalcStream("window", 640, output / "cf-window-probe").start()
    try:
        first = pa.table(
            {
                "id": [1, 2],
                "k": [0, 0],
                "v": [2, 3],
                "ts": [BASE + timedelta(seconds=9), BASE + timedelta(seconds=10)],
            },
            schema=SCHEMA,
        )
        cf_first, _ = await stream.append(
            cf.Data(cf.Batch.from_pyarrow(first), cf.Cursor(b"1", {"i": 1})),
            cf.Watermark(BASE + timedelta(seconds=10)),
        )
        await rw.execute(insert_sql(first, name))
        await rw.execute("FLUSH")
        rw_equal = normalize(await rw.execute(_select_view(name).as_string()))
        wanted_first = [[BASE_US, BASE_US + 10_000_000, 0, 2, 1]]
        require(cf_first == wanted_first, "CF window-end equality oracle")
        require(
            rw_equal in ([], wanted_first), "RW boundary returned unexpected values"
        )

        late = pa.table(
            {
                "id": [3, 4],
                "k": [0, 0],
                "v": [7, None],
                "ts": [
                    BASE + timedelta(seconds=1),
                    BASE + timedelta(seconds=20, microseconds=1),
                ],
            },
            schema=SCHEMA,
        )
        cf_second, _ = await stream.append(
            cf.Data(cf.Batch.from_pyarrow(late), cf.Cursor(b"2", {"i": 2})),
            cf.Watermark(BASE + timedelta(seconds=20, microseconds=1)),
        )
        await rw.execute(insert_sql(late, name))
        await rw.execute("FLUSH")
        rw_final = normalize(await rw.execute(_select_view(name).as_string()))
        wanted_second = [[BASE_US + 10_000_000, BASE_US + 20_000_000, 0, 3, 1]]
        require(cf_second == wanted_second, "CF late row changed a closed window")
        write_json(
            output / "window-diagnostic.json",
            {
                "cf_first": cf_first,
                "rw_equal": rw_equal,
                "cf_second": cf_second,
                "rw_final": rw_final,
                "wanted": wanted_first + wanted_second,
            },
        )
        require(rw_final == wanted_first + wanted_second, "RW late/boundary oracle")
        await stream.source.push(None)
        outcome = await asyncio.wait_for(stream.job.wait_async(), timeout=30)
        require(outcome.state == "completed", "CF probe EOF failed")
        eof_table = await stream.sink.receive(timeout=1)
        eof_rows = normalize(
            list(
                zip(
                    *(
                        eof_table[c].to_pylist()
                        for c in ("window_start", "window_end", "k", "total", "n")
                    ),
                    strict=True,
                )
            )
        )
        require(
            eof_rows == [[BASE_US + 20_000_000, BASE_US + 30_000_000, 0, None, 0]],
            "CF EOF did not flush the open progress window",
        )
        require(stream.sink._tables.empty(), "unexpected extra CF window output")
        return {
            "passed": True,
            "cf_at_end_equality": cf_first,
            "rw_at_end_equality": rw_equal,
            "cf_after_late": cf_second,
            "rw_after_late": rw_final,
            "cf_eof_open_window": eof_rows,
            "late_value": 7,
            "window": "[start,end); ts=10 belongs to [10,20)",
        }
    finally:
        await stream.close()
        await rw.drop(name, if_exists=False)


async def semantics(rw: RisingWave, output: Path) -> dict:
    result = {}
    await rw.execute("CREATE TABLE mutations(id INT PRIMARY KEY, k INT, v BIGINT)")
    await rw.execute(
        "CREATE MATERIALIZED VIEW mutation_totals AS "
        "SELECT k, SUM(v) AS total, COUNT(v) AS n FROM mutations GROUP BY k"
    )
    await rw.execute("INSERT INTO mutations VALUES (1,0,10),(2,0,20),(3,1,NULL)")
    await rw.execute("FLUSH")
    initial = normalize(await rw.execute("SELECT * FROM mutation_totals"))
    require(initial == [[0, 30, 2], [1, None, 0]], "initial mutation oracle")
    await rw.execute("UPDATE mutations SET v=50 WHERE id=1")
    await rw.execute("DELETE FROM mutations WHERE id=2")
    await rw.execute("FLUSH")
    updated = normalize(await rw.execute("SELECT * FROM mutation_totals"))
    require(updated == [[0, 50, 1], [1, None, 0]], "update/delete oracle")
    result["rw_update_delete"] = {"initial": initial, "after": updated, "passed": True}
    write_json(output / "semantics.json", result)

    stream = await CalcStream("grouped", 640, output / "cf-mutation").start()
    try:
        snapshots = []
        for index, value in enumerate((10, 50)):
            t = pa.table(
                {"id": [1], "k": [0], "v": [value], "ts": [BASE]}, schema=SCHEMA
            )
            actual, _ = await stream.append(
                cf.Data(
                    cf.Batch.from_pyarrow(t),
                    cf.Cursor(str(index).encode(), {"i": index}),
                ),
                None,
            )
            snapshots.append(actual)
        require(snapshots == [[[0, 10, 1]], [[0, 60, 2]]], "Calc Flow append oracle")
        result["cf_repeated_id_is_append"] = {"snapshots": snapshots, "passed": True}
    finally:
        await stream.close()
    try:
        source = cf.table_input("events", schema=SCHEMA)
        cf.Program(
            "reject-update",
            engine="streaming",
            outputs={"result": source.sql("UPDATE input SET v=50 WHERE id=1")},
        ).compile_stream(cf.Runtime())
    except Exception as error:
        result["cf_update_sql_rejected"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    else:
        raise RuntimeError("Calc Flow unexpectedly accepted UPDATE")

    # Reuse the maintained replay/cursor/checkpoint example; this is orderly
    # cancellation and a new runner, not a process crash or a latency benchmark.
    recover_midstream = importlib.import_module(
        "examples.10_symbolic_streaming_recovery"
    ).recover_midstream

    capture = StringIO()
    with redirect_stdout(capture):
        await recover_midstream(output / "cf-recovery")
    result["cf_managed_recovery"] = {
        "passed": True,
        "stdout": capture.getvalue(),
        "mode": "checkpoint/cancel/new-runner/terminal-restart",
    }
    write_json(output / "semantics.json", result)

    await rw.stop(crash=True)
    await rw.start()
    recovered = normalize(await rw.execute("SELECT * FROM mutation_totals"))
    require(recovered == updated, "RisingWave persisted MV changed after SIGKILL")
    await rw.execute("INSERT INTO mutations VALUES(4,0,7)")
    await rw.execute("FLUSH")
    resumed = normalize(await rw.execute("SELECT * FROM mutation_totals"))
    require(resumed == [[0, 57, 2], [1, None, 0]], "RisingWave resumed update")
    result["rw_recovery"] = {
        "passed": True,
        "after_sigkill": recovered,
        "resumed": resumed,
        "mode": "FLUSH/SIGKILL/same-store/restart",
    }
    write_json(output / "semantics.json", result)
    result["window_boundary_and_late"] = await window_semantics(rw, output)
    write_json(output / "semantics.json", result)
    return result


def _case_samples(records: list[dict], case_id: str, engine: str) -> list[dict]:
    return [
        sample
        for record in records
        if record["case"] == case_id
        for sample in record["samples"]
        if sample["engine"] == engine and not sample["warmup"]
    ]


def _latency_summary(samples: list[dict]) -> dict:
    durations = [sample["seconds"] for sample in samples]
    quantiles = np.percentile(durations, [50, 95, 99]).tolist()
    return {
        "count": len(samples),
        "p50_seconds": quantiles[0],
        "p95_seconds": quantiles[1],
        "p99_seconds": quantiles[2],
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "payload_rows_per_second": sum(sample["payload_rows"] for sample in samples)
        / sum(durations),
    }


def _resource_summary(samples: list[dict]) -> dict:
    cpu_seconds = sum(sample["cpu_seconds"] for sample in samples)
    payload_rows = sum(sample["payload_rows"] for sample in samples)
    return {
        "cpu_seconds": cpu_seconds,
        "cpu_seconds_per_million_rows": cpu_seconds / payload_rows * 1_000_000,
        "rss_peak_bytes": max(sample["rss_peak_bytes"] for sample in samples),
        "engine_rss_peak_bytes": max(
            sample["engine_rss_peak_bytes"] for sample in samples
        ),
        "write_bytes": sum(sample["write_bytes"] for sample in samples),
    }


def _round_medians(records: list[dict], case_id: str, engine: str) -> list[float]:
    return [
        statistics.median(
            sample["seconds"]
            for sample in _case_samples(
                [record for record in records if record["round"] == rid],
                case_id,
                engine,
            )
        )
        for rid in range(ROUNDS)
    ]


def _case_summary(records: list[dict], case_id: str, engine: str) -> dict:
    samples = _case_samples(records, case_id, engine)
    return {
        "case": case_id,
        "engine": engine,
        **_latency_summary(samples),
        **_resource_summary(samples),
        "round_p50_seconds": _round_medians(records, case_id, engine),
    }


def summaries(records: list[dict]) -> list[dict]:
    return [
        _case_summary(records, f"{kind}_{rows}", engine)
        for kind, rows in CASES
        for engine in ("calc_flow", "risingwave")
    ]


async def _cpu_description() -> str:
    # This Linux-only collector invokes a fixed system binary, never PATH or
    # command text supplied by a workload, CLI argument, or shell.
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/lscpu", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    require(process.returncode == 0, f"lscpu failed: {stderr.decode()}")
    return stdout.decode()


async def provenance(args) -> dict:
    import calc_flow._native as native

    run = json.loads(args.ci_run.read_text())
    require(run["head_sha"] == CF_COMMIT, "CI wheel source commit mismatch")
    require(
        sha256_file(Path(native.__file__)) == wheel_native_sha256(args.wheel),
        "imported native module does not match CI wheel",
    )
    return {
        "contract": "dal416-application-boundary-v1",
        "utc": datetime.now(UTC).isoformat(),
        "calc_flow_commit": CF_COMMIT,
        "risingwave_commit": RW_COMMIT,
        "ci_run_url": run["html_url"],
        "wheel": args.wheel.name,
        "wheel_sha256": sha256_file(args.wheel),
        "native_sha256": sha256_file(Path(native.__file__)),
        "native_import": str(native.__file__),
        "rw_binary_sha256": sha256_file(args.risingwave),
        "python": sys.version,
        "kernel": platform.platform(),
        "argv": sys.argv,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "address_space_limit": list(resource.getrlimit(resource.RLIMIT_AS)),
        "ram": dict(psutil.virtual_memory()._asdict()),
        "load_average": os.getloadavg(),
        "lscpu": await _cpu_description(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in (
                "calc-flow-python",
                "numpy",
                "pyarrow",
                "psutil",
                "psycopg",
                "psycopg-binary",
            )
        },
        "harness_sha256": sha256_file(Path(__file__)),
        "warm_source_sha256": sha256_file(Path("benchmarks/warm_stream.py")),
        "cases": CASES,
        "rounds": ROUNDS,
        "warmup_per_round": WARMUP,
        "measured_per_round": SAMPLES,
        "budget_seconds": 600,
        "durability": {
            "calc_flow": "checkpointing=False; Arrow ordinary sink",
            "risingwave": "local persistent single_node; INSERT + FLUSH + SELECT",
        },
        "verdict": "descriptive only; unequal durability/serving; no engine ranking",
    }


def validate_reuse(previous: Path, identity: dict) -> None:
    old = json.loads((previous / "provenance.json").read_text())
    for field in (
        "calc_flow_commit",
        "risingwave_commit",
        "wheel_sha256",
        "native_sha256",
        "rw_binary_sha256",
        "packages",
        "cpu_affinity",
        "address_space_limit",
        "kernel",
        "lscpu",
        "warm_source_sha256",
    ):
        require(old[field] == identity[field], f"incompatible reuse identity: {field}")
    original = previous / "harness.py"
    require(
        sha256_file(original) == old["harness_sha256"], "original harness hash differs"
    )
    protected = {
        "fixture",
        "insert_sql",
        "program",
        "CalcStream",
        "RisingWave",
        "case",
        "measured_call",
        "observe_peak",
        "process_usage",
        "normalize",
    }

    def timed_nodes(path: Path):
        return [
            ast.dump(node)
            for node in ast.parse(
                path.read_text().replace(
                    "seconds=(batch + 1) * 10, microseconds=1",
                    "seconds=(batch + 1) * 10",
                )
            ).body
            if isinstance(node, ast.Assign) or getattr(node, "name", None) in protected
        ]

    require(
        timed_nodes(original) == timed_nodes(Path(__file__)), "timed workload changed"
    )


def _record_reuse(args, identity: dict, output: Path) -> None:
    if args.reuse is None:
        return
    validate_reuse(args.reuse, identity)
    write_json(
        output / "reuse.json",
        {
            "source": str(args.reuse.resolve()),
            "provenance_sha256": sha256_file(args.reuse / "provenance.json"),
            "reason": "untimed oracle conversion changed; timed AST equal",
        },
    )


def _reused_case(args, kind: str, rows: int, rid: int, output: Path) -> dict | None:
    if args.reuse is None:
        return None
    previous = args.reuse / f"bench_{kind}_{rows}_{rid}.json"
    if not previous.is_file():
        return None
    record = json.loads(previous.read_text())
    if len(record["samples"]) != 2 * (WARMUP + SAMPLES):
        return None
    tables = _prepared_tables(kind, rows)
    require(
        record["input_sha256"] == digest([t.to_pydict() for t in tables]),
        "reused inputs changed",
    )
    write_json(output / previous.name, record)
    return record


async def _collect_cases(args, rw: RisingWave, output: Path):
    for rid in range(ROUNDS):
        for kind, rows in CASES:
            record = _reused_case(args, kind, rows, rid, output)
            if record is not None:
                yield record, False
            else:
                print(f"round={rid} case={kind}_{rows}", flush=True)
                yield await case(rw, kind, rows, rid, output), True


async def _start_collection(rw: RisingWave, output: Path) -> None:
    await rw.start()
    version = await rw.execute("SELECT version()")
    require(RW_COMMIT[:10] in str(version), "RisingWave binary version mismatch")
    write_json(output / "rw-version.json", version)
    write_json(output / "rw-parameters.json", await rw.execute("SHOW PARAMETERS"))
    await semantics(rw, output)


def _write_completion(rw, output: Path, start: float, records: list[dict]) -> None:
    write_json(
        output / "completion.json",
        {
            "elapsed_seconds": time.perf_counter() - start,
            "completed_cases": len(records),
            "load_average_after": os.getloadavg(),
            "risingwave_returncode": rw.process.returncode if rw.process else None,
        },
    )


async def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "risingwave.toml").write_text(
        "[server]\ntelemetry_enabled = false\n", encoding="utf-8"
    )
    start = time.perf_counter()
    identity = await provenance(args)
    identity["budget_seconds"] = args.budget_seconds
    write_json(output / "provenance.json", identity)
    rw = RisingWave(args, output)
    records = []
    try:
        async with asyncio.timeout(args.budget_seconds):
            _record_reuse(args, identity, output)
            await _start_collection(rw, output)
            async for record, measured in _collect_cases(args, rw, output):
                records.append(record)
                if measured:
                    write_json(output / "summary.json", summaries_partial(records))
            write_json(output / "summary.json", summaries(records))
    except BaseException as error:
        write_json(
            output / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        await rw.stop()
        _write_completion(rw, output, start, records)


def summaries_partial(records: list[dict]) -> dict:
    return {"completed": [r["case"] for r in records], "final": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risingwave", type=Path, required=True)
    parser.add_argument("--python312-lib", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--ci-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--budget-seconds", type=int, default=600)
    args = parser.parse_args()
    require(0 < args.budget_seconds <= 600, "budget must be within 1..600 seconds")
    args.risingwave = args.risingwave.resolve()
    # Identical process CPU/address-space ceilings; RSS is observed, not capped.
    os.sched_setaffinity(0, {0, 1, 2, 3})
    resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
