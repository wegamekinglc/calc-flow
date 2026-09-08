"""Native-oracle execution and managed recovery for symbolic event windows."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    Cursor,
    Data,
    Idle,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    PipelineBuilder,
    ReplayPositioning,
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    SourceProvidedWatermarks,
    StreamExecutionPlan,
    StreamingRunner,
    StreamingRuntimeError,
    Watermark,
)
from calc_flow.symbolic import FeatureSet, Field, Program, table, table_input, window
from calc_flow.symbolic.lower import lower_program_document

BASE = datetime(2026, 1, 1, tzinfo=UTC)
MINUTE = 60_000_000


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC")),
            pa.field("symbol", pa.string()),
            pa.field("trade_id", pa.uint64(), nullable=False),
            pa.field("quantity", pa.int64()),
            pa.field("price", pa.float64()),
        ]
    )


def _trades() -> pa.Table:
    return pa.table(
        {
            "ts": [
                BASE + timedelta(seconds=seconds) if seconds is not None else None
                for seconds in [5, 35, 59, 15, 60, 95, None]
            ],
            "symbol": ["A", "A", "A", "B", "A", "A", "A"],
            "trade_id": list(range(1, 8)),
            "quantity": [10, None, 30, None, 7, 13, 999],
            "price": [100.0, 102.0, None, None, 110.0, 90.0, 999.0],
        },
        schema=_schema(),
    )


def _program(*, hopping: bool = False, fanout: bool = False) -> Program:
    trades = table_input(
        "trades",
        schema=[
            Field("ts", "timestamp[us, UTC]"),
            Field("symbol", "string"),
            Field("trade_id", "uint64", nullable=False),
            Field("quantity", "int64"),
            Field("price", "float64"),
        ],
    )

    def declaration():
        aggregates = [
            window.count("trade_id", output="trade_count"),
            window.sum("quantity", output="volume"),
            window.min("price", output="low"),
            window.max("price", output="high"),
            window.avg("price", output="avg_price"),
        ]
        if hopping:
            return window.hopping(
                trades,
                event_time="ts",
                size_micros=2 * MINUTE,
                slide_micros=MINUTE,
                group_by=["symbol"],
                aggregates=aggregates,
            )
        return window.tumbling(
            trades,
            event_time="ts",
            size_micros=MINUTE,
            group_by=["symbol"],
            aggregates=aggregates,
        )

    minute = declaration()
    outputs = [("minute", minute)]
    if fanout:
        outputs.append(("copy", declaration()))
    return Program("minute-bars", inputs=[trades], outputs=outputs)


def test_minute_aggregates_compile_to_native_window() -> None:
    document = lower_program_document(_program(), Runtime(), "stream")
    windows = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "window"
    ]
    assert len(windows) == 1
    assert windows[0]["operator"]["spec"]["geometry"] == {
        "kind": "tumbling",
        "size_micros": MINUTE,
    }


def _native_document(*, hopping: bool = False) -> dict[str, object]:
    geometry = (
        {"kind": "hopping", "size_micros": 2 * MINUTE, "slide_micros": MINUTE}
        if hopping
        else {"kind": "tumbling", "size_micros": MINUTE}
    )
    return {
        "format_version": 3,
        "id": "native-minute-bars",
        "name": "native-minute-bars",
        "runtime": {"mode": "stream", "options": {}},
        "data_sources": [],
        "graph": {
            "name": "native-minute-bars",
            "edges": [],
            "nodes": [
                {
                    "id": "native_window",
                    "input_ports": [
                        {
                            "name": "input",
                            "kind": "table",
                            "required": True,
                            "schema": [
                                {
                                    "name": "ts",
                                    "data_type": "timestamp[us, UTC]",
                                    "nullable": True,
                                },
                                {
                                    "name": "symbol",
                                    "data_type": "string",
                                    "nullable": True,
                                },
                                {
                                    "name": "trade_id",
                                    "data_type": "uint64",
                                    "nullable": False,
                                },
                                {
                                    "name": "quantity",
                                    "data_type": "int64",
                                    "nullable": True,
                                },
                                {
                                    "name": "price",
                                    "data_type": "float64",
                                    "nullable": True,
                                },
                            ],
                        }
                    ],
                    "output_ports": [],
                    "operator": {
                        "kind": "window",
                        "spec": {
                            "event_time_column": "ts",
                            "group_by": ["symbol"],
                            "geometry": geometry,
                            "aggregates": [
                                {
                                    "function": "count",
                                    "column": "trade_id",
                                    "output": "trade_count",
                                },
                                {
                                    "function": "sum",
                                    "column": "quantity",
                                    "output": "volume",
                                },
                                {"function": "min", "column": "price", "output": "low"},
                                {
                                    "function": "max",
                                    "column": "price",
                                    "output": "high",
                                },
                                {
                                    "function": "avg",
                                    "column": "price",
                                    "output": "avg_price",
                                },
                            ],
                        },
                    },
                }
            ],
        },
    }


def _compile(kind: str, *, hopping: bool = False) -> StreamExecutionPlan:
    runtime = Runtime()
    if kind == "native":
        return PipelineBuilder._from_json(
            json.dumps(_native_document(hopping=hopping))
        ).compile_stream(runtime=runtime)
    return _program(hopping=hopping).compile_stream(runtime)


class _ScriptedSource:
    def __init__(
        self,
        events: list[tuple[str, object]],
        *,
        pauses: tuple[int, ...] = (),
        opened: list[int] | None = None,
    ) -> None:
        self._events = tuple(events)
        self._pauses = set(pauses)
        self._reported: set[int] = set()
        self._opened = opened
        self._index = 0
        self.paused: asyncio.Queue[int] = asyncio.Queue()

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=64,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._index = 0 if cursor is None else int(cursor.payload["index"])
        if self._opened is not None:
            self._opened.append(self._index)

    async def next(self) -> Data | Watermark | Idle | None:
        if self._index in self._pauses:
            if self._index not in self._reported:
                self._reported.add(self._index)
                self.paused.put_nowait(self._index)
            await asyncio.sleep(0)
            return Idle()
        if self._index == len(self._events):
            return None
        kind, payload = self._events[self._index]
        self._index += 1
        if kind == "watermark":
            return Watermark(BASE + timedelta(seconds=int(payload)))
        return Data(
            Batch.from_pyarrow(payload),
            Cursor(self._index.to_bytes(8, "big"), {"index": self._index}),
        )

    def release(self, index: int) -> None:
        self._pauses.remove(index)

    async def close(self) -> None:
        return None


class _CollectSink:
    def __init__(self) -> None:
        self.tables: list[pa.Table] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        value = batch.to_pyarrow()
        if value.num_rows:
            self.tables.append(value)

    async def close(self) -> None:
        return None


def _runner(
    plan: StreamExecutionPlan,
    source: _ScriptedSource,
    sinks: dict[str, _CollectSink],
    root: Path,
) -> StreamingRunner:
    assert len(plan.source_binding_ids) == 1
    return StreamingRunner(
        plan,
        {
            plan.source_binding_ids[0]: SourceBinding(
                source, watermark_policy=SourceProvidedWatermarks()
            )
        },
        {
            output: [SinkBinding.ordinary(f"archive_{index}", sink)]
            for index, (output, sink) in enumerate(sinks.items())
        },
        ManagedCheckpointRuntime(root),
    )


def _one_sink(plan: StreamExecutionPlan, sink: _CollectSink) -> dict[str, _CollectSink]:
    assert len(plan.sink_binding_ids) == 1
    return {plan.sink_binding_ids[0]: sink}


def _window_metrics(status: dict[str, object]) -> dict[str, int | None]:
    fields = (
        "late_rows",
        "late_affected_batches",
        "max_lateness_micros",
        "null_event_time_rows",
        "null_event_time_batches",
    )
    operators = status["operators"]
    return {
        field: (
            max(
                (
                    operator[field]
                    for operator in operators.values()
                    if operator[field] is not None
                ),
                default=None,
            )
            if field == "max_lateness_micros"
            else sum(operator[field] for operator in operators.values())
        )
        for field in fields
    }


async def _complete(
    plan: StreamExecutionPlan,
    events: list[tuple[str, object]],
    root: Path,
) -> tuple[pa.Table, dict[str, int | None]]:
    sink = _CollectSink()
    job = await _runner(
        plan, _ScriptedSource(events), _one_sink(plan, sink), root
    ).start_async()
    try:
        outcome = await asyncio.wait_for(job.wait_async(), timeout=30)
        assert outcome.state == "completed", outcome
        return pa.concat_tables(sink.tables), _window_metrics(job.status())
    finally:
        if job.status()["state"] == "running":
            await job.cancel_async()


def _assert_tables_equal(actual: pa.Table, expected: pa.Table) -> None:
    assert actual.schema == expected.schema
    assert actual.num_rows == expected.num_rows
    for field in expected.schema:
        assert actual[field.name].is_null().equals(expected[field.name].is_null())
        if pa.types.is_floating(field.type):
            for observed, wanted in zip(
                actual[field.name].to_pylist(),
                expected[field.name].to_pylist(),
                strict=True,
            ):
                if wanted is None:
                    assert observed is None
                elif math.isnan(wanted):
                    assert math.isnan(observed)
                else:
                    assert struct.pack("!d", observed) == struct.pack("!d", wanted)
        else:
            assert actual[field.name].equals(expected[field.name])


def _table_snapshot(value: pa.Table) -> tuple[pa.Schema, list[list[object]]]:
    """Preserve schema and epoch values without consulting host timezone data."""
    return value.schema, [
        (
            column.cast(pa.int64()).to_pylist()
            if pa.types.is_timestamp(column.type)
            else column.to_pylist()
        )
        for column in value.columns
    ]


def _split_events(segmentation: str) -> list[tuple[str, object]]:
    value = _trades()
    if segmentation == "single":
        tables = [value]
    elif segmentation == "rows":
        tables = [value.slice(index, 1) for index in range(value.num_rows)]
    elif segmentation == "irregular":
        tables = [value.slice(0, 2), value.slice(2, 3), value.slice(5)]
    else:
        tables = [
            value.slice(0, 0),
            pa.Table.from_batches(value.to_batches(max_chunksize=2)),
            value.slice(0, 0),
        ]
        assert tables[1].column(0).num_chunks > 1
    return [
        *(("data", table) for table in tables),
        ("watermark", 60),
        ("watermark", 120),
    ]


@pytest.mark.parametrize("hopping", [False, True], ids=["tumbling", "hopping"])
@pytest.mark.parametrize(
    "segmentation", ["single", "rows", "irregular", "empty-and-chunked"]
)
def test_batch_splits_with_fixed_watermarks_match_native(
    tmp_path: Path, hopping: bool, segmentation: str
) -> None:
    events = _split_events(segmentation)
    originals = [value for kind, value in events if kind == "data"]
    snapshots = [_table_snapshot(value) for value in originals]

    async def exercise() -> None:
        expected, native_metrics = await _complete(
            _compile("native", hopping=hopping), events, tmp_path / "native"
        )
        actual, symbolic_metrics = await _complete(
            _compile("symbolic", hopping=hopping), events, tmp_path / "symbolic"
        )
        _assert_tables_equal(actual, expected)
        assert symbolic_metrics == native_metrics
        assert symbolic_metrics["null_event_time_rows"] == 1
        assert symbolic_metrics["late_rows"] == 0
        if hopping:
            assert actual["symbol"].to_pylist() == ["A", "B", "A", "B", "A"]
            assert actual["trade_count"].to_pylist() == [3, 1, 5, 1, 2]
            assert actual["volume"].to_pylist() == [40, None, 60, None, 20]
            assert actual["avg_price"].to_pylist() == [101.0, None, 100.5, None, 100.0]
        if not hopping:
            assert actual["symbol"].to_pylist() == ["A", "B", "A"]
            assert actual["trade_count"].to_pylist() == [3, 1, 2]
            assert actual["volume"].to_pylist() == [40, None, 20]
            assert actual["low"].to_pylist() == [100.0, None, 90.0]
            assert actual["high"].to_pylist() == [102.0, None, 110.0]
            assert actual["avg_price"].to_pylist() == [101.0, None, 100.0]
            assert actual.schema == pa.schema(
                [
                    pa.field(
                        "window_start", pa.timestamp("us", tz="UTC"), nullable=False
                    ),
                    pa.field(
                        "window_end", pa.timestamp("us", tz="UTC"), nullable=False
                    ),
                    pa.field("symbol", pa.string()),
                    pa.field("trade_count", pa.uint64(), nullable=False),
                    pa.field("volume", pa.int64()),
                    pa.field("low", pa.float64()),
                    pa.field("high", pa.float64()),
                    pa.field("avg_price", pa.float64()),
                ]
            )

    asyncio.run(exercise())
    assert [_table_snapshot(value) for value in originals] == snapshots


def test_native_fixture_uses_handwritten_window_spec(tmp_path: Path) -> None:
    output, metrics = asyncio.run(
        _complete(_compile("native"), _split_events("single"), tmp_path)
    )
    assert output["trade_count"].to_pylist() == [3, 1, 2]
    assert metrics["null_event_time_rows"] == 1


def _filtered_window_native_plan() -> StreamExecutionPlan:
    document = _native_document()
    native_window = document["graph"]["nodes"][0]
    original_schema = native_window["input_ports"][0]["schema"]
    native_window["input_ports"][0]["schema"] = [
        {"name": "event_ts", "data_type": "timestamp[us, UTC]", "nullable": True},
        {"name": "symbol", "data_type": "string", "nullable": True},
        {"name": "trade_id", "data_type": "uint64", "nullable": False},
        {"name": "amount", "data_type": "int64", "nullable": True},
    ]
    native_window["operator"]["spec"]["event_time_column"] = "event_ts"
    native_window["operator"]["spec"]["aggregates"] = [
        {"function": "sum", "column": "amount", "output": "volume"},
        {"function": "count", "column": "trade_id", "output": "trade_count"},
    ]
    native_output_schema = [
        {"name": "window_start", "data_type": "timestamp[us, UTC]", "nullable": False},
        {"name": "window_end", "data_type": "timestamp[us, UTC]", "nullable": False},
        {"name": "symbol", "data_type": "string", "nullable": True},
        {"name": "volume", "data_type": "int64", "nullable": True},
        {"name": "trade_count", "data_type": "uint64", "nullable": False},
    ]
    document["graph"]["nodes"] = [
        {
            "id": "prepare",
            "input_ports": [
                {
                    "name": "input",
                    "kind": "table",
                    "required": True,
                    "schema": original_schema,
                }
            ],
            "output_ports": [
                {
                    "name": "output",
                    "kind": "table",
                    "required": True,
                    "schema": native_window["input_ports"][0]["schema"],
                }
            ],
            "operator": {
                "kind": "expression",
                "expression": "",
                "select": [
                    '"ts" AS "event_ts"',
                    '"symbol"',
                    '"trade_id"',
                    '("quantity" * 2) AS "amount"',
                ],
                "filter": '"quantity" > 0',
                "udfs": [],
            },
        },
        native_window,
        {
            "id": "all",
            "input_ports": [
                {
                    "name": "input",
                    "kind": "table",
                    "required": True,
                    "schema": native_output_schema,
                }
            ],
            "operator": {
                "kind": "expression",
                "expression": "",
                "select": [
                    '"window_start"',
                    '"window_end"',
                    '"symbol"',
                    '"volume"',
                    '"trade_count"',
                ],
                "filter": None,
                "udfs": [],
            },
        },
        {
            "id": "selected",
            "input_ports": [
                {
                    "name": "input",
                    "kind": "table",
                    "required": True,
                    "schema": native_output_schema,
                }
            ],
            "operator": {
                "kind": "expression",
                "expression": "",
                "select": [
                    '"symbol"',
                    '"window_end"',
                    '"volume"',
                    '("volume" - 25) AS "excess"',
                ],
                "filter": '"volume" > 25',
                "udfs": [],
            },
        },
    ]
    document["graph"]["edges"] = [
        {
            "source_node": source,
            "source_port": "output",
            "target_node": target,
            "target_port": "input",
        }
        for source, target in [
            ("prepare", "native_window"),
            ("native_window", "all"),
            ("native_window", "selected"),
        ]
    ]
    native_plan = PipelineBuilder._from_json(json.dumps(document)).compile_stream(
        runtime=Runtime()
    )
    return native_plan


def test_filters_and_derived_columns_preserve_shared_window_boundary(
    tmp_path: Path,
) -> None:
    trades = _program().inputs[0]
    positive = table.filter(trades, trades["quantity"] > 0)
    prepared = positive.with_columns(
        FeatureSet([("event_ts", positive["ts"]), ("amount", positive["quantity"] * 2)])
    )
    prepared = table.project(prepared, ["event_ts", "symbol", "trade_id", "amount"])
    minute = window.tumbling(
        prepared,
        event_time="event_ts",
        size_micros=MINUTE,
        group_by=["symbol"],
        aggregates=[
            window.sum("amount", output="volume"),
            window.count("trade_id", output="trade_count"),
        ],
    )
    selected = table.filter(minute, minute["volume"] > 25)
    selected = selected.with_columns(FeatureSet([("excess", selected["volume"] - 25)]))
    selected = table.project(selected, ["symbol", "window_end", "volume", "excess"])
    symbolic_plan = Program(
        "window-filter-boundary",
        inputs=[trades],
        outputs=[("all", minute), ("selected", selected)],
    ).compile_stream(Runtime())

    native_plan = _filtered_window_native_plan()
    values = pa.table(
        {
            "ts": [
                BASE + timedelta(seconds=seconds) if seconds is not None else None
                for seconds in [5, 10, 15, 20, 30, 45, 60, 70, 80, 95, None]
            ],
            "symbol": ["A", "A", "A", "B", "B", "C", "A", "A", "B", "B", "A"],
            "trade_id": list(range(1, 12)),
            "quantity": [6, 7, -100, 20, None, 0, 3, 4, 8, 9, 999],
            "price": [None] * 11,
        },
        schema=_schema(),
    )
    original = _table_snapshot(values)
    events = [
        ("data", values.slice(0, 6)),
        ("data", values.slice(6)),
        ("watermark", 60),
        ("watermark", 120),
    ]
    expected_all = pa.table(
        {
            "window_start": [
                BASE,
                BASE,
                BASE + timedelta(seconds=60),
                BASE + timedelta(seconds=60),
            ],
            "window_end": [BASE + timedelta(seconds=60)] * 2
            + [BASE + timedelta(seconds=120)] * 2,
            "symbol": ["A", "B", "A", "B"],
            "volume": [26, 40, 14, 34],
            "trade_count": [2, 1, 2, 2],
        },
        schema=pa.schema(
            [
                pa.field("window_start", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("window_end", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string()),
                pa.field("volume", pa.int64()),
                pa.field("trade_count", pa.uint64(), nullable=False),
            ]
        ),
    )
    expected_selected = pa.table(
        {
            "symbol": ["A", "B", "B"],
            "window_end": [BASE + timedelta(seconds=60)] * 2
            + [BASE + timedelta(seconds=120)],
            "volume": [26, 40, 34],
            "excess": [1, 15, 9],
        },
        schema=pa.schema(
            [
                pa.field("symbol", pa.string()),
                pa.field("window_end", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("volume", pa.int64()),
                pa.field("excess", pa.int64()),
            ]
        ),
    )

    async def run(plan: StreamExecutionPlan, root: Path) -> dict[str, pa.Table]:
        assert set(plan.sink_binding_ids) == {"all.output", "selected.output"}
        sinks = {name: _CollectSink() for name in plan.sink_binding_ids}
        job = await _runner(plan, _ScriptedSource(events), sinks, root).start_async()
        try:
            outcome = await asyncio.wait_for(job.wait_async(), timeout=30)
            assert outcome.state == "completed", outcome
            return {name: pa.concat_tables(sink.tables) for name, sink in sinks.items()}
        finally:
            if job.status()["state"] == "running":
                await job.cancel_async()

    async def exercise() -> None:
        native_outputs = await run(native_plan, tmp_path / "native")
        symbolic_outputs = await run(symbolic_plan, tmp_path / "symbolic")
        for name, expected in [
            ("all.output", expected_all),
            ("selected.output", expected_selected),
        ]:
            _assert_tables_equal(native_outputs[name], expected)
            _assert_tables_equal(symbolic_outputs[name], expected)

    asyncio.run(exercise())
    assert _table_snapshot(values) == original


def test_watermark_end_closes_exactly_and_eoi_flushes_without_early_output(
    tmp_path: Path,
) -> None:
    async def run(kind: str) -> tuple[pa.Table, dict[str, int | None]]:
        plan = _compile(kind)
        sink = _CollectSink()
        source = _ScriptedSource(
            [("data", _trades()), ("watermark", 60), ("watermark", 90)],
            pauses=(1, 2, 3),
        )
        job = await _runner(
            plan, source, _one_sink(plan, sink), tmp_path / kind
        ).start_async()
        try:
            assert await asyncio.wait_for(source.paused.get(), timeout=30) == 1
            assert await job.trigger_checkpoint_async() == 1
            assert sink.tables == []
            source.release(1)
            assert await asyncio.wait_for(source.paused.get(), timeout=30) == 2
            assert await job.trigger_checkpoint_async() == 2
            first_window = pa.concat_tables(sink.tables)
            assert first_window["symbol"].to_pylist() == ["A", "B"]
            assert first_window["window_end"].equals(
                pa.chunked_array(
                    [[BASE + timedelta(seconds=60)] * 2],
                    type=pa.timestamp("us", tz="UTC"),
                )
            )
            source.release(2)
            assert await asyncio.wait_for(source.paused.get(), timeout=30) == 3
            assert await job.trigger_checkpoint_async() == 3
            _assert_tables_equal(pa.concat_tables(sink.tables), first_window)
            source.release(3)
            outcome = await asyncio.wait_for(job.wait_async(), timeout=30)
            assert outcome.state == "completed", outcome
            return pa.concat_tables(sink.tables), _window_metrics(job.status())
        finally:
            if job.status()["state"] == "running":
                await job.cancel_async()

    async def exercise() -> None:
        expected, native_metrics = await run("native")
        actual, symbolic_metrics = await run("symbolic")
        _assert_tables_equal(actual, expected)
        assert actual.num_rows == 3
        assert actual["window_end"][-1].equals(
            pa.scalar(BASE + timedelta(seconds=120), type=pa.timestamp("us", tz="UTC"))
        )
        assert symbolic_metrics == native_metrics

    asyncio.run(exercise())


def test_hopping_late_row_contributes_only_to_still_open_assignment(
    tmp_path: Path,
) -> None:
    first = _trades().slice(0, 1)
    late = pa.table(
        {
            "ts": [BASE + timedelta(seconds=30), BASE + timedelta(seconds=60), None],
            "symbol": ["A", "A", "A"],
            "trade_id": [8, 9, 10],
            "quantity": [7, 13, 999],
            "price": [110.0, 90.0, 999.0],
        },
        schema=_schema(),
    )
    events = [("data", first), ("watermark", 60), ("data", late)]

    async def exercise() -> None:
        expected, native_metrics = await _complete(
            _compile("native", hopping=True), events, tmp_path / "native"
        )
        actual, symbolic_metrics = await _complete(
            _compile("symbolic", hopping=True), events, tmp_path / "symbolic"
        )
        _assert_tables_equal(actual, expected)
        assert actual["window_start"].equals(
            pa.chunked_array(
                [[BASE + timedelta(seconds=seconds) for seconds in [-60, 0, 60]]],
                type=pa.timestamp("us", tz="UTC"),
            )
        )
        assert actual["trade_count"].to_pylist() == [1, 3, 1]
        assert actual["volume"].to_pylist() == [10, 30, 13]
        assert symbolic_metrics == native_metrics
        assert symbolic_metrics["late_rows"] == 1
        assert symbolic_metrics["null_event_time_rows"] == 1

    asyncio.run(exercise())


def _manifest(root: Path) -> tuple[Path, dict[str, object]]:
    paths = list((root / "manifests").glob("manifest-*.json"))
    assert len(paths) == 1
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


async def _checkpoint_pending(
    plan: StreamExecutionPlan,
    events: list[tuple[str, object]],
    root: Path,
    opened: list[int],
    *,
    pause_at: int,
) -> dict[str, _CollectSink]:
    sinks = {output: _CollectSink() for output in plan.sink_binding_ids}
    source = _ScriptedSource(events, pauses=(pause_at,), opened=opened)
    job = await _runner(plan, source, sinks, root).start_async()
    try:
        assert await asyncio.wait_for(source.paused.get(), timeout=30) == pause_at
        assert await job.trigger_checkpoint_async() == 1
        assert all(not sink.tables for sink in sinks.values())
    finally:
        assert (await job.cancel_async()).state == "cancelled"
    return sinks


@pytest.mark.parametrize("hopping", [False, True], ids=["tumbling", "hopping"])
def test_durable_unclosed_window_restart_with_new_runtime_matches_native(
    tmp_path: Path, hopping: bool
) -> None:
    events = _split_events("irregular")
    pause_at = 2

    async def run(kind: str) -> pa.Table:
        expected, _ = await _complete(
            _compile(kind, hopping=hopping), events, tmp_path / kind / "uninterrupted"
        )
        root = tmp_path / kind / "restart"
        opened: list[int] = []
        first_plan = _compile(kind, hopping=hopping)
        fingerprint = first_plan.fingerprint
        sinks = await _checkpoint_pending(
            first_plan, events, root, opened, pause_at=pause_at
        )
        second_plan = _compile(kind, hopping=hopping)
        assert second_plan.fingerprint == fingerprint
        assert tuple(sinks) == second_plan.sink_binding_ids
        second = await _runner(
            second_plan, _ScriptedSource(events, opened=opened), sinks, root
        ).start_async()
        try:
            outcome = await asyncio.wait_for(second.wait_async(), timeout=30)
            assert outcome.state == "completed", outcome
        finally:
            if second.status()["state"] == "running":
                await second.cancel_async()
        actual = pa.concat_tables(next(iter(sinks.values())).tables)
        _assert_tables_equal(actual, expected)
        assert opened == [0, pause_at]
        return actual

    async def exercise() -> None:
        expected = await run("native")
        actual = await run("symbolic")
        _assert_tables_equal(actual, expected)

    asyncio.run(exercise())


def test_equal_rebuilt_declarations_share_one_manifest_state_owner(
    tmp_path: Path,
) -> None:
    program = _program(fanout=True)
    document = lower_program_document(program, Runtime(), "stream")
    window_ids = [
        node["id"]
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "window"
    ]
    assert len(window_ids) == 1
    rebuilt = lower_program_document(_program(fanout=True), Runtime(), "stream")
    assert rebuilt == document
    events = [("data", _trades())]

    async def exercise() -> None:
        state_root = tmp_path / "shared"
        opened: list[int] = []
        sinks = await _checkpoint_pending(
            program.compile_stream(Runtime()), events, state_root, opened, pause_at=1
        )
        assert len(sinks) == 2
        _, manifest = await asyncio.to_thread(_manifest, state_root)
        owners = {
            owner: entry
            for owner, entry in manifest["operators"].items()
            if entry["segments"]
        }
        assert list(owners) == window_ids
        entry = owners[window_ids[0]]
        assert entry["inline_metadata"]["state_layout_version"] == 1
        assert entry["inline_metadata"]["next_output_sequence"] == 0
        assert all(
            segment["operator_id"] == window_ids[0] for segment in entry["segments"]
        )
        recovered_plan = _program(fanout=True).compile_stream(Runtime())
        job = await _runner(
            recovered_plan, _ScriptedSource(events, opened=opened), sinks, state_root
        ).start_async()
        try:
            outcome = await asyncio.wait_for(job.wait_async(), timeout=30)
            assert outcome.state == "completed", outcome
        finally:
            if job.status()["state"] == "running":
                await job.cancel_async()
        expected, _ = await _complete(_compile("native"), events, tmp_path / "native")
        for sink in sinks.values():
            _assert_tables_equal(pa.concat_tables(sink.tables), expected)
        assert opened == [0, 1]

    asyncio.run(exercise())


def test_distinct_window_declarations_own_distinct_manifest_state(
    tmp_path: Path,
) -> None:
    base = _program(fanout=True)
    trades = base.inputs[0]
    two_minutes = window.tumbling(
        trades,
        event_time="ts",
        size_micros=2 * MINUTE,
        group_by=["symbol"],
        aggregates=[window.count("price", output="price_count")],
    )
    program = Program(
        base.name,
        inputs=base.inputs,
        outputs=[*base.outputs, ("two_minutes", two_minutes)],
    )
    document = lower_program_document(program, Runtime(), "stream")
    window_ids = {
        node["id"]
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "window"
    }
    assert len(window_ids) == 2
    asyncio.run(
        _checkpoint_pending(
            program.compile_stream(Runtime()),
            [("data", _trades())],
            tmp_path,
            [],
            pause_at=1,
        )
    )
    _, manifest = _manifest(tmp_path)
    owners = {
        owner for owner, entry in manifest["operators"].items() if entry["segments"]
    }
    assert owners == window_ids
    assert len(manifest["sources"]) == 1


@pytest.mark.parametrize("hopping", [False, True], ids=["tumbling", "hopping"])
def test_empty_batches_and_empty_run_emit_no_window_rows(
    tmp_path: Path, hopping: bool
) -> None:
    async def exercise() -> None:
        for kind in ("native", "symbolic"):
            plan = _compile(kind, hopping=hopping)
            sink = _CollectSink()
            source = _ScriptedSource(
                [("data", _trades().slice(0, 0)), ("watermark", 180)]
            )
            job = await _runner(
                plan, source, _one_sink(plan, sink), tmp_path / kind
            ).start_async()
            try:
                outcome = await asyncio.wait_for(job.wait_async(), timeout=30)
                assert outcome.state == "completed", outcome
                assert sink.tables == []
                assert _window_metrics(job.status())["null_event_time_rows"] == 0
            finally:
                if job.status()["state"] == "running":
                    await job.cancel_async()

    asyncio.run(exercise())


def _state_checksum(manifest: dict[str, object]) -> str:
    state = {key: manifest[key] for key in ("operators", "sinks", "sources")}
    if manifest.get("static_inputs"):
        state["static_inputs"] = manifest["static_inputs"]
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _corrupt_checkpoint(root: Path, fault: str) -> None:
    path, manifest = _manifest(root)
    assert manifest["state_checksum"] == _state_checksum(manifest)
    owners = [entry for entry in manifest["operators"].values() if entry["segments"]]
    assert len(owners) == 1
    entry = owners[0]
    if fault == "segment":
        relative = entry["segments"][0]["relative_path"]
        candidates = [
            candidate
            for candidate in root.rglob(Path(relative).name)
            if candidate.is_file()
        ]
        assert len(candidates) == 1
        original = candidates[0].read_bytes()
        assert original
        candidates[0].write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
        return
    if fault == "pipeline_fingerprint":
        manifest["pipeline_fingerprint"] = "0" * 64
    else:
        entry["inline_metadata"][fault] = (
            999 if fault == "state_layout_version" else "0" * 64
        )
        manifest["state_checksum"] = _state_checksum(manifest)
    path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


@pytest.mark.parametrize("kind", ["native", "symbolic"])
@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("state_layout_version", "streaming runtime initialization failed"),
        ("configuration_hash", "streaming runtime initialization failed"),
        ("state_schema_fingerprint", "streaming runtime initialization failed"),
        ("pipeline_fingerprint", "checkpoint lineage contains invalid recovery data"),
        ("segment", "checkpoint lineage contains invalid recovery data"),
    ],
)
def test_corrupt_restore_rejects_before_source_open(
    tmp_path: Path, kind: str, fault: str, message: str
) -> None:
    events = [("data", _trades()), ("watermark", 120)]
    opened: list[int] = []

    async def exercise() -> None:
        await _checkpoint_pending(_compile(kind), events, tmp_path, opened, pause_at=1)
        await asyncio.to_thread(_corrupt_checkpoint, tmp_path, fault)
        plan = _compile(kind)
        runner = _runner(
            plan,
            _ScriptedSource(events, opened=opened),
            _one_sink(plan, _CollectSink()),
            tmp_path,
        )
        with pytest.raises(StreamingRuntimeError, match=message):
            await runner.start_async()

    asyncio.run(exercise())
    assert opened == [0]


@pytest.mark.parametrize("kind", ["native", "symbolic"])
def test_changed_window_geometry_rejects_recovery_before_source_open(
    tmp_path: Path, kind: str
) -> None:
    events = [("data", _trades()), ("watermark", 120)]
    opened: list[int] = []

    async def exercise() -> None:
        await _checkpoint_pending(_compile(kind), events, tmp_path, opened, pause_at=1)
        plan = _compile(kind, hopping=True)
        runner = _runner(
            plan,
            _ScriptedSource(events, opened=opened),
            _one_sink(plan, _CollectSink()),
            tmp_path,
        )
        with pytest.raises(StreamingRuntimeError, match="checkpoint lineage"):
            await runner.start_async()

    asyncio.run(exercise())
    assert opened == [0]


def test_cached_plan_starts_independent_window_state_in_two_jobs(
    tmp_path: Path,
) -> None:
    runtime = Runtime()

    async def exercise() -> None:
        first = _program().compile_stream(runtime)
        second = _program().compile_stream(runtime)
        assert first.fingerprint == second.fingerprint
        first_events = [("data", _trades().slice(0, 1))]
        second_events = [("data", _trades().slice(4, 2))]
        first_result, second_result = await asyncio.gather(
            _complete(first, first_events, tmp_path / "first"),
            _complete(second, second_events, tmp_path / "second"),
        )
        assert first_result[0]["trade_count"].to_pylist() == [1]
        assert first_result[0]["volume"].to_pylist() == [10]
        assert second_result[0]["trade_count"].to_pylist() == [2]
        assert second_result[0]["volume"].to_pylist() == [20]

    asyncio.run(exercise())


def _portable_aggregate_functions(arrow_type: pa.DataType, ordered: bool) -> list[str]:
    functions = ["count"]
    if ordered:
        functions.extend(["min", "max"])
    if pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type):
        functions.extend(["sum", "avg"])
    return functions


@pytest.mark.parametrize(
    ("event_type", "arrow_event_type"),
    [
        ("timestamp[ms]", pa.timestamp("ms")),
        ("timestamp[us]", pa.timestamp("us")),
        ("timestamp[us, UTC]", pa.timestamp("us", tz="UTC")),
    ],
)
def test_full_portable_type_matrix_and_null_groups_match_native(
    tmp_path: Path, event_type: str, arrow_event_type: pa.DataType
) -> None:
    types = [
        ("boolean", "bool", pa.bool_(), [True, None, False, None]),
        ("signed8", "int8", pa.int8(), [-2, None, 3, None]),
        ("signed16", "int16", pa.int16(), [-2, None, 3, None]),
        ("signed32", "int32", pa.int32(), [-2, None, 3, None]),
        ("signed64", "int64", pa.int64(), [-2, None, 3, None]),
        ("unsigned8", "uint8", pa.uint8(), [2, None, 3, None]),
        ("unsigned16", "uint16", pa.uint16(), [2, None, 3, None]),
        ("unsigned32", "uint32", pa.uint32(), [2, None, 3, None]),
        ("unsigned64", "uint64", pa.uint64(), [2, None, 3, None]),
        ("float32", "float32", pa.float32(), [-2.5, None, 3.5, None]),
        ("float64", "float64", pa.float64(), [-2.5, None, 3.5, None]),
        ("string", "string", pa.string(), ["b", None, "a", None]),
        ("large_string", "large_string", pa.large_string(), ["b", None, "a", None]),
        ("date32", "date32", pa.date32(), [1, None, 2, None]),
        ("date64", "date64", pa.date64(), [86_400_000, None, 172_800_000, None]),
        ("timestamp_ms", "timestamp[ms]", pa.timestamp("ms"), [1, None, 2, None]),
        ("timestamp_us", "timestamp[us]", pa.timestamp("us"), [1, None, 2, None]),
        (
            "timestamp_utc",
            "timestamp[us, UTC]",
            pa.timestamp("us", tz="UTC"),
            [1, None, 2, None],
        ),
        ("time32", "time32[s]", pa.time32("s"), [1, None, 2, None]),
        ("time64", "time64[us]", pa.time64("us"), [1, None, 2, None]),
    ]
    fields = [Field("ts", event_type)]
    arrays = {
        "ts": pa.array(
            [BASE + timedelta(seconds=seconds) for seconds in [5, 15, 35, 20]],
            type=arrow_event_type,
        )
    }
    group_by = []
    aggregates = []
    native_aggregates = []
    for name, dtype, arrow_type, values in types:
        fields.append(Field(name, dtype))
        arrays[name] = pa.array(values, type=arrow_type)
        ordered = dtype not in {"timestamp[ms]", "time32[s]", "time64[us]"}
        if ordered:
            group_name = f"group_{name}"
            group_by.append(group_name)
            fields.append(Field(group_name, dtype))
            arrays[group_name] = pa.array([values[0]] * 3 + [None], type=arrow_type)
        for function in _portable_aggregate_functions(arrow_type, ordered):
            output = f"{name}_{function}"
            aggregates.append(getattr(window, function)(name, output=output))
            native_aggregates.append(
                {"function": function, "column": name, "output": output}
            )
    source = table_input("all_types", schema=fields)
    result = window.tumbling(
        source,
        event_time="ts",
        size_micros=MINUTE,
        group_by=group_by,
        aggregates=aggregates,
    )
    symbolic_plan = Program(
        "window-types", inputs=[source], outputs=[("result", result)]
    ).compile_stream(Runtime())
    document = _native_document()
    node = document["graph"]["nodes"][0]
    node["input_ports"][0]["schema"] = [
        {"name": field.name, "data_type": field.data_type, "nullable": field.nullable}
        for field in fields
    ]
    node["operator"]["spec"]["group_by"] = group_by
    node["operator"]["spec"]["aggregates"] = native_aggregates
    native_plan = PipelineBuilder._from_json(json.dumps(document)).compile_stream(
        runtime=Runtime()
    )
    events = [("data", pa.table(arrays))]

    async def exercise() -> None:
        expected, _ = await _complete(native_plan, events, tmp_path / "native")
        actual, _ = await _complete(symbolic_plan, events, tmp_path / "symbolic")
        _assert_tables_equal(actual, expected)
        assert actual.num_rows == 2
        for index, group in enumerate(actual["group_boolean"].to_pylist()):
            for name, _, _, _ in types:
                assert actual[f"{name}_count"][index].as_py() == (
                    0 if group is None else 2
                )
            if group is None:
                assert all(
                    actual[aggregate["output"]][index].as_py() is None
                    for aggregate in native_aggregates
                    if aggregate["function"] != "count"
                )

    asyncio.run(exercise())


def test_nan_signed_zero_group_order_and_epoch_boundaries_match_native(
    tmp_path: Path,
) -> None:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    source = table_input(
        "special",
        schema=[
            Field("ts", "timestamp[us, UTC]"),
            Field("symbol", "float64"),
            Field("price", "float64"),
        ],
    )
    result = window.hopping(
        source,
        event_time="ts",
        size_micros=2 * MINUTE,
        slide_micros=MINUTE,
        group_by=["symbol"],
        aggregates=[
            window.count("price", output="count"),
            window.min("price", output="low"),
            window.max("price", output="high"),
            window.avg("price", output="avg"),
        ],
    )
    symbolic_plan = Program(
        "special-windows", inputs=[source], outputs=[("result", result)]
    ).compile_stream(Runtime())
    document = _native_document(hopping=True)
    node = document["graph"]["nodes"][0]
    node["input_ports"][0]["schema"] = [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": True},
        {"name": "symbol", "data_type": "float64", "nullable": True},
        {"name": "price", "data_type": "float64", "nullable": True},
    ]
    node["operator"]["spec"]["aggregates"] = [
        {"function": "count", "column": "price", "output": "count"},
        {"function": "min", "column": "price", "output": "low"},
        {"function": "max", "column": "price", "output": "high"},
        {"function": "avg", "column": "price", "output": "avg"},
    ]
    native_plan = PipelineBuilder._from_json(json.dumps(document)).compile_stream(
        runtime=Runtime()
    )
    value = pa.table(
        {
            "ts": pa.array(
                [
                    epoch - timedelta(microseconds=1),
                    epoch,
                    epoch + timedelta(seconds=60),
                    epoch,
                    epoch,
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "symbol": [-0.0, 0.0, float("nan"), float("nan"), None],
            "price": [-0.0, 0.0, float("nan"), 2.0, None],
        }
    )

    async def exercise() -> None:
        expected, _ = await _complete(
            native_plan, [("data", value)], tmp_path / "native"
        )
        actual, _ = await _complete(
            symbolic_plan, [("data", value)], tmp_path / "symbolic"
        )
        _assert_tables_equal(actual, expected)
        assert min(actual["window_start"].cast(pa.int64()).to_pylist()) == -2 * MINUTE
        assert max(actual["window_end"].cast(pa.int64()).to_pylist()) == 3 * MINUTE
        assert sum(actual["count"].to_pylist()) == 8

    asyncio.run(exercise())
