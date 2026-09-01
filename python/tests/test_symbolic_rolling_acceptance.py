"""Independent acceptance vectors for SCE-06 Python lowering.

Authored by cf-tester during the independent verification of PR #198 and
absorbed into the shipped suite: non-goal rejections, lowering shape, batch
execution values, and caller-input immutability with independently chosen
fixtures and an independently derived reference model.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    Cursor,
    Data,
    DisabledWatermarks,
    Idle,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    ReplayPositioning,
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    StreamingRunner,
)
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    cs,
    duration,
    exact_time,
    rows,
    table_input,
    ts,
)
from calc_flow.symbolic import row as row_ops
from calc_flow.symbolic.lower import lower_program_document


def _ordered() -> object:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64"),
            Field("v", "int64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def _program(features: list[tuple[str, object]]) -> Program:
    quotes = _ordered()
    signals = quotes.with_columns(FeatureSet(features))
    return Program("p", inputs=[quotes], outputs=[("signals", signals)])


def test_min_max_covariance_and_correlation_execute_with_reference_values() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("floor", ts.min(quotes["x"], window=rows(3))),
            ("peak", ts.max(quotes["x"], window=rows(3))),
            ("cov", ts.covariance(quotes["x"], quotes["v"], window=rows(3), ddof=1)),
            ("corr", ts.correlation(quotes["x"], quotes["v"], window=rows(3))),
        ]
    )
    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_probe_table())})
    table = result.outputs["output"].to_pyarrow()
    # Canonical order is (event_time, entity, sequence): a10, b10, a11,
    # b11, a12s3, a12s4, b13, a14, b15, a16. Null and NaN samples are
    # excluded; covariance counts pairwise-valid positions only.
    assert table.column("floor").to_pylist() == [
        1.0,
        10.0,
        1.0,
        10.0,
        1.0,
        3.0,
        10.0,
        3.0,
        11.0,
        3.5,
    ]
    assert table.column("peak").to_pylist() == [
        1.0,
        10.0,
        1.0,
        11.0,
        3.0,
        3.5,
        13.0,
        3.5,
        14.0,
        6.0,
    ]
    covariances = table.column("cov").to_pylist()
    expected_cov = [None, None, None, -50.0, 20.0, 2.5, -50.0, 2.5, -150.0, 25.0]
    for actual, expected in zip(covariances, expected_cov, strict=True):
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    correlations = table.column("corr").to_pylist()
    expected_corr = [None, None, None, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    for actual, expected in zip(correlations, expected_corr, strict=True):
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_duration_frames_execute_and_correlation_compiles_stream() -> None:
    quotes = _ordered()
    program = _program([("m", ts.mean(quotes["x"], window=duration(10_000)))])
    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_probe_table())})
    table = result.outputs["output"].to_pyarrow()
    # Windows (t - 10ms, t] in canonical (time, entity, sequence) order:
    # a10, b10, a11, b11, a12s3, a12s4, b13, a14, b15, a16.
    means = table.column("m").to_pylist()
    expected = [1.0, 10.0, 1.0, 10.5, 2.0, 2.5, 34.0 / 3.0, 2.5, 12.0, 3.375]
    for actual, reference in zip(means, expected, strict=True):
        assert actual == pytest.approx(reference, rel=1e-12, abs=1e-12)

    pair_program = _program(
        [
            (
                "c",
                ts.correlation(
                    quotes["x"], quotes["v"], window=duration(10_000), ddof=0
                ),
            )
        ]
    )
    pair_program.compile_stream(Runtime())


def test_cross_section_winsorize_lowers_in_both_modes() -> None:
    quotes = _ordered()
    primitive = cs.winsorize(
        quotes["x"], group=exact_time(quotes["ts"]), lower=0.1, upper=0.9
    )
    for mode in ("batch", "stream"):
        document = lower_program_document(
            _program([("feature", primitive)]), Runtime(), mode
        )
        kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
        assert "cross_section" in kinds, (mode, kinds)


def test_cross_section_primitives_lower_in_both_modes() -> None:
    quotes = _ordered()
    for primitive in (
        cs.rank(quotes["x"], group=exact_time(quotes["ts"])),
        cs.percentile(quotes["x"], group=exact_time(quotes["ts"])),
        cs.zscore(quotes["x"], group=exact_time(quotes["ts"])),
        cs.demean(quotes["x"], group=exact_time(quotes["ts"])),
        cs.winsorize(quotes["x"], group=exact_time(quotes["ts"]), lower=0.1, upper=0.9),
        cs.top(quotes["x"], group=exact_time(quotes["ts"]), count=2),
        cs.bottom(quotes["x"], group=exact_time(quotes["ts"]), count=2),
        cs.mean_fill(quotes["x"], group=exact_time(quotes["ts"])),
    ):
        for mode in ("batch", "stream"):
            document = lower_program_document(
                _program([("feature", primitive)]), Runtime(), mode
            )
            kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
            assert "cross_section" in kinds, (mode, kinds)


def test_composite_lag_delta_arguments_compile_through_materialization() -> None:
    quotes = _ordered()
    for feature in (
        ("prev", ts.lag(quotes["x"] + 1.0)),
        ("change", ts.delta(quotes["x"] * 2.0)),
    ):
        _program([feature]).compile_batch(Runtime())

    derived = quotes.with_columns(FeatureSet([("y", quotes["x"] + 1.0)]))
    signals = derived.with_columns(FeatureSet([("prev", ts.lag(derived["y"]))]))
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])
    program.compile_batch(Runtime())


def test_probe_lowering_shape_with_periods_and_lateness() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("prev3", ts.lag(quotes["x"], periods=3)),
            ("dvol2", ts.delta(quotes["v"], periods=2)),
        ]
    )

    document = lower_program_document(
        program,
        Runtime(),
        "stream",
        allowed_lateness_micros=9,
        late_policy="drop",
    )

    rolling = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    ]
    assert len(rolling) == 1
    assert rolling[0]["operator"]["spec"] == {
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["seq"],
        "outputs": [
            {
                "kind": "lag",
                "primitive_version": 1,
                "input": "x",
                "output": "prev3",
                "periods": 3,
            },
            {
                "kind": "delta",
                "primitive_version": 1,
                "input": "v",
                "output": "dvol2",
                "periods": 2,
            },
        ],
        "allowed_lateness_micros": 9,
        "late_policy": {"kind": "drop", "metrics_version": 1},
        "value_policy": "stateful_numeric_v1",
    }


def test_lowering_does_not_mutate_the_program() -> None:
    quotes = _ordered()
    program = _program([("prev", ts.lag(quotes["x"]))])

    first = lower_program_document(program, Runtime(), "batch")
    second = lower_program_document(program, Runtime(), "batch")

    assert first == second


_PROBE_ROWS = [
    (12, "a", 3, 3.0, 120),
    (10, "a", 1, 1.0, 100),
    (15, "b", 4, 14.0, 800),
    (11, "a", 2, None, 110),
    (10, "b", 1, 10.0, 1000),
    (14, "a", 5, float("nan"), 140),
    (12, "a", 4, 3.5, 130),
    (13, "b", 3, 13.0, None),
    (11, "b", 2, 11.0, 900),
    (16, "a", 6, 6.0, 150),
]


# The independent reference oracle keeps per-entity lag/delta derivation in
# one flat loop for line-by-line auditability against the engine.
def _reference(rows_data, lag_periods, delta_periods):
    # #lizard forgives
    ordered = sorted(rows_data, key=lambda row: (row[0], row[1], row[2]))
    by_entity: dict[str, list] = {}
    for row in ordered:
        by_entity.setdefault(row[1], []).append(row)
    lag_of: dict[int, object] = {}
    delta_of: dict[int, object] = {}
    for entity_rows in by_entity.values():
        for index, row in enumerate(entity_rows):
            lag_of[id(row)] = (
                entity_rows[index - lag_periods][3] if index >= lag_periods else None
            )
            if index < delta_periods:
                delta_of[id(row)] = None
                continue
            reference = entity_rows[index - delta_periods][4]
            current = row[4]
            delta_of[id(row)] = (
                None if current is None or reference is None else current - reference
            )
    return [
        (row[0], row[1], row[2], lag_of[id(row)], delta_of[id(row)]) for row in ordered
    ]


def _table_from_rows(rows_data) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("x", pa.float64()),
            pa.field("v", pa.int64()),
        ]
    )
    return pa.table(
        {
            "ts": pa.array(
                [row[0] for row in rows_data], type=pa.timestamp("us", tz="UTC")
            ),
            "symbol": [row[1] for row in rows_data],
            "seq": pa.array([row[2] for row in rows_data], type=pa.uint64()),
            "x": pa.array([row[3] for row in rows_data], type=pa.float64()),
            "v": pa.array([row[4] for row in rows_data], type=pa.int64()),
        },
        schema=schema,
    )


def _probe_table() -> pa.Table:
    return _table_from_rows(_PROBE_ROWS)


def _table_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _assert_values_match(actual, expected) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got[:3] == want[:3]
        for column in (3, 4):
            if want[column] is None:
                assert got[column] is None
            elif isinstance(want[column], float) and math.isnan(want[column]):
                assert isinstance(got[column], float) and math.isnan(got[column])
            else:
                assert got[column] == want[column]


def test_batch_execution_matches_independent_reference_and_preserves_input() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("prev3", ts.lag(quotes["x"], periods=3)),
            ("dvol2", ts.delta(quotes["v"], periods=2)),
        ]
    )
    table = _probe_table()
    before = _table_bytes(table)

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(table)})

    output = result.outputs["output"].to_pyarrow()
    assert output.schema.names == ["ts", "symbol", "seq", "x", "v", "prev3", "dvol2"]
    ts_micros = output.column("ts").cast(pa.int64()).to_pylist()
    columns = output.drop_columns(["ts"]).to_pydict()
    actual = list(
        zip(
            ts_micros,
            columns["symbol"],
            columns["seq"],
            columns["prev3"],
            columns["dvol2"],
            strict=True,
        )
    )
    _assert_values_match(actual, _reference(_PROBE_ROWS, 3, 2))
    assert _table_bytes(table) == before, "caller-owned input table was mutated"


def _window_samples(window: list) -> list:
    return [
        item[3] for item in window if item[3] is not None and not math.isnan(item[3])
    ]


def _reference_count(valid: list):
    return len(valid) if len(valid) >= 2 else None


def _reference_sum(valid: list):
    return sum(valid) if valid else None


def _count_exact(values: list, target: float) -> int:
    return sum(1 for value in values if value == target)


def _reference_mean(valid: list):
    # Frozen ±inf readout (SCE-07 defect 1 ruling): both signs is the
    # undefined inf − inf (NaN), one sign is that infinity, none is the
    # finite average.
    if not valid:
        return None
    pos_inf = _count_exact(valid, math.inf)
    neg_inf = _count_exact(valid, -math.inf)
    if pos_inf and neg_inf:
        return math.nan
    if pos_inf:
        return math.inf
    if neg_inf:
        return -math.inf
    return sum(valid) / len(valid)


def _reference_spread(valid: list):
    # Variance/stddev after the null gates; any inf window pins to NaN.
    count = len(valid)
    if count < 2:
        return None, None
    if any(math.isinf(value) for value in valid):
        return math.nan, math.nan
    mean = sum(valid) / count
    squared = sum((value - mean) ** 2 for value in valid)
    return squared / (count - 1), math.sqrt(squared / count)


def _aggregate_reference(rows_data, size):
    # Independent oracle with the frozen ±inf readout semantics (SCE-07
    # defect 1 ruling): sum is the naive IEEE fold, mean classifies from the
    # ±inf counts, and variance/stddev over any inf window is NaN after the
    # null gates.
    ordered = sorted(rows_data, key=lambda row: (row[0], row[1], row[2]))
    by_entity: dict[str, list] = {}
    for row in ordered:
        by_entity.setdefault(row[1], []).append(row)
    results: dict[int, tuple] = {}
    for entity_rows in by_entity.values():
        window: list = []
        for row in entity_rows:
            window.append(row)
            if len(window) > size:
                window.pop(0)
            valid = _window_samples(window)
            variance, stddev = _reference_spread(valid)
            results[id(row)] = (
                _reference_count(valid),
                _reference_sum(valid),
                _reference_mean(valid),
                variance,
                stddev,
            )
    return [(row[0], row[1], row[2], *results[id(row)]) for row in ordered]


def _aggregate_program() -> Program:
    quotes = _ordered()
    return _program(
        [
            ("n3", ts.count(quotes["x"], window=rows(3), min_periods=2)),
            ("total3", ts.sum(quotes["x"], window=rows(3))),
            ("avg3", ts.mean(quotes["x"], window=rows(3))),
            ("var3", ts.variance(quotes["x"], window=rows(3), min_periods=2, ddof=1)),
            ("std3", ts.stddev(quotes["x"], window=rows(3), min_periods=2, ddof=0)),
        ]
    )


def _aggregate_columns(output: pa.Table) -> list[tuple]:
    ts_micros = output.column("ts").cast(pa.int64()).to_pylist()
    columns = output.drop_columns(["ts"]).to_pydict()
    return list(
        zip(
            ts_micros,
            columns["symbol"],
            columns["seq"],
            columns["n3"],
            columns["total3"],
            columns["avg3"],
            columns["var3"],
            columns["std3"],
            strict=True,
        )
    )


def _assert_classified_value(got, wanted, rtol: float) -> None:
    # D13 semantics: null matches null exactly, NaN only NaN, infinities by
    # sign, and everything else within the per-column tolerance.
    if wanted is None:
        assert got is None
    elif isinstance(wanted, float) and math.isnan(wanted):
        assert isinstance(got, float) and math.isnan(got)
    elif isinstance(wanted, float) and math.isinf(wanted):
        assert got == wanted
    else:
        assert got == pytest.approx(wanted, rel=rtol, abs=1e-12)


def _assert_aggregates_match(actual, expected) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got[:3] == want[:3]
        assert got[3] == want[3]
        for column, rtol in ((4, 1e-12), (5, 1e-12), (6, 1e-10), (7, 1e-10)):
            _assert_classified_value(got[column], want[column], rtol)


def test_aggregate_batch_execution_matches_independent_reference() -> None:
    table = _probe_table()
    before = _table_bytes(table)

    plan = _aggregate_program().compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(table)})

    output = result.outputs["output"].to_pyarrow()
    assert output.schema.names == [
        "ts",
        "symbol",
        "seq",
        "x",
        "v",
        "n3",
        "total3",
        "avg3",
        "var3",
        "std3",
    ]
    _assert_aggregates_match(
        _aggregate_columns(output), _aggregate_reference(_PROBE_ROWS, 3)
    )
    assert _table_bytes(table) == before, "caller-owned input table was mutated"


class _SegmentedSource:
    def __init__(self, table: pa.Table, segment: int) -> None:
        self._table = table
        self._segment = segment
        self._offset = 0

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=10_000,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0

    async def next(self) -> Data | None:
        if self._offset >= self._table.num_rows:
            return None
        end = min(self._offset + self._segment, self._table.num_rows)
        chunk = self._table.slice(self._offset, end - self._offset)
        self._offset = end
        order = end.to_bytes(8, "big")
        return Data(Batch.from_pyarrow(chunk), Cursor(order, {"offset": end}))

    async def close(self) -> None:
        return None


class _CollectSink:
    def __init__(self) -> None:
        self.tables: list[pa.Table] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        table = batch.to_pyarrow()
        if table.num_rows:
            self.tables.append(table)

    async def close(self) -> None:
        return None


class _ReplayPauseSource:
    def __init__(
        self,
        table: pa.Table,
        pause_at: int | None,
        opened_offsets: list[int],
    ) -> None:
        self._table = table
        self._pause_at = pause_at
        self._opened_offsets = opened_offsets
        self._offset = 0
        self.paused = asyncio.Event()

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=1,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0 if cursor is None else int(cursor.payload["offset"])
        self._opened_offsets.append(self._offset)

    async def next(self) -> Data | Idle | None:
        if self._pause_at == self._offset:
            self.paused.set()
            await asyncio.sleep(0)
            return Idle()
        if self._offset >= self._table.num_rows:
            return None
        chunk = self._table.slice(self._offset, 1)
        self._offset += 1
        return Data(
            Batch.from_pyarrow(chunk),
            Cursor(
                self._offset.to_bytes(8, "big"),
                {"offset": self._offset},
            ),
        )

    async def close(self) -> None:
        return None


def _multi_stage_program() -> Program:
    quotes = _ordered()
    change = ts.delta(quotes["x"])
    gain = row_ops.clip(change, lower=0.0, upper=1.0e100)
    macd = ts.macd(quotes["x"], fast_span=2, slow_span=4)
    return _program(
        [
            ("change", change),
            ("average_gain", ts.mean(gain, window=rows(3))),
            ("ema_3", ts.ewma(quotes["x"], span=3, min_periods=2)),
            ("macd_2_4", macd),
            ("ema_macd_3", ts.ema(macd, span=3)),
        ]
    )


def test_multi_stage_symbolic_checkpoint_recovery_matches_batch(tmp_path: Path) -> None:
    table = _probe_table()
    expected = (
        _multi_stage_program()
        .compile_batch(Runtime())
        .execute({"input": Batch.from_pyarrow(table)})
        .outputs["output"]
        .to_pyarrow()
    )
    sink = _CollectSink()
    opened_offsets: list[int] = []

    async def runner(source: _ReplayPauseSource) -> StreamingRunner:
        return StreamingRunner(
            _multi_stage_program().compile_stream(Runtime()),
            {
                "input": SourceBinding(
                    source,
                    watermark_policy=DisabledWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        )

    async def exercise() -> None:
        first_source = _ReplayPauseSource(table, 5, opened_offsets)
        first = await (await runner(first_source)).start_async()
        await asyncio.wait_for(first_source.paused.wait(), timeout=2)
        assert await first.trigger_checkpoint_async() == 1
        assert (await first.cancel_async()).state == "cancelled"

        second_source = _ReplayPauseSource(table, None, opened_offsets)
        second = await (await runner(second_source)).start_async()
        assert (await second.wait_async()).state == "completed"

    asyncio.run(exercise())

    recovered = pa.concat_tables(sink.tables)
    assert recovered.schema == expected.schema
    for field in recovered.schema:
        if pa.types.is_floating(field.type):
            actual = recovered[field.name].to_pylist()
            reference = expected[field.name].to_pylist()
            for observed, wanted in zip(actual, reference, strict=True):
                _assert_classified_value(observed, wanted, 1e-12)
        else:
            assert recovered[field.name].equals(expected[field.name])
    assert opened_offsets == [0, 5]


@pytest.mark.parametrize("segmentation", (1, 3, 1000))
def test_aggregate_stream_matches_batch_across_segmentation(
    tmp_path: Path, segmentation: int
) -> None:
    plan = _aggregate_program().compile_stream(Runtime())
    sink = _CollectSink()

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    _SegmentedSource(_probe_table(), segmentation),
                    watermark_policy=DisabledWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        outcome = await job.wait_async()
        assert outcome.state == "completed"

    asyncio.run(exercise())

    stream_output = pa.concat_tables(sink.tables)
    batch_result = (
        _aggregate_program()
        .compile_batch(Runtime())
        .execute({"input": Batch.from_pyarrow(_probe_table())})
    )
    batch_output = batch_result.outputs["output"].to_pyarrow()
    _assert_aggregates_match(
        _aggregate_columns(stream_output), _aggregate_columns(batch_output)
    )


# Infinities placed at differing positions inside identical multisets, plus
# mixed-sign, NaN, and null windows (SCE-07 defect 1 ruling, B3).
_INF_PROBE_ROWS = [
    (10, "a", 1, math.inf, 100),
    (11, "a", 2, 1.0, 110),
    (12, "a", 3, 2.0, 120),
    (10, "b", 1, 1.0, 200),
    (11, "b", 2, 2.0, 210),
    (12, "b", 3, math.inf, 220),
    (10, "c", 1, 1.0, 300),
    (11, "c", 2, math.inf, 310),
    (12, "c", 3, 2.0, 320),
    (10, "d", 1, math.inf, 400),
    (11, "d", 2, -math.inf, 410),
    (12, "d", 3, 5.0, 420),
    (10, "e", 1, math.inf, 500),
    (11, "e", 2, float("nan"), 510),
    (12, "e", 3, None, 520),
]


def test_aggregate_inf_windows_match_the_frozen_reference() -> None:
    table = _table_from_rows(_INF_PROBE_ROWS)
    before = _table_bytes(table)

    plan = _aggregate_program().compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(table)})

    output = result.outputs["output"].to_pyarrow()
    _assert_aggregates_match(
        _aggregate_columns(output), _aggregate_reference(_INF_PROBE_ROWS, 3)
    )
    assert _table_bytes(table) == before, "caller-owned input table was mutated"


@pytest.mark.parametrize("segmentation", (1, 2, 1000))
def test_aggregate_inf_stream_matches_batch_across_segmentation(
    tmp_path: Path, segmentation: int
) -> None:
    table = _table_from_rows(_INF_PROBE_ROWS)
    plan = _aggregate_program().compile_stream(Runtime())
    sink = _CollectSink()

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    _SegmentedSource(table, segmentation),
                    watermark_policy=DisabledWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        outcome = await job.wait_async()
        assert outcome.state == "completed"

    asyncio.run(exercise())

    stream_output = pa.concat_tables(sink.tables)
    batch_result = (
        _aggregate_program()
        .compile_batch(Runtime())
        .execute({"input": Batch.from_pyarrow(table)})
    )
    batch_output = batch_result.outputs["output"].to_pyarrow()
    _assert_aggregates_match(
        _aggregate_columns(stream_output), _aggregate_columns(batch_output)
    )
