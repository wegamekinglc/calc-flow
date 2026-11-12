from __future__ import annotations

import pyarrow as pa
import pytest

import calc_flow as cf


def source():
    return cf.table_input(
        "events",
        schema=pa.schema([("ts", pa.timestamp("us", tz="UTC")), ("q", pa.int64())]),
    )


def program(before, time, sql):
    if sql:
        queries = {
            "shifted": (
                "SELECT window_start, window_end, SUM(q) AS s FROM "
                "CF_TUMBLE('a', 'shifted', 10) GROUP BY window_start, window_end"
            ),
            "time": (
                "SELECT window_start, window_end, SUM(q) AS s FROM "
                "CF_TUMBLE('a', 'time', 10) GROUP BY window_start, window_end"
            ),
            "ts": (
                "SELECT window_start, window_end, SUM(q) AS s FROM "
                "CF_TUMBLE('a', 'ts', 10) GROUP BY window_start, window_end"
            ),
        }
        bars = cf.sql(queries[time], a=before)
    else:
        bars = cf.window.tumbling(
            before,
            event_time=time,
            size_micros=10,
            aggregates=[cf.window.sum("q", output="s")],
        )
    return cf.Program("proof", mode="batch", outputs={"bars": bars})


def eof():
    return cf.PortabilityRequest(
        comparison="final_relation",
        history="accepted_history",
        cutoff="all_sources_eof",
        ordering="unordered",
    )


def assert_rejected(candidate):
    for mode in ("batch", "stream"):
        current = candidate.for_mode(mode)
        compile_plan = (
            current.compile_batch if mode == "batch" else current.compile_stream
        )
        for request in (None, eof()):
            with pytest.raises(cf.CompileError, match="capability_mismatch"):
                compile_plan(portability=request)


@pytest.mark.parametrize("sql", [False, True])
@pytest.mark.parametrize("rename", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        "SELECT *, ts - INTERVAL '100 microseconds' AS shifted FROM input",
        "SELECT input.*, ts - INTERVAL '100 microseconds' AS shifted FROM input",
    ],
)
def test_wildcard_cannot_prove_computed_time_or_rename(query, rename, sql):
    before = source().sql(query)
    if rename:
        before = before.sql("SELECT shifted AS time, q FROM input")
    candidate = program(before, "time" if rename else "shifted", sql)
    assert_rejected(candidate)
    assert_rejected(candidate.with_update_semantics("latest_relation"))


@pytest.mark.parametrize("sql", [False, True])
def test_cte_cannot_shadow_registered_input_in_time_proof(sql):
    before = source().sql(
        "WITH input AS (SELECT ts - INTERVAL '100 microseconds' AS ts, q FROM input) "
        "SELECT ts, q FROM input"
    )
    candidate = program(before, "ts", sql)
    assert_rejected(candidate)
    assert_rejected(candidate.with_update_semantics("latest_relation"))


@pytest.mark.parametrize("mode", ["batch", "stream"])
def test_legacy_window_with_independent_sql_keeps_structured_rejection(mode):
    events = cf.table_input(
        "events",
        schema=pa.schema(
            [
                pa.field("ts", pa.timestamp("us", tz="UTC"), False),
                pa.field("g", pa.string(), False),
                pa.field("seq", pa.uint64(), False),
                pa.field("q", pa.int64()),
            ]
        ),
        event_time="ts",
        entity_by=["g"],
        sequence_by=["seq"],
    )
    legacy = cf.window.tumbling(events, event_time="ts", size_micros=10, group_by=["g"])
    identity = (legacy._node.digest, legacy._node.node_bytes)
    assert legacy._node.op.version == 1
    candidate = cf.Program(
        "legacy",
        mode=mode,
        outputs={"bars": legacy, "other": events.sql("SELECT q FROM input")},
    )
    assert not candidate.analyze().issues
    compile_plan = (
        candidate.compile_batch if mode == "batch" else candidate.compile_stream
    )
    with pytest.raises(cf.CompileError, match="unknown_primitive_version"):
        compile_plan()
    assert (legacy._node.digest, legacy._node.node_bytes) == identity


def assert_admitted(candidate):
    for mode in ("batch", "stream"):
        current = candidate.for_mode(mode)
        compile_plan = (
            current.compile_batch if mode == "batch" else current.compile_stream
        )
        for request in (None, eof()):
            compile_plan(portability=request)


@pytest.mark.parametrize("sql", [False, True])
def test_unquoted_time_rename_follows_native_case_rules(sql):
    candidate = program(source().sql("SELECT TS AS TIME, Q FROM INPUT"), "time", sql)
    assert_admitted(candidate)
    assert_admitted(candidate.with_update_semantics("latest_relation"))


def test_unquoted_window_fields_and_output_alias_follow_native_case_rules():
    bars = cf.sql(
        "SELECT WINDOW_START, WINDOW_END, COUNT(Q) AS N FROM CF_TUMBLE('a', 'ts', 10) "
        "GROUP BY WINDOW_START, WINDOW_END",
        a=source(),
    )
    candidate = cf.Program("case", mode="batch", outputs={"bars": bars})
    assert candidate.collect_schema()["bars"].names == [
        "window_start",
        "window_end",
        "n",
    ]
    assert_admitted(candidate)
    assert_admitted(candidate.with_update_semantics("latest_relation"))


@pytest.mark.parametrize("sql", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM input",
        "SELECT E.* FROM INPUT AS E",
        "SELECT *, q + 1 AS extra FROM input",
    ],
)
def test_real_wildcard_time_survives_unrelated_computation(query, sql):
    candidate = program(source().sql(query), "ts", sql)
    assert_admitted(candidate)
    assert_admitted(candidate.with_update_semantics("latest_relation"))


def test_quoted_identifiers_preserve_exact_field_and_alias_case():
    events = cf.table_input(
        "events",
        schema=pa.schema([("Ts", pa.timestamp("us", tz="UTC")), ("Q", pa.int64())]),
    )
    before = events.sql('SELECT "eX"."Ts" AS "TIME", "eX"."Q" FROM input AS "eX"')
    bars = cf.sql(
        'SELECT window_start, window_end, SUM("Q") AS "Total" '
        "FROM CF_TUMBLE('a', 'TIME', 10) GROUP BY window_start, window_end",
        a=before,
    )
    candidate = cf.Program("quoted", mode="batch", outputs={"bars": bars})
    assert candidate.collect_schema()["bars"].names == [
        "window_start",
        "window_end",
        "Total",
    ]
    assert_admitted(candidate)
    assert_admitted(candidate.with_update_semantics("latest_relation"))


def test_cte_remains_ordinary_sql_outside_window_proof():
    before = source().sql(
        "WITH input AS (SELECT ts - INTERVAL '100 microseconds' AS ts, q FROM input) "
        "SELECT ts, q FROM input"
    )
    data = pa.Table.from_arrays(
        [pa.array([100, 101], type=pa.timestamp("us", tz="UTC")), pa.array([1, 2])],
        names=["ts", "q"],
    )
    candidate = cf.Program("ordinary", mode="batch", outputs={"out": before})
    actual = candidate.collect({"events": data})["out"]
    assert actual["ts"].cast(pa.int64()).to_pylist() == [0, 1]
    assert actual["q"].to_pylist() == [1, 2]


@pytest.mark.parametrize("sql", [False, True])
def test_real_wildcard_control_keeps_source_watermark_and_eof_semantics(sql):
    import asyncio
    from datetime import datetime, timedelta, timezone

    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), False),
            pa.field("q", pa.int64(), False),
        ]
    )
    events = cf.table_input("events", schema=schema, event_time="ts")
    candidate = program(events.sql("SELECT * FROM input"), "ts", sql)
    data = pa.Table.from_arrays(
        [pa.array([100, 101], type=schema.field(0).type), pa.array([1, 2])],
        schema=schema,
    )

    async def feed():
        yield data.slice(0, 1)
        yield cf.Watermark(
            datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=99)
        )
        yield data.slice(1, 1)

    async def exercise():
        observed = []
        async with candidate.for_mode("stream").stream(
            {"events": feed()}, watermarks=cf.SourceProvidedWatermarks()
        ) as output:
            async for event in output:
                observed.extend(event.table["s"].to_pylist())
        finite = (await candidate.collect_async({"events": data}))["bars"][
            "s"
        ].to_pylist()
        assert finite == [3]
        assert observed == [3]

    asyncio.run(exercise())
