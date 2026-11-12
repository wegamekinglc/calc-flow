from __future__ import annotations

import asyncio
from collections import Counter
from datetime import timedelta

import pyarrow as pa
import pytest

import calc_flow as cf

QUERIES = {
    "limited": """SELECT * FROM CF_INNER_JOIN('a', 'b', '{
        "spec": {"join_type": "inner", "left_keys": ["k"], "right_keys": ["k"],
            "left_event_time": "ts", "right_event_time": "ts",
            "bounds": {"before_micros": 2, "after_micros": 3},
            "limits": {"max_state_rows_per_side": 100,
                "max_state_bytes_per_side": 1048576,
                "max_matches_per_input_batch": 3}}
    }')""",
    "closed": """SELECT * FROM CF_INNER_JOIN('a', 'b', '{
        "spec": {"join_type": "inner", "left_keys": ["k"], "right_keys": ["k"],
            "left_event_time": "ts", "right_event_time": "ts",
            "bounds": {"before_micros": 2, "after_micros": 3},
            "limits": {"max_state_rows_per_side": 100,
                "max_state_bytes_per_side": 1048576,
                "max_matches_per_input_batch": 1000}}
    }')""",
    "zero": """SELECT * FROM CF_INNER_JOIN('a', 'b', '{
        "spec": {"join_type": "inner", "left_keys": ["k"], "right_keys": ["k"],
            "left_event_time": "ts", "right_event_time": "ts",
            "bounds": {"before_micros": 0, "after_micros": 0},
            "limits": {"max_state_rows_per_side": 100,
                "max_state_bytes_per_side": 1048576,
                "max_matches_per_input_batch": 1000}}
    }')""",
    "ordered": """SELECT * FROM CF_INNER_JOIN('a', 'b', '{
        "spec": {"join_type": "inner", "left_keys": ["k"], "right_keys": ["k"],
            "left_event_time": "ts", "right_event_time": "ts",
            "bounds": {"before_micros": 2, "after_micros": 3},
            "limits": {"max_state_rows_per_side": 100,
                "max_state_bytes_per_side": 1048576,
                "max_matches_per_input_batch": 1000}},
        "output_ordering": {"entity_by": ["left__k"], "event_time": "left__ts",
            "sequence_by": ["left__id", "right__id"]}
    }')""",
    "nested": """SELECT * FROM CF_INNER_JOIN('a', 'b', '{
        "spec": {"join_type": "inner", "left_keys": ["left__k"], "right_keys": ["k"],
            "left_event_time": "left__ts", "right_event_time": "ts",
            "bounds": {"before_micros": 0, "after_micros": 0},
            "limits": {"max_state_rows_per_side": 100,
                "max_state_bytes_per_side": 1048576,
                "max_matches_per_input_batch": 1000}}
    }')""",
}

SIX_PAIRS = [(1, 13), (1, 80), (1, 81), (2, 13), (2, 80), (2, 81)]


def _source(name, *, nullable_key=False, nullable_time=False):
    return cf.table_input(
        name,
        schema=[
            cf.Field("k", "int64", nullable=nullable_key),
            cf.Field("ts", "timestamp[us, UTC]", nullable=nullable_time),
            cf.Field("id", "uint64", nullable=False),
        ],
        entity_by=["id"] if nullable_key else ["k"],
        event_time="ts",
        sequence_by=["id"],
    )


def _values(rows, *, nullable_key=False):
    schema = pa.schema(
        [
            pa.field("k", pa.int64(), nullable_key),
            pa.field("ts", pa.timestamp("us", tz="UTC"), False),
            pa.field("id", pa.uint64(), False),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([row[i] for row in rows], type=field.type)
            for i, field in enumerate(schema)
        ],
        schema=schema,
    )
    return pa.Table.from_batches(table.to_batches(max_chunksize=2), schema=schema)


def _join(a, b, sql, kind="closed"):
    if sql:
        return cf.sql(QUERIES[kind], a=a, b=b)
    key = "left__k" if kind == "nested" else "k"
    time = "left__ts" if kind == "nested" else "ts"
    before, after = (2, 3) if kind in ("closed", "ordered", "limited") else (0, 0)
    ordering = (
        dict(
            output_entity_by=["left__k"],
            output_event_time="left__ts",
            output_sequence_by=["left__id", "right__id"],
        )
        if kind == "ordered"
        else {}
    )
    return cf.table.stream_join(
        a,
        b,
        left_keys=[key],
        right_keys=["k"],
        left_event_time=time,
        right_event_time="ts",
        bounds=cf.JoinTimeBounds(
            timedelta(microseconds=before), timedelta(microseconds=after)
        ),
        limits=cf.JoinStateLimits(100, 1048576, 3 if kind == "limited" else 1000),
        **ordering,
    )


async def _stream_rows(program, tables):
    opened, read = Counter(), Counter()
    rows = {name: [] for name, _ in program.outputs}

    async def ingress(name, table):
        opened[name] += 1
        for record in table.to_batches(max_chunksize=2):
            read[name] += 1
            yield pa.Table.from_batches([record])

    async with program.stream(
        {name: ingress(name, table) for name, table in tables.items()},
        watermarks={name: cf.DisabledWatermarks() for name in tables},
    ) as stream:
        async for event in stream:
            rows[event.name].extend(event.table.to_pylist())
    return rows, opened, read


def _run(program, tables, mode):
    if mode == "batch":
        return {
            name: table.to_pylist() for name, table in program.collect(tables).items()
        }
    return asyncio.run(_stream_rows(program, tables))[0]


@pytest.mark.parametrize("sql", [False, True], ids=["dataframe", "sql"])
@pytest.mark.parametrize("mode", ["batch", "stream"])
@pytest.mark.parametrize(
    "case",
    [
        (
            "closed",
            [(1, 10, 1), (1, 10, 2), (None, 10, 3), (2, 10, 4)],
            [
                (1, 7, 7),
                (1, 8, 80),
                (1, 8, 81),
                (None, 10, 90),
                (3, 10, 91),
                (1, 13, 13),
                (1, 14, 14),
            ],
            SIX_PAIRS,
        ),
        ("zero", [(1, 10, 1)], [(1, 9, 9), (1, 10, 10), (1, 11, 11)], [(1, 10)]),
        (
            "closed",
            [(1, -10, 1)],
            [(1, -13, 13), (1, -12, 12), (1, -7, 7), (1, -6, 6)],
            [(1, 7), (1, 12)],
        ),
    ],
    ids=["six_pairs_null_keys", "zero_bounds", "negative_time"],
)
def test_public_terminal_pairs_match_literal_oracle(sql, mode, case):
    kind, left, right, expected = case
    a, b = _source("a", nullable_key=True), _source("b", nullable_key=True)
    program = cf.Program("oracle", mode=mode, outputs={"pairs": _join(a, b, sql, kind)})
    tables = {
        "a": _values(left, nullable_key=True),
        "b": _values(right, nullable_key=True),
    }
    before = {name: table.to_pylist() for name, table in tables.items()}
    result = _run(program, tables, mode)["pairs"]
    assert sorted((row["left__id"], row["right__id"]) for row in result) == expected
    assert {name: table.to_pylist() for name, table in tables.items()} == before


def _cached_pairs(plan, left, right):
    assert plan.external_inputs == ("a.input", "b.input")
    result = plan.execute(
        inputs={
            "a.input": cf.Batch.from_pyarrow(left),
            "b.input": cf.Batch.from_pyarrow(right),
        }
    )
    assert len(result.outputs) == 1
    rows = next(iter(result.outputs.values())).to_pyarrow().to_pylist()
    return sorted((row["left__id"], row["right__id"]) for row in rows)


@pytest.mark.parametrize("sql", [False, True], ids=["dataframe", "sql"])
def test_cached_public_plan_rejects_complete_match_budget_and_reuses_without_reset(sql):
    program = cf.Program(
        "cached",
        mode="batch",
        outputs={"pairs": _join(_source("a"), _source("b"), sql, "limited")},
    )
    runtime = cf.Runtime()
    plan = program.compile_batch(runtime)
    left = _values([(1, 10, 1), (1, 10, 2)])
    assert _cached_pairs(plan, left, _values([(1, 8, 80)])) == [(1, 80), (2, 80)]
    right = _values([(1, 8, 80), (1, 13, 13)])
    right = pa.Table.from_batches(right.to_batches(max_chunksize=1))
    assert len(right.to_batches()) == 2
    with pytest.raises(cf.ExecutionError, match="match limit"):
        _cached_pairs(plan, left, right)
    assert _cached_pairs(plan, _values([]), right) == []
    assert _cached_pairs(plan, left, _values([])) == []
    assert _cached_pairs(plan, _values([]), _values([])) == []
    assert _cached_pairs(plan, _values([(1, 100, 500)]), _values([(1, 100, 600)])) == [
        (500, 600)
    ]
    assert program.compile_batch(runtime) is plan


def _nested_program(sql, mode):
    a, b, c = (_source(name) for name in ("a", "b", "c"))
    first = _join(a.sql("SELECT * FROM input"), b, sql, "ordered")
    nested = _join(first.sql("SELECT * FROM input"), c, sql, "nested")
    fan = first.select(lid=first["left__id"], rid=first["right__id"])
    return cf.Program(
        "owners",
        mode=mode,
        outputs={
            "first": first,
            "fan": fan,
            "nested": nested,
            "independent": _join(a, b, sql, "zero"),
            "outside": a.select(source_id=a["id"]),
        },
    )


def _assert_nested_rows(rows):
    assert sorted((r["left__id"], r["right__id"]) for r in rows["first"]) == SIX_PAIRS
    assert sorted((r["lid"], r["rid"]) for r in rows["fan"]) == SIX_PAIRS
    assert sorted(
        (r["left__left__id"], r["left__right__id"], r["right__id"])
        for r in rows["nested"]
    ) == [
        (1, 13, 100),
        (1, 13, 101),
        (1, 80, 100),
        (1, 80, 101),
        (1, 81, 100),
        (1, 81, 101),
        (2, 13, 100),
        (2, 13, 101),
        (2, 80, 100),
        (2, 80, 101),
        (2, 81, 100),
        (2, 81, 101),
    ]
    assert rows["independent"] == []
    assert sorted(row["source_id"] for row in rows["outside"]) == [1, 2]


@pytest.mark.parametrize("sql", [False, True], ids=["dataframe", "sql"])
def test_three_source_nested_fanout_has_exact_triples_and_single_source_reads(sql):
    program = _nested_program(sql, "batch")
    tables = {
        "a": _values([(1, 10, 1), (1, 10, 2)]),
        "b": _values([(1, 8, 80), (1, 8, 81), (1, 13, 13)]),
        "c": _values([(1, 10, 100), (1, 10, 101), (2, 10, 102)]),
    }
    joins = [
        node
        for node in program.to_project().model_dump()["graph"]["nodes"]
        if node["operator"]["kind"] == "stream_join"
    ]
    assert len(joins) == 3
    _assert_nested_rows(_run(program, tables, "batch"))
    streamed = program.for_mode("stream")
    rows, opened, read = asyncio.run(_stream_rows(streamed, tables))
    _assert_nested_rows(rows)
    assert opened == Counter(a=1, b=1, c=1)
    assert read == Counter(a=1, b=2, c=2)


@pytest.mark.parametrize("sql", [False, True], ids=["dataframe", "sql"])
@pytest.mark.parametrize("mode", ["batch", "stream"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_nullable_time_is_rejected_at_each_public_ingress(sql, mode, side):
    a = _source("a", nullable_time=side == "left")
    b = _source("b", nullable_time=side == "right")
    program = cf.Program("nullable", mode=mode, outputs={"pairs": _join(a, b, sql)})
    role = "sql.inner_join" if sql else "stream_join"
    expected_path = f"outputs.pairs.{role}.{side}_event_time"
    assert any(issue.path == expected_path for issue in program.analyze().issues)
    with pytest.raises(cf.CompileError, match="non-null"):
        if mode == "batch":
            program.compile_batch()
        else:
            program.compile_stream()
