from __future__ import annotations

from datetime import timedelta

import pyarrow as pa
import pytest

import calc_flow as cf


def _source(name):
    return cf.table_input(
        name,
        schema=[
            cf.Field("key", "int64", nullable=False),
            cf.Field("time", "timestamp[us, UTC]", nullable=False),
            cf.Field("sequence", "uint64", nullable=False),
        ],
        entity_by=["key"],
        event_time="time",
        sequence_by=["sequence"],
    )


def _table(times):
    schema = pa.schema(
        [
            pa.field("key", pa.int64(), False),
            pa.field("time", pa.timestamp("us", tz="UTC"), False),
            pa.field("sequence", pa.uint64(), False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array([1] * len(times)),
            pa.array(times, type=schema.field("time").type),
            pa.array(range(len(times)), type=pa.uint64()),
        ],
        schema=schema,
    )


def _declaration(join_type="left_semi", **ordering):
    return cf.table.stream_join(
        _source("left"),
        _source("right"),
        join_type=join_type,
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="time",
        right_event_time="time",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=2), timedelta(microseconds=3)),
        limits=cf.JoinStateLimits(100, 1_000_000, 2),
        **ordering,
    )


def test_bounded_left_semi_collects_original_left_rows_and_exports_project():
    witnessed = _declaration()
    program = cf.Program("existence", mode="batch", outputs={"witnessed": witnessed})
    nodes = program.to_project().model_dump()["graph"]["nodes"]
    (join,) = [
        node["operator"] for node in nodes if node["operator"]["kind"] == "stream_join"
    ]
    assert join["spec"]["join_type"] == "left_semi"
    left = _table([10, 10, 20])
    result = program.collect({"left": left, "right": _table([8, 13, 13])})["witnessed"]
    assert result.equals(left.slice(0, 2), check_metadata=True)


@pytest.mark.parametrize("kind", ["left_semi", "left_anti"])
def test_existence_builder_serializes_nullable_operands(kind):
    schema = [
        cf.ArrowFieldSpec("key", "int64", True),
        cf.ArrowFieldSpec("time", "timestamp[us]", True),
    ]
    builder = cf.PipelineBuilder("advanced").stream_join(
        "exists",
        left_schema=schema,
        right_schema=schema,
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="time",
        right_event_time="time",
        bounds=cf.JoinTimeBounds(timedelta(0), timedelta(0)),
        limits=cf.JoinStateLimits(8, 65536, 1),
        join_type=kind,
    )
    assert builder.project["graph"]["nodes"][0]["operator"]["spec"]["join_type"] == kind


@pytest.mark.parametrize("kind", ["left_semi", "left_anti"])
def test_existence_output_identity_uses_only_declared_left_names(kind):
    witnessed = _declaration(
        kind,
        output_entity_by=["key"],
        output_event_time="time",
        output_sequence_by=["sequence"],
    )
    program = cf.Program("identity", mode="batch", outputs={"witnessed": witnessed})
    errors = program.analyze(cf.Runtime(), mode="batch").issues
    assert not errors, errors


def test_left_semi_public_stream_emits_each_distinct_left_identity_once():
    import asyncio

    async def exercise():
        program = cf.Program(
            "stream", mode="stream", outputs={"exists": _declaration()}
        )
        left = _table([10, 10, 20])
        async with program.stream(
            {"left": [left], "right": [_table([8]), _table([13, 13])]},
            watermarks={name: cf.DisabledWatermarks() for name in ("left", "right")},
        ) as stream:
            tables = [event.table async for event in stream]
        assert pa.concat_tables(tables).equals(left.slice(0, 2), check_metadata=True)

    asyncio.run(exercise())


def test_bounded_left_anti_finite_emits_the_unwitnessed_left_suffix():
    relation = _declaration("left_anti")
    program = cf.Program("absent", mode="batch", outputs={"absent": relation})
    left = _table([10, 10, 10, 20, 21, 21])
    result = program.collect({"left": left, "right": _table([8, 13, 13])})["absent"]
    assert result.equals(left.slice(3), check_metadata=True)
    (join,) = [
        node["operator"]
        for node in program.to_project().model_dump()["graph"]["nodes"]
        if node["operator"]["kind"] == "stream_join"
    ]
    assert join["spec"]["join_type"] == "left_anti"


def test_left_anti_public_stream_waits_for_right_eof_and_skips_witnesses():
    import asyncio

    async def exercise():
        program = cf.Program(
            "stream", mode="stream", outputs={"absent": _declaration("left_anti")}
        )
        left = _table([10, 10, 10, 20, 21, 21])
        async with program.stream(
            {
                "left": [left.slice(i, 2) for i in range(0, len(left), 2)],
                "right": [_table([8]), _table([13, 13])],
            },
            watermarks={name: cf.DisabledWatermarks() for name in ("left", "right")},
        ) as stream:
            tables = [event.table async for event in stream]
        assert pa.concat_tables(tables).equals(left.slice(3), check_metadata=True)

    asyncio.run(exercise())


def test_left_anti_rejects_null_in_declared_nonnull_left_time_before_output():
    import asyncio

    async def exercise():
        program = cf.Program(
            "reject", mode="stream", outputs={"absent": _declaration("left_anti")}
        )
        events = []
        with pytest.raises(
            TypeError,
            match="Column 'time' is declared as non-nullable but contains null values",
        ):
            async with program.stream(
                {"left": [_table([None])], "right": [_table([])]}
            ) as stream:
                async for event in stream:
                    events.append(event)
        assert events == []

    asyncio.run(exercise())
