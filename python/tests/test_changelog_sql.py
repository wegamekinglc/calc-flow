from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

import calc_flow as cf


def _source(name: str = "events") -> cf.TableExpr:
    return cf.table_input(
        name,
        schema=pa.schema([("key", pa.string()), ("value", pa.int64())]),
    )


def _query(alias: str = "events") -> str:
    return f"SELECT key, SUM(value) AS total FROM {alias} GROUP BY key"


def _program(output: cf.TableExpr) -> cf.Program:
    return cf.Program("changes", engine="streaming", outputs={"result": output})


def _sql_node(program: cf.Program) -> dict[str, object]:
    document = program.to_project().root
    return next(
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "sql" and "SUM" in node["operator"]["query"]
    )


async def _tables(*tables: pa.Table):
    for table in tables:
        yield table


def test_changelog_declarations_lower_opt_in_and_preserve_default_project_shape():
    source = _source()
    query = _query("input")
    output = cf.changelog_sql(query, input=source)
    method_output = source.changelog_sql(query)
    assert output._node.digest == method_output._node.digest
    assert output._node.digest != cf.sql(query, input=source)._node.digest

    operator = _sql_node(_program(output))["operator"]
    assert operator["aggregate_output"] == "changelog"
    schema = _sql_node(_program(output))["output_ports"][0]["schema"]
    assert schema[-1] == {
        "name": "_cf_op",
        "data_type": "string",
        "nullable": False,
    }

    default_program = _program(cf.sql(_query(), events=source))
    default = _sql_node(default_program)
    assert "aggregate_output" not in default["operator"]
    explicit_snapshot = default_program.to_project().model_dump()
    sql_node = next(
        node
        for node in explicit_snapshot["graph"]["nodes"]
        if node["operator"]["kind"] == "sql" and "SUM" in node["operator"]["query"]
    )
    sql_node["operator"]["aggregate_output"] = "snapshot"
    restored = cf.ProjectDocument.model_validate(explicit_snapshot).model_dump_json()
    assert "aggregate_output" not in restored
    assert (
        cf.Runtime()._compile_stream_graph_project(restored).fingerprint
        == default_program.compile_stream().fingerprint
    )


def test_changelog_is_stream_only_single_alias_and_compact_only():
    source = _source()
    with pytest.raises(cf.CompileError, match="changelog_sql.*streaming engine"):
        cf.Program(
            "batch",
            engine="sql",
            outputs={"result": source.changelog_sql(_query("input"))},
        ).to_project()
    with pytest.raises(ValueError, match="one table alias"):
        _program(
            cf.changelog_sql(
                "SELECT SUM(l.value) AS total FROM l CROSS JOIN r",
                l=source,
                r=_source("other"),
            )
        )
    for query in (
        "SELECT value FROM input",
        "SELECT SUM(value) AS total FROM input GROUP BY key",
        "SELECT key, SUM(value) AS total FROM input GROUP BY key ORDER BY key",
        "SELECT key, SUM(value) AS total FROM input GROUP BY key LIMIT 1",
        "SELECT key AS _cf_op, SUM(value) AS total FROM input GROUP BY key",
        "SELECT key, COUNT(DISTINCT value) AS n FROM input GROUP BY key",
    ):
        with pytest.raises((cf.CompileError, cf.ConfigError, cf.ExecutionError)):
            _program(source.changelog_sql(query)).to_project()


def test_changelog_stream_emits_old_and_new_visible_rows_in_group_order():
    output = _source().changelog_sql(_query("input"))
    first = pa.table(
        {"key": ["b", "a", "a"], "value": [2, 1, 3]},
        schema=pa.schema([("key", pa.string()), ("value", pa.int64())]),
    )
    second = pa.table(
        {"key": ["a"], "value": [1]},
        schema=first.schema,
    )

    async def run():
        async with output.stream(_tables(first, second)) as results:
            return [table async for table in results]

    actual = asyncio.run(run())
    assert [table.to_pydict() for table in actual] == [
        {
            "key": ["b", "a"],
            "total": [2, 4],
            "_cf_op": ["upsert", "upsert"],
        },
        {
            "key": ["a", "a"],
            "total": [4, 5],
            "_cf_op": ["retract", "upsert"],
        },
    ]
    assert actual[0].schema.field("_cf_op").nullable is False


def test_changelog_having_transitions_and_unchanged_visible_value():
    output = _source().changelog_sql(
        "SELECT key, SUM(value) AS total FROM input GROUP BY key HAVING SUM(value) >= 3"
    )
    schema = pa.schema([("key", pa.string()), ("value", pa.int64())])
    batches = [
        pa.table({"key": ["a"], "value": [1]}, schema=schema),
        pa.table({"key": ["a"], "value": [2]}, schema=schema),
        pa.table({"key": ["a"], "value": [-4]}, schema=schema),
        pa.table({"key": ["a"], "value": [0]}, schema=schema),
    ]

    async def run():
        async with output.stream(_tables(*batches)) as results:
            return [table.to_pydict() async for table in results]

    assert asyncio.run(run()) == [
        {"key": [], "total": [], "_cf_op": []},
        {"key": ["a"], "total": [3], "_cf_op": ["upsert"]},
        {"key": ["a"], "total": [3], "_cf_op": ["retract"]},
        {"key": [], "total": [], "_cf_op": []},
    ]


def test_default_sql_and_snapshot_sql_declarations_do_not_gain_changelog_field():
    source = _source()
    assert (
        "aggregate_output"
        not in _sql_node(_program(cf.sql(_query(), events=source)))["operator"]
    )
    snapshot = cf.snapshot_sql(_query(), events=source)
    operator = next(
        node["operator"]
        for node in _program(snapshot).to_project().root["graph"]["nodes"]
        if node["operator"]["kind"] == "snapshot_sql"
    )
    assert "aggregate_output" not in operator
