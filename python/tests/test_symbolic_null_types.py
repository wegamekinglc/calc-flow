from __future__ import annotations

import re

import pyarrow as pa
import pytest

import calc_flow as cf


@pytest.mark.parametrize(
    ("data_type", "values"),
    [
        ("float64", [1.0, None, 3.0]),
        ("float32", [1.0, None, 3.0]),
        ("int64", [1, None, 3]),
        ("bool", [True, None, False]),
        ("string", ["a", None, "b"]),
    ],
)
def test_coalesce_infers_type_after_leading_null(
    data_type: str, values: list[object]
) -> None:
    arrow_type = pa.type_for_alias(data_type)
    data = pa.table({"x": pa.array(values, type=arrow_type)})

    actual = cf.compute(
        data, lambda table: table.select(value=cf.row.coalesce(None, table["x"]))
    )

    assert actual.to_pydict() == {"value": values}
    assert actual.schema == pa.schema([pa.field("value", arrow_type)])


@pytest.mark.parametrize("position", ["leading", "trailing", "multiple_leading"])
@pytest.mark.parametrize("nullable", [False, True])
def test_coalesce_preserves_type_and_nullability_for_null_positions(
    position: str, nullable: bool
) -> None:
    schema = pa.schema([pa.field("x", pa.float64(), nullable=nullable)])
    data = pa.table({"x": [1.0, None if nullable else 2.0, 3.0]}, schema=schema)
    table = cf.table_input("input", schema=schema)
    operands = {
        "leading": (None, table["x"], None),
        "trailing": (table["x"], None, None),
        "multiple_leading": (None, None, table["x"], None),
    }[position]
    program = cf.Program(
        "null-types",
        engine="sql",
        outputs={"output": table.select(value=cf.row.coalesce(*operands))},
    )

    assert program.analyze().issues == ()
    assert f"field value float64 nullable={str(nullable).lower()}" in program.explain()
    actual = program.collect({"input": data})["output"]
    assert actual.to_pydict() == {"value": data["x"].to_pylist()}
    assert actual.schema == pa.schema(
        [pa.field("value", pa.float64(), nullable=nullable)]
    )


def test_where_infers_type_from_false_branch_after_null_true_branch() -> None:
    data = pa.table({"x": [1.0, None, 3.0], "condition": [True, False, None]})

    actual = cf.compute(
        data,
        lambda table: table.select(
            value=cf.row.where(table["condition"], None, table["x"])
        ),
    )

    assert actual.to_pydict() == {"value": [None, None, 3.0]}
    assert actual.schema == pa.schema([pa.field("value", pa.float64())])


@pytest.mark.parametrize("null_branch", ["true", "false"])
@pytest.mark.parametrize("typed_null", [False, True])
def test_where_preserves_typed_and_untyped_null_branches(
    null_branch: str, typed_null: bool
) -> None:
    schema = pa.schema(
        [pa.field("x", pa.float64(), nullable=False), pa.field("condition", pa.bool_())]
    )
    data = pa.table({"x": [1.0, 2.0, 3.0], "condition": [True, False, None]}, schema)
    table = cf.table_input("input", schema=schema)
    null = cf.row.cast(None, "float64") if typed_null else None
    branches = (null, table["x"]) if null_branch == "true" else (table["x"], null)
    program = cf.Program(
        "null-branches",
        engine="sql",
        outputs={
            "output": table.select(value=cf.row.where(table["condition"], *branches))
        },
    )

    assert program.analyze().issues == ()
    assert "field value float64 nullable=true" in program.explain()
    actual = program.collect({"input": data})["output"]
    expected = [None, 2.0, 3.0] if null_branch == "true" else [1.0, None, None]
    assert actual.to_pydict() == {"value": expected}
    assert actual.schema == pa.schema([pa.field("value", pa.float64())])


def test_coalesce_preserves_typed_null_and_non_nullable_value() -> None:
    schema = pa.schema([pa.field("x", pa.float64(), nullable=False)])
    data = pa.table({"x": [1.0, 2.0]}, schema=schema)
    table = cf.table_input("input", schema=schema)
    program = cf.Program(
        "typed-null",
        engine="sql",
        outputs={
            "output": table.select(
                value=cf.row.coalesce(cf.row.cast(None, "float64"), table["x"])
            )
        },
    )

    assert "field value float64 nullable=false" in program.explain()
    actual = program.collect({"input": data})["output"]
    assert actual.to_pydict() == {"value": [1.0, 2.0]}
    assert actual.schema == pa.schema([pa.field("value", pa.float64(), nullable=False)])


@pytest.mark.parametrize("other_type", ["float32", "int64", "string"])
@pytest.mark.parametrize("operation", ["coalesce", "where"])
def test_null_type_inference_still_rejects_incompatible_concrete_types(
    operation: str, other_type: str
) -> None:
    table = cf.table_input(
        "input",
        schema=[
            cf.Field("condition", "bool"),
            cf.Field("x", "float64"),
            cf.Field("y", other_type),
        ],
    )
    value = (
        cf.row.coalesce(None, table["x"], table["y"])
        if operation == "coalesce"
        else cf.row.where(
            table["condition"], cf.row.coalesce(None, table["x"]), table["y"]
        )
    )
    program = cf.Program(
        "incompatible-types",
        engine="sql",
        outputs={"output": table.with_columns(value=value)},
    )
    suffix = (
        "coalesce.values[2].dtype"
        if operation == "coalesce"
        else "where.when_false.dtype"
    )
    path = f"outputs.output.value.{suffix}"

    assert any(
        issue.path == path and issue.code == "unsupported_type"
        for issue in program.analyze().issues
    )
    with pytest.raises(cf.CompileError, match=re.escape(f"{path}: unsupported_type:")):
        program.compile_batch()


@pytest.mark.parametrize("operation", ["coalesce", "where"])
def test_null_type_inference_still_rejects_mixed_input_lineages(
    operation: str,
) -> None:
    schema = [cf.Field("x", "float64"), cf.Field("condition", "bool")]
    left = cf.table_input("left", schema=schema)
    right = cf.table_input("right", schema=schema)
    value = (
        cf.row.coalesce(None, left["x"], right["x"])
        if operation == "coalesce"
        else cf.row.where(left["condition"], None, right["x"])
    )
    program = cf.Program(
        "mixed-lineage",
        engine="sql",
        inputs=[left, right],
        outputs={"output": left.with_columns(value=value)},
    )
    suffix = (
        "coalesce.values[2].lineage"
        if operation == "coalesce"
        else "where.when_false.lineage"
    )
    path = f"outputs.output.value.{suffix}"

    assert any(
        issue.path == path and issue.code == "schema_mismatch"
        for issue in program.analyze().issues
    )
    with pytest.raises(cf.CompileError, match=re.escape(f"{path}: schema_mismatch:")):
        program.compile_batch()


@pytest.mark.parametrize("operation", ["coalesce", "where"])
def test_all_untyped_null_values_still_require_an_explicit_cast(
    operation: str,
) -> None:
    table = cf.table_input("input", schema=[cf.Field("condition", "bool")])
    value = (
        cf.row.coalesce(None, None)
        if operation == "coalesce"
        else cf.row.where(table["condition"], None, None)
    )
    program = cf.Program(
        "all-null",
        engine="sql",
        outputs={"output": table.select(value=value)},
    )

    with pytest.raises(cf.CompileError, match="unresolved_type: unknown field 'value'"):
        program.compile_batch()


@pytest.mark.parametrize("row_count", [0, 1])
@pytest.mark.parametrize("operation", ["coalesce", "where"])
def test_leading_null_handles_empty_and_single_row_inputs(
    operation: str, row_count: int
) -> None:
    schema = pa.schema([pa.field("x", pa.float64()), pa.field("condition", pa.bool_())])
    data = pa.table({"x": [1.0] * row_count, "condition": [False] * row_count}, schema)

    def build(table: cf.TableExpr) -> cf.TableExpr:
        value = (
            cf.row.coalesce(None, table["x"])
            if operation == "coalesce"
            else cf.row.where(table["condition"], None, table["x"])
        )
        return table.select(value=value)

    actual = cf.compute(data, build)

    assert actual.to_pydict() == {"value": [1.0] * row_count}
    assert actual.schema == pa.schema([pa.field("value", pa.float64())])
