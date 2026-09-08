from __future__ import annotations

import pyarrow as pa
import pytest

import calc_flow as cf
from calc_flow import symbolic as old


def test_preferred_declarations_preserve_identity_and_copy_names():
    schema = pa.schema([pa.field("x", pa.float64(), nullable=False)])
    t = cf.table_input("q", schema=schema)
    explicit = old.table_input("q", schema=[old.Field("x", "float64", False)])
    assert t.identical(explicit)
    mapping = {"twice": t["x"] * 2}
    result = t.with_columns(mapping, plus=t["x"] + 1)
    expected = explicit.with_columns(
        old.FeatureSet(tuple(mapping.items()) + (("plus", t["x"] + 1),))
    )
    mapping.clear()
    assert result.identical(expected)
    assert t.select("x", twice=t["x"] * 2).identical(
        old.table.project(
            t.with_columns(old.FeatureSet((("twice", t["x"] * 2),))), ("x", "twice")
        )
    )
    assert t.filter(t["x"] > cf.lit(0)).identical(old.table.filter(t, t["x"] > 0))
    for name in old.__all__:
        assert getattr(cf, name) is getattr(old, name)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (pa.int32(), "int32"),
        (pa.float32(), "float32"),
        (pa.string(), "string"),
        (pa.timestamp("us", "UTC"), "timestamp[us, UTC]"),
    ],
)
def test_arrow_schema_normalizes_types(dtype, expected):
    assert cf.table_input("q", schema=pa.schema([pa.field("v", dtype)])).identical(
        old.table_input("q", schema=[old.Field("v", expected)])
    )


def test_declaration_errors():
    with pytest.raises(ValueError, match=r"inputs.q.schema.price.*unsupported_type"):
        cf.table_input("q", schema=pa.schema([pa.field("price", pa.list_(pa.int64()))]))
    t = cf.table_input("q", schema=pa.schema([pa.field("x", pa.int64())]))
    with pytest.raises(ValueError, match="duplicate"):
        t.with_columns({"v": t["x"]}, v=t["x"])
    with pytest.raises(ValueError, match="at least one"):
        t.select()
    with pytest.raises(ValueError, match="duplicate"):
        t.select("x", "x")


def test_program_infers_ordered_roots_and_copies_outputs():
    schema = [old.Field("x", "int64")]
    left = cf.table_input("left", schema=schema)
    right = cf.table_input("right", schema=schema)
    outputs = {"r": right.select("x"), "l": left.select("x")}
    program = cf.Program("p", outputs=outputs)
    explicit = cf.Program("p", inputs=(right, left), outputs=tuple(outputs.items()))
    outputs.clear()
    assert program.fingerprint == explicit.fingerprint
    assert program.analyze().issues == ()
    assert isinstance(program.explain(), str)
    assert program.compile_batch().name == "p"
    assert program.compile_stream().name == "p"
    assert cf.Program("p", inputs=(), outputs={"l": left}).analyze().issues
    conflict = cf.table_input("left", schema=[old.Field("y", "int64")])
    with pytest.raises(ValueError, match="duplicate"):
        cf.Program("p", outputs={"a": left, "b": conflict})


def test_project_export_is_strict_and_keeps_explicit_document():
    from calc_flow.symbolic.lower import lower_program_document

    t = cf.table_input("q", schema=[cf.Field("x", "int64")])
    program = cf.Program("p", outputs={"answer": t.select(y=t["x"] + 1)})
    runtime = cf.Runtime()
    project = program.to_project(runtime)
    assert isinstance(project, cf.ProjectDocument)
    assert project.model_dump()["format_version"] == 3
    from_lowerer = cf.ProjectDocument.model_validate(
        lower_program_document(program, runtime, "batch")
    )
    assert project == from_lowerer
    assert runtime.compile_batch_project(project.model_dump_json()).name == "p"
    assert program.to_project(mode="stream").model_dump()["runtime"]["mode"] == "stream"


@pytest.mark.parametrize("value", [None, False, 3, 2.5, "literal's value"])
def test_literal_columns_use_existing_scalar_semantics(value):
    data = pa.table({"x": [1, 2]})

    def build(t):
        literal = cf.lit(value)
        if value is None:
            literal = cf.row.cast(literal, "float64")
        return t.select(v=literal)

    assert cf.compute(data, build)["v"].to_pylist() == [value, value]
