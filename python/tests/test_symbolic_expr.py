from __future__ import annotations

import hashlib

import pytest

from calc_flow import symbolic
from calc_flow.symbolic import (
    ArrayExpr,
    ColumnExpr,
    Field,
    TableExpr,
    cs,
    event_time_bucket,
    exact_time,
    linalg,
    parameter,
    row,
    rows,
    table,
    table_input,
    ts,
    window,
)
from calc_flow.symbolic.nodes import (
    CBytes,
    CDType,
    CEnum,
    CFloat,
    CInt,
    CMap,
    CNull,
    CSeq,
    CShape,
    CStr,
    encode_value,
)

# Normative v1 golden vectors from
# .codex/artifacts/specs/symbolic-computation-contract.md section 4.4.
VALUE_BYTES_HEX = (
    "0800000000000000090003010000000000000001048000000000000000047ff8"
    "000000000000050000000000000002c3a906000000000000000200ff07000000"
    "000000000a62617463685f6b696e64000000000000000561727261790a000000"
    "00000000020000000000000000020100000000000000016e0b00000000000000"
    "07666c6f61743634"
)
VALUE_DIGEST = "b3ec4d3d06b466e01f5e9de9fe2e9d2f77a48257a5bfeda79e9d6b8deee92008"
NODE_BYTES_HEX = (
    "20000000000000000b7461626c655f696e707574000000000000000131000000"
    "00000000000900000000000000050000000000000009656e746974795f627908"
    "0000000000000000000000000000000a6576656e745f74696d65000000000000"
    "0000046e616d6505000000000000000671756f74657300000000000000067363"
    "68656d6108000000000000000109000000000000000300000000000000096461"
    "74615f747970650b0000000000000007666c6f6174363400000000000000046e"
    "616d650500000000000000017800000000000000086e756c6c61626c65020000"
    "00000000000b73657175656e63655f6279080000000000000000"
)
NODE_DIGEST = "961b52bfdfb340125fa0b241b312521a43c9dce63dcc6b92717e1f1f2cdb7772"


def _quotes() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("x", "float64"),
            Field("y", "float64"),
        ],
    )


def _array() -> ArrayExpr:
    return linalg.from_columns(_quotes(), columns=["x", "y"], backend="numpy")


def test_public_surface_is_the_frozen_export_list() -> None:
    assert symbolic.__all__ == [
        "ArrayExpr",
        "ColumnExpr",
        "CrossSectionGroup",
        "DurationFrame",
        "EventTimeBucket",
        "Expr",
        "Field",
        "Parameter",
        "RowFrame",
        "TableExpr",
        "cs",
        "duration",
        "event_time_bucket",
        "exact_time",
        "linalg",
        "parameter",
        "row",
        "rows",
        "table",
        "table_input",
        "ts",
        "window",
    ]
    for name in symbolic.__all__:
        assert hasattr(symbolic, name)


def test_no_public_execution_members_exist() -> None:
    public_names = {name for name in dir(symbolic) if not name.startswith("_")}
    for forbidden in ("eval", "push", "value", "transform"):
        assert forbidden not in public_names
    for namespace in (row, ts, cs, window, table, linalg):
        for forbidden in ("eval", "push", "value", "transform"):
            assert not hasattr(namespace, forbidden)
    for expression_class in (ColumnExpr, ArrayExpr, TableExpr):
        for forbidden in ("eval", "push", "value", "transform"):
            assert not hasattr(expression_class, forbidden)


def test_canonical_value_encoder_reproduces_golden_bytes() -> None:
    value = CSeq(
        (
            CNull(),
            CInt(-1),
            CFloat(-0.0),
            CFloat(float("nan")),
            CStr("é"),
            CBytes(b"\x00\xff"),
            CEnum("batch_kind", "array"),
            CShape((CInt(2), CStr("n"))),
            CDType("float64"),
        )
    )
    encoded = encode_value(value)
    assert encoded.hex() == VALUE_BYTES_HEX


def test_canonical_value_digest_matches_golden_vector() -> None:
    value = CSeq(
        (
            CNull(),
            CInt(-1),
            CFloat(-0.0),
            CFloat(float("nan")),
            CStr("é"),
            CBytes(b"\x00\xff"),
            CEnum("batch_kind", "array"),
            CShape((CInt(2), CStr("n"))),
            CDType("float64"),
        )
    )
    assert hashlib.sha256(encode_value(value)).hexdigest() == VALUE_DIGEST


def test_table_input_node_bytes_and_digest_match_golden_vector() -> None:
    quotes = table_input("quotes", schema=[Field("x", "float64")])
    assert quotes._node.node_bytes.hex() == NODE_BYTES_HEX
    assert quotes.digest == NODE_DIGEST


def test_shape_rejects_negative_known_dimensions() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CShape((CInt(2), CInt(-1)))


def test_defaults_materialize_to_identical_digest() -> None:
    explicit = table_input(
        "quotes",
        schema=[Field("x", "float64")],
        entity_by=(),
        event_time=None,
        sequence_by=(),
    )
    implicit = table_input("quotes", schema=[Field("x", "float64")])
    assert implicit.digest == explicit.digest


def test_mapping_insertion_order_is_not_identity() -> None:
    first = CMap.from_mapping({"name": CStr("q"), "entity_by": CSeq(())})
    second = CMap.from_mapping({"entity_by": CSeq(()), "name": CStr("q")})
    assert encode_value(first) == encode_value(second)


def test_declaration_order_is_identity() -> None:
    first = table_input("q", schema=[Field("x", "float64"), Field("y", "float32")])
    second = table_input("q", schema=[Field("y", "float32"), Field("x", "float64")])
    assert first.digest != second.digest


def test_argument_order_is_identity() -> None:
    x = _quotes()["x"]
    y = _quotes()["y"]
    assert (x - y).digest != (y - x).digest


def test_scalar_literal_type_identity_is_preserved() -> None:
    x = _quotes()["x"]
    assert len({(x + True).digest, (x + 1).digest, (x + 1.0).digest}) == 3
    assert (x + 0.0).digest != (x + -0.0).digest


def test_constructors_do_not_mutate_caller_sequences() -> None:
    entity = ["symbol"]
    sequence = ["seq"]
    schema = [Field("x", "float64")]
    columns = ["x"]
    partition = ["symbol"]
    quotes = table_input(
        "q",
        schema=schema,
        entity_by=entity,
        sequence_by=sequence,
    )
    exact_time(quotes["ts"], partition_by=[quotes["symbol"]])
    table.project(quotes, columns)
    digest_before = quotes.digest
    assert entity == ["symbol"]
    assert sequence == ["seq"]
    assert schema == [Field("x", "float64")]
    assert columns == ["x"]
    assert partition == ["symbol"]
    entity.append("other")
    sequence.append("other")
    schema.append(Field("z", "int64"))
    columns.append("z")
    partition.append("other")
    assert quotes.digest == digest_before


def test_internal_collections_are_immutable() -> None:
    quotes = _quotes()
    node = quotes["x"]._node
    assert isinstance(node.args, tuple)
    with pytest.raises(AttributeError):
        node.args = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        quotes._node = node  # type: ignore[misc]


def test_column_dunders_return_column_expressions() -> None:
    x = _quotes()["x"]
    y = _quotes()["y"]
    composed = [
        x == y,
        x != y,
        x < y,
        x <= y,
        x > y,
        x >= y,
        x + y,
        x - y,
        x * y,
        x / y,
        -x,
        x & y,
        x | y,
        ~x,
    ]
    assert all(isinstance(value, ColumnExpr) for value in composed)
    with_scalar = [
        x + 1,
        1 + x,
        x - 1,
        1 - x,
        x * 2.0,
        2.0 * x,
        x / 2,
        x + None,
        x + "s",
        x + True,
        x & False,
        False | x,
    ]
    assert all(isinstance(value, ColumnExpr) for value in with_scalar)


def test_array_dunders_return_array_expressions() -> None:
    array = _array()
    other = _array()
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    composed = [
        array == other,
        array != other,
        array < other,
        array <= other,
        array > other,
        array >= other,
        array + other,
        array - other,
        array * other,
        array / other,
        -array,
        array & other,
        array | other,
        ~array,
        array + weights,
        weights + array,
        array + 1.0,
        1.0 + array,
    ]
    assert all(isinstance(value, ArrayExpr) for value in composed)


def test_parameter_exposes_name_and_kind() -> None:
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(3, 1)
    )
    assert weights.name == "weights"
    assert weights.kind == "array"
    tables = parameter("reference", kind="table", schema=[Field("k", "string")])
    assert tables.kind == "table"


def test_linalg_and_table_namespaces_compose() -> None:
    quotes = _quotes()
    array = linalg.from_columns(quotes, columns=["x", "y"], backend="numpy")
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    scores = linalg.matmul(array, weights)
    assert isinstance(scores, ArrayExpr)
    enriched = table.attach_columns(quotes, scores, names=["score"])
    assert isinstance(enriched, TableExpr)
    projected = table.project(enriched, ["ts", "symbol", "score"])
    assert isinstance(projected, TableExpr)
    filtered = table.filter(projected, projected["score"] > 0)
    assert isinstance(filtered, TableExpr)
    tumbling = window.tumbling(
        quotes, event_time="ts", size_micros=60_000_000, group_by=["symbol"]
    )
    assert isinstance(tumbling, TableExpr)
    hopping = window.hopping(
        quotes,
        event_time="ts",
        size_micros=120_000_000,
        slide_micros=60_000_000,
    )
    assert isinstance(hopping, TableExpr)


def test_temporal_and_cross_section_namespaces_compose() -> None:
    quotes = _quotes()
    x = quotes["x"]
    momentum = ts.mean(x, window=rows(20), min_periods=20)
    assert isinstance(momentum, ColumnExpr)
    lagged = ts.lag(momentum, periods=3)
    spread = ts.covariance(x, quotes["y"], window=rows(5), ddof=0)
    group = event_time_bucket(
        quotes["ts"], width_micros=60_000_000, partition_by=[quotes["symbol"]]
    )
    ranked = cs.rank(momentum, group=group, direction="descending")
    zscored = cs.zscore(momentum, group=exact_time(quotes["ts"]), ddof=1)
    winsorized = cs.winsorize(momentum, group=group, lower=0.05, upper=0.95)
    composed = row.clip(
        row.cast(lagged + spread + ranked + zscored + winsorized, "float64"),
        lower=0.0,
        upper=1.0,
    )
    assert isinstance(composed, ColumnExpr)
    assert isinstance(row.coalesce(composed, 0.0), ColumnExpr)
    assert isinstance(row.where(quotes["symbol"] != "", composed, 0.0), ColumnExpr)


def test_explain_is_deterministic_and_address_free() -> None:
    first = ts.mean(_quotes()["x"], window=rows(20))
    second = ts.mean(_quotes()["x"], window=rows(20))
    assert first.explain() == second.explain()
    explained = first.explain()
    assert "mean" in explained
    assert "20" in explained
    assert "0x" not in explained


def test_identical_compares_normalized_structure() -> None:
    x = _quotes()["x"]
    assert (x + 1).identical(x + 1)
    assert not (x + 1).identical(x + 2)
    assert not (x + 1).identical(x)
    assert not (x + 1).identical(1)
    assert not (x + 1).identical(None)
    weights = parameter("w", kind="array", backend="numpy", dtype="float64", shape=(1,))
    assert weights.identical(
        parameter("w", kind="array", backend="numpy", dtype="float64", shape=(1,))
    )


def test_digest_is_lowercase_sha256_hex() -> None:
    digest = _quotes().digest
    assert len(digest) == 64
    assert digest == digest.lower()
    int(digest, 16)


def test_row_and_cs_defaults_materialize_identically() -> None:
    x = _quotes()["x"]
    group = exact_time(_quotes()["ts"])
    explicit = cs.rank(
        x,
        group=group,
        direction="ascending",
        tie_method="average",
        null_placement="exclude",
        min_samples=1,
    )
    implicit = cs.rank(x, group=group)
    assert implicit.digest == explicit.digest
    assert (
        ts.count(x, window=rows(5)).digest
        == ts.count(x, window=rows(5), min_periods=1).digest
    )
