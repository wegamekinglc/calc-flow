from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from calc_flow import symbolic
from calc_flow.symbolic import Field, Program, table, table_input, window
from calc_flow.symbolic.nodes import CMap, CSeq, CStr, build

_OLD_GOLDENS = [
    (
        "tumbling",
        60,
        None,
        ("symbol",),
        "6d1aa619d52e84dd059a0166b1994ab5ad9bc4963fca545d61af5dc7ad3f0a50",
        "35aa76f26d6b217efa060ecad4794a60b95aa431d8e7268d49130cda62fe65f6",
    ),
    (
        "hopping",
        60,
        30,
        ("symbol",),
        "fb4aca7852f1df5a7a6622be36d2c7cda79507fa668eb2d6068964d5803fd1cd",
        "92b631592d8a2241d90d96f29a14c368c50ef01081c2ef36e4a8b90fb79b2591",
    ),
    (
        "hopping",
        3,
        2,
        (),
        "ccac44b64f3bb21fee30b89638b49109633fc033b2a1aaf90b4769f44bd46ccd",
        "6b694faa8469da679739260168f46d0cd374d127e38c053f0671dec7fe2f6c22",
    ),
    (
        "hopping",
        1025,
        1,
        (),
        "6fdc8bb7db734162f4d5961027017f1b2de138aeea376d1c63292a232701cae0",
        "a5fb2d74e7d99d1b1f2e39e2a8a0e6b4310fd8a927fe95c2620941f0203d83ed",
    ),
    (
        "tumbling",
        60,
        None,
        ("symbol", "symbol"),
        "9a30d77c212dae875986b6f00ee319eb20ede53f1b29b08cf5f75267ae9e9e11",
        "4559a4827ea465094ba0fecc2b95f248d1543999a181774abade495659b78518",
    ),
    (
        "tumbling",
        60,
        None,
        ("window_start", "window_end"),
        "8e5c378cd18f62ffaa8b36390a4c3868ce0348da854c63ebd4119eb25af6c592",
        "2b8b0d05ccc2b4b94fcbff70f7dca120492aa5c0ca04e66f30caf3409979897f",
    ),
]

_NODE_GOLDEN_BYTES = {
    (
        "tumbling",
        1,
    ): (
        "20000000000000000f77696e646f775f74756d626c696e67000000000000000131000000"
        "000000000194b95ac14496f9154b6786109101365c0b759e7d7097e8e98477a2b14c47f6"
        "24090000000000000003000000000000000a6576656e745f74696d650500000000000000"
        "027473000000000000000867726f75705f62790800000000000000010500000000000000"
        "0673796d626f6c000000000000000b73697a655f6d6963726f730300000000000000003c"
    ),
    (
        "hopping",
        1,
    ): (
        "20000000000000000e77696e646f775f686f7070696e6700000000000000013100000000"
        "0000000194b95ac14496f9154b6786109101365c0b759e7d7097e8e98477a2b14c47f624"
        "090000000000000004000000000000000a6576656e745f74696d65050000000000000002"
        "7473000000000000000867726f75705f6279080000000000000001050000000000000006"
        "73796d626f6c000000000000000b73697a655f6d6963726f730300000000000000003c00"
        "0000000000000c736c6964655f6d6963726f730300000000000000001e"
    ),
    (
        "tumbling",
        2,
    ): (
        "20000000000000000f77696e646f775f74756d626c696e67000000000000000132000000"
        "000000000194b95ac14496f9154b6786109101365c0b759e7d7097e8e98477a2b14c47f6"
        "24090000000000000004000000000000000a616767726567617465730800000000000000"
        "010900000000000000030000000000000006636f6c756d6e05000000000000000576616c"
        "7565000000000000000866756e6374696f6e05000000000000000373756d000000000000"
        "00066f7574707574050000000000000005746f74616c000000000000000a6576656e745f"
        "74696d650500000000000000027473000000000000000867726f75705f62790800000000"
        "0000000105000000000000000673796d626f6c000000000000000b73697a655f6d696372"
        "6f730300000000000000003c"
    ),
    (
        "hopping",
        2,
    ): (
        "20000000000000000e77696e646f775f686f7070696e6700000000000000013200000000"
        "0000000194b95ac14496f9154b6786109101365c0b759e7d7097e8e98477a2b14c47f624"
        "090000000000000005000000000000000a61676772656761746573080000000000000001"
        "0900000000000000030000000000000006636f6c756d6e05000000000000000576616c75"
        "65000000000000000866756e6374696f6e05000000000000000373756d00000000000000"
        "066f7574707574050000000000000005746f74616c000000000000000a6576656e745f74"
        "696d650500000000000000027473000000000000000867726f75705f6279080000000000"
        "00000105000000000000000673796d626f6c000000000000000b73697a655f6d6963726f"
        "730300000000000000003c000000000000000c736c6964655f6d6963726f730300000000"
        "000000001e"
    ),
}


def _trades():
    return table_input(
        "trades",
        schema=[
            Field("ts", "timestamp[us, UTC]"),
            Field("symbol", "string"),
            Field("value", "int64"),
        ],
    )


@pytest.mark.parametrize("function", ["count", "sum", "min", "max", "avg"])
def test_window_aggregate_helpers_produce_immutable_declarations(function):
    aggregate = getattr(window, function)("value", output="result")
    assert aggregate == symbolic.WindowAggregate(function, "value", "result")
    assert not hasattr(aggregate, "__dict__")
    with pytest.raises(FrozenInstanceError):
        aggregate.output = "changed"


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("function", "median", ValueError),
        ("function", None, TypeError),
        ("column", "", ValueError),
        ("column", 1, TypeError),
        ("output", "", ValueError),
        ("output", lambda: "result", TypeError),
    ],
)
def test_window_aggregate_rejects_invalid_data_fields(field, value, exception):
    fields = {"function": "sum", "column": "value", "output": "result"}
    fields[field] = value
    with pytest.raises(exception, match=rf"WindowAggregate\.{field}"):
        symbolic.WindowAggregate(**fields)


@pytest.mark.parametrize("function", ["count", "sum", "min", "max", "avg"])
@pytest.mark.parametrize(
    ("column", "output", "field"), [("", "result", "column"), ("value", "", "output")]
)
def test_window_aggregate_helpers_preserve_namespace_error_path(
    function, column, output, field
):
    with pytest.raises(
        ValueError,
        match=rf"calc_flow\.symbolic\.window\.{function}\.{field}: invalid_literal",
    ):
        getattr(window, function)(column, output=output)


@pytest.mark.parametrize("case", _OLD_GOLDENS)
def test_legacy_window_declarations_keep_constructor_domain_and_identity(case):
    kind, size, slide, groups, digest, fingerprint = case
    trades = _trades()
    kwargs = dict(event_time="ts", size_micros=size, group_by=groups)
    if slide is not None:
        kwargs["slide_micros"] = slide
    implicit = getattr(window, kind)(trades, **kwargs)
    explicit = getattr(window, kind)(trades, **kwargs, aggregates=None)
    assert implicit.digest == explicit.digest == digest
    assert implicit._node.node_bytes == explicit._node.node_bytes
    assert implicit._node.op.version == 1
    assert implicit._node.attrs.get("aggregates") is None
    assert (
        Program("golden", inputs=[trades], outputs=[("result", explicit)]).fingerprint
        == fingerprint
    )
    if size == 60 and groups == ("symbol",):
        assert implicit._node.node_bytes.hex() == _NODE_GOLDEN_BYTES[kind, 1]


@pytest.mark.parametrize(
    ("kind", "digest"),
    [
        (
            "tumbling",
            "27901486d53149124148f9d2faa1e9c824da55ae486a9456e7aceb1a597d3f93",
        ),
        ("hopping", "979352cbfe39c5e4ebf9c53b32605b43fe7944b593ca4ff697e41a7f6f949ff7"),
    ],
)
def test_executable_window_declarations_match_frozen_bytes_and_digest(kind, digest):
    kwargs = dict(
        event_time="ts",
        size_micros=60,
        group_by=["symbol"],
        aggregates=[window.sum("value", output="total")],
    )
    if kind == "hopping":
        kwargs["slide_micros"] = 30
    expression = getattr(window, kind)(_trades(), **kwargs)
    assert expression._node.op.version == 2
    assert expression.digest == digest
    assert expression._node.node_bytes.hex() == _NODE_GOLDEN_BYTES[kind, 2]


def test_executable_window_copies_ordered_groups_and_aggregates():
    trades = _trades()
    groups = ["symbol"]
    aggregates = [
        window.sum("value", output="first"),
        window.sum("value", output="second"),
    ]
    expression = window.tumbling(
        trades, event_time="ts", size_micros=60, group_by=groups, aggregates=aggregates
    )
    program = Program("p", inputs=[trades], outputs=[("result", expression)])
    fingerprint = program.fingerprint
    equivalent = window.tumbling(
        _trades(),
        event_time="ts",
        size_micros=60,
        group_by=tuple(groups),
        aggregates=tuple(aggregates),
    )
    assert expression.digest == equivalent.digest
    groups.clear()
    aggregates.reverse()
    assert program.fingerprint == fingerprint
    assert expression._node.attrs.get("group_by") == CSeq((CStr("symbol"),))
    assert expression._node.attrs.get("aggregates") == CSeq(
        tuple(
            CMap.from_mapping(
                {
                    "function": CStr("sum"),
                    "column": CStr("value"),
                    "output": CStr(output),
                }
            )
            for output in ("first", "second")
        )
    )
    changed = window.tumbling(
        trades,
        event_time="ts",
        size_micros=60,
        group_by=["symbol"],
        aggregates=aggregates,
    )
    assert changed.digest != expression.digest
    assert (
        Program("p", inputs=[trades], outputs=[("result", changed)]).fingerprint
        != fingerprint
    )


@pytest.mark.parametrize("kind", ["tumbling", "hopping"])
@pytest.mark.parametrize(
    "aggregates",
    [
        [],
        (),
        "sum",
        b"sum",
        {"sum": "value"},
        {"sum"},
        iter(()),
        ["sum"],
        [{"function": "sum", "column": "value", "output": "total"}],
        [lambda: "sum"],
    ],
)
def test_executable_window_rejects_non_declaration_aggregate_sequences(
    kind, aggregates
):
    kwargs = dict(event_time="ts", size_micros=60, aggregates=aggregates)
    if kind == "hopping":
        kwargs["slide_micros"] = 30
    with pytest.raises(
        (TypeError, ValueError),
        match=rf"calc_flow\.symbolic\.window\.{kind}\.aggregates",
    ):
        getattr(window, kind)(_trades(), **kwargs)


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("tumbling", "size_micros"),
        ("hopping", "size_micros"),
        ("hopping", "slide_micros"),
    ],
)
@pytest.mark.parametrize("value", [True, 1.0, "1", 0, -1, 2**64])
def test_executable_window_geometry_has_strict_u64_field_errors(kind, field, value):
    kwargs = dict(
        event_time="ts",
        size_micros=60,
        aggregates=[window.sum("value", output="total")],
    )
    if kind == "hopping":
        kwargs["slide_micros"] = 30
    kwargs[field] = value
    with pytest.raises(
        (TypeError, ValueError), match=rf"calc_flow\.symbolic\.window\.{kind}\.{field}"
    ):
        getattr(window, kind)(_trades(), **kwargs)


@pytest.mark.parametrize(("size", "slide"), [(3, 2), (1, 2), (1025, 1)])
def test_executable_hopping_rejects_nonmultiple_or_excessive_overlap(size, slide):
    with pytest.raises(
        ValueError,
        match=r"calc_flow\.symbolic\.window\.hopping\.slide_micros: invalid_literal",
    ):
        window.hopping(
            _trades(),
            event_time="ts",
            size_micros=size,
            slide_micros=slide,
            aggregates=[window.sum("value", output="total")],
        )


@pytest.mark.parametrize(
    ("groups", "outputs", "path"),
    [
        (["symbol", "symbol"], ["total"], r"group_by\[1\]"),
        (["window_start"], ["total"], r"group_by\[0\]"),
        (["window_end"], ["total"], r"group_by\[0\]"),
        (["symbol"], ["symbol"], r"aggregates\[0\]\.output"),
        ([], ["window_start"], r"aggregates\[0\]\.output"),
        ([], ["window_end"], r"aggregates\[0\]\.output"),
        ([], ["total", "total"], r"aggregates\[1\]\.output"),
    ],
)
@pytest.mark.parametrize("kind", ["tumbling", "hopping"])
def test_executable_window_rejects_output_and_group_name_collisions(
    kind, groups, outputs, path
):
    kwargs = dict(
        event_time="ts",
        size_micros=60,
        group_by=groups,
        aggregates=[window.sum("value", output=output) for output in outputs],
    )
    if kind == "hopping":
        kwargs["slide_micros"] = 30
    with pytest.raises(
        ValueError,
        match=rf"calc_flow\.symbolic\.window\.{kind}\.{path}: duplicate_name",
    ):
        getattr(window, kind)(_trades(), **kwargs)


@pytest.mark.parametrize("kind", ["tumbling", "hopping"])
def test_window_node_attributes_are_versioned_and_future_versions_fail_closed(kind):
    kwargs = dict(event_time="ts", size_micros=60)
    if kind == "hopping":
        kwargs["slide_micros"] = 30
    legacy = getattr(window, kind)(_trades(), **kwargs)._node
    attrs = dict(legacy.attrs.entries)
    with pytest.raises(ValueError, match="aggregates"):
        build(legacy.op.name, legacy.args, attrs, version=2)
    for version in (0, 3, True, "2"):
        with pytest.raises(ValueError, match="unknown_primitive_version"):
            build(legacy.op.name, legacy.args, attrs, version=version)
    attrs["aggregates"] = CSeq(())
    with pytest.raises(ValueError, match="does not accept attributes"):
        build(legacy.op.name, legacy.args, attrs, version=1)


@pytest.mark.parametrize(
    ("kind", "size", "slide"),
    [
        ("tumbling", 1, None),
        ("tumbling", 2**64 - 1, None),
        ("hopping", 1024, 1),
        ("hopping", 2**64 - 1, 2**64 - 1),
    ],
)
def test_executable_window_accepts_geometry_limits_and_input_output_name_reuse(
    kind, size, slide
):
    kwargs = dict(
        event_time="ts",
        size_micros=size,
        aggregates=[window.sum("value", output="value")],
    )
    if slide is not None:
        kwargs["slide_micros"] = slide
    expression = getattr(window, kind)(_trades(), **kwargs)
    assert expression._node.op.version == 2


@pytest.mark.parametrize(
    "changed",
    [
        "event_time",
        "size_micros",
        "slide_micros",
        "group_by",
        "function",
        "column",
        "output",
        "input",
    ],
)
def test_executable_window_identity_covers_all_semantic_inputs(changed):
    trades = _trades()
    kwargs = dict(
        event_time="ts",
        size_micros=60,
        slide_micros=30,
        group_by=["symbol", "value"],
        aggregates=[window.sum("value", output="total")],
    )
    baseline = window.hopping(trades, **kwargs)
    changes = {
        "event_time": "time",
        "size_micros": 120,
        "slide_micros": 60,
        "group_by": ["value", "symbol"],
    }
    if changed in changes:
        kwargs[changed] = changes[changed]
    elif changed == "function":
        kwargs["aggregates"] = [window.avg("value", output="total")]
    elif changed == "column":
        kwargs["aggregates"] = [window.sum("other", output="total")]
    elif changed == "output":
        kwargs["aggregates"] = [window.sum("value", output="volume")]
    else:
        trades = table.filter(trades, trades["value"] > 0)
    assert window.hopping(trades, **kwargs).digest != baseline.digest


def test_window_aggregate_names_accept_only_exact_strings():
    class Name(str):
        pass

    for field in ("function", "column", "output"):
        kwargs = {"function": "sum", "column": "value", "output": "total"}
        kwargs[field] = Name(kwargs[field])
        with pytest.raises(TypeError, match=rf"WindowAggregate\.{field}"):
            symbolic.WindowAggregate(**kwargs)
    with pytest.raises(TypeError, match=r"window\.sum\.column"):
        window.sum(_trades()["value"], output="total")
