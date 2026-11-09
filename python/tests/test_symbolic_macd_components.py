from __future__ import annotations

import pytest

import calc_flow as cf
from calc_flow.symbolic.nodes import CEnum, CInt, build


def _source(dtype: str = "float64", *, ordered: bool = True) -> cf.TableExpr:
    return cf.table_input(
        "quotes",
        schema=(
            cf.Field("entity", "string", nullable=False),
            cf.Field("time", "timestamp[us, UTC]", nullable=False),
            cf.Field("seq", "uint64", nullable=False),
            cf.Field("close", dtype),
        ),
        entity_by=("entity",) if ordered else (),
        event_time="time" if ordered else None,
        sequence_by=("seq",) if ordered else (),
    )


def test_macd_components_are_ordinary_canonical_component_references() -> None:
    source = _source()
    column = source["close"]
    parts = cf.ts.macd_components(column)
    assert isinstance(parts, cf.MacdOutputs)
    for component, value in (
        ("line", parts.line),
        ("signal", parts.signal),
        ("histogram", parts.histogram),
    ):
        assert isinstance(value, cf.ColumnExpr)
        assert value._node.op.name == "macd_component"
        assert value._node.op.version == 1
        assert value._node.args == (column._node,)
        assert value._node.attr("component") == CEnum("macd_component", component)
        assert value._node.attr("fast_span") == CInt(12)
        assert value._node.attr("slow_span") == CInt(26)
        assert value._node.attr("signal_span") == CInt(9)
        assert value._node.attr("min_periods") == CInt(1)
        assert value._node.attr("signal_min_periods") == CInt(1)
    assert parts.line.identical(cf.ts.macd_components(source["close"]).line)
    with pytest.raises(ValueError, match="unknown_primitive_version"):
        build(
            "macd_component",
            parts.line._node.args,
            dict(parts.line._node.attrs.entries),
            version=2,
        )


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("fast_span", True, TypeError),
        ("slow_span", 1.5, TypeError),
        ("signal_span", "2", TypeError),
        ("min_periods", 0, ValueError),
        ("signal_min_periods", 2**64, ValueError),
    ],
)
def test_macd_declarations_validate_all_parameters(
    name: str, value: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match=name):
        cf.ts.macd_components(_source()["close"], **{name: value})


def test_macd_declarations_keep_large_minima_and_reject_reversed_spans() -> None:
    source = _source()
    parts = cf.ts.macd_components(
        source["close"],
        fast_span=1,
        slow_span=2,
        min_periods=2**64 - 1,
        signal_min_periods=99,
    )
    assert parts.signal._node.attr("min_periods") == CInt(2**64 - 1)
    assert parts.signal._node.attr("signal_min_periods") == CInt(99)
    with pytest.raises(ValueError, match="fast_span.*less than slow_span"):
        cf.ts.macd_components(source["close"], fast_span=2, slow_span=2)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_macd_analysis_accepts_exact_legacy_numeric_types(dtype: str) -> None:
    source = _source(dtype)
    parts = cf.ts.macd_components(source["close"])
    program = cf.Program(
        "numeric",
        engine="sql",
        outputs={
            "out": source.with_columns(
                line=parts.line, signal=parts.signal, histogram=parts.histogram
            )
        },
    )
    assert program.analyze(cf.Runtime()).issues == ()


def test_macd_source_rejects_decimal_before_declaration() -> None:
    with pytest.raises(ValueError, match="unsupported_type.*decimal"):
        _source("decimal128(10, 2)")


def test_macd_analysis_rejects_boolean_input() -> None:
    source = _source("bool")
    program = cf.Program(
        "boolean",
        engine="sql",
        outputs={
            "out": source.with_columns(line=cf.ts.macd_components(source["close"]).line)
        },
    )
    assert any(
        issue.code == "unsupported_type" and "numeric" in issue.message
        for issue in program.analyze(cf.Runtime()).issues
    )


def test_macd_analysis_requires_ordering_in_batch() -> None:
    source = _source(ordered=False)
    program = cf.Program(
        "order",
        engine="sql",
        outputs={
            "out": source.with_columns(line=cf.ts.macd_components(source["close"]).line)
        },
    )
    assert any(
        issue.code == "ordering_required"
        for issue in program.analyze(cf.Runtime()).issues
    )


@pytest.mark.parametrize("mismatch", ["filter", "definition", "outer", "transitive"])
def test_macd_analysis_preserves_consumed_row_sets_and_definitions(
    mismatch: str,
) -> None:
    source = _source()
    if mismatch == "filter":
        operand = source.filter(source["close"] > 0.0)["close"]
        target, foreign = source, None
    elif mismatch == "definition":
        declared = source.with_columns(adjusted=source["close"] + 1.0)
        target = source.with_columns(adjusted=source["close"] + 2.0)
        operand, foreign = declared["adjusted"], None
    elif mismatch == "outer":
        target, operand = source, source["close"]
        foreign = source.filter(source["close"] > 0.0)["close"]
    else:
        first = source.with_columns(x=source["close"] + 1.0)
        other = source.with_columns(x=source["close"] + 2.0)
        declared = first.with_columns(y=first["x"] + 1.0)
        target = other.with_columns(y=first["x"] + 1.0)
        operand, foreign = declared["y"], None
    result = cf.ts.macd_components(operand).histogram
    if foreign is not None:
        result = result + foreign
    program = cf.Program(
        "proof", engine="sql", outputs={"out": target.with_columns(result=result)}
    )
    assert any(
        issue.code == "schema_mismatch" and "same row set" in issue.message
        for issue in program.analyze(cf.Runtime()).issues
    )
    with pytest.raises(cf.CompileError, match="same row set"):
        program.to_project(cf.Runtime())


def test_macd_analysis_accepts_matching_filtered_derived_definitions() -> None:
    source = _source()
    filtered = source.filter(source["close"] > 0.0)
    derived = filtered.with_columns(adjusted=filtered["close"] + 1.0)
    result = cf.ts.macd_components(derived["adjusted"]).histogram
    program = cf.Program(
        "same", engine="sql", outputs={"out": derived.with_columns(result=result)}
    )
    assert program.analyze(cf.Runtime()).issues == ()


def _document(program: cf.Program, mode: str = "batch") -> dict[str, object]:
    from calc_flow.symbolic.lower import lower_program_document

    return lower_program_document(program, cf.Runtime(), mode)


def _rolling_nodes(document: dict[str, object]) -> list[dict[str, object]]:
    return [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    ]


def _nodes_with_kind(document: dict[str, object], kind: str) -> list[dict[str, object]]:
    return [
        node
        for node in _rolling_nodes(document)
        if any(item["kind"] == kind for item in node["operator"]["spec"]["outputs"])
    ]


def _has_edge(document: dict[str, object], source: str, target: str) -> bool:
    return any(
        edge["source_node"] == source and edge["target_node"] == target
        for edge in document["graph"]["edges"]
    )


def _selected_names(document: dict[str, object], output: str) -> list[str]:
    node = next(node for node in document["graph"]["nodes"] if node["id"] == output)
    return [
        item.rsplit(" AS ", 1)[-1].strip('"') for item in node["operator"]["select"]
    ]


def test_macd_lowering_emits_normalized_components_and_layout_six() -> None:
    source = _source("uint64")
    parts = cf.ts.macd_components(
        source["close"],
        fast_span=2,
        slow_span=4,
        signal_span=3,
        min_periods=7,
        signal_min_periods=11,
    )
    output = source.with_columns(
        line=parts.line, signal=parts.signal, histogram=parts.histogram
    )
    program = cf.Program("components", engine="sql", outputs={"out": output})
    nodes = _rolling_nodes(_document(program))
    assert len(nodes) == 1
    spec = nodes[0]["operator"]["spec"]
    assert spec["state_layout_version"] == 6
    assert spec["outputs"] == [
        {
            "kind": "macd_component",
            "primitive_version": 1,
            "input": "close",
            "output": name,
            "component": name,
            "fast_span": 2,
            "slow_span": 4,
            "signal_span": 3,
            "min_periods": 7,
            "signal_min_periods": 11,
        }
        for name in ("line", "signal", "histogram")
    ]


def test_macd_named_outputs_share_one_owner_before_fork_and_keep_aliases() -> None:
    source = _source()
    parts = cf.ts.macd_components(source["close"])
    program = cf.Program(
        "fork",
        engine="sql",
        outputs={
            "lines": source.with_columns(a=parts.line, duplicate=parts.line),
            "signals": source.with_columns(b=parts.signal),
            "histograms": source.with_columns(c=parts.histogram),
        },
    )
    document = _document(program)
    nodes = _rolling_nodes(document)
    assert len(nodes) == 1
    declarations = nodes[0]["operator"]["spec"]["outputs"]
    assert len(declarations) == 3
    assert {item["component"] for item in declarations} == {
        "line",
        "signal",
        "histogram",
    }
    assert (
        len(
            [
                edge
                for edge in document["graph"]["edges"]
                if edge["source_node"] == nodes[0]["id"]
            ]
        )
        == 3
    )
    explain = program.explain(cf.Runtime())
    assert "shared_state_groups=1" in explain
    assert "fallback=primitive_macd_component_missing_from_census" in explain


def test_macd_named_outputs_share_row_local_input_materialization() -> None:
    source = _source()
    parts = cf.ts.macd_components(source["close"] + 1.0)
    program = cf.Program(
        "row_local_fork",
        engine="sql",
        outputs={
            "first": source.with_columns(a=parts.line),
            "second": source.with_columns(b=parts.histogram),
        },
    )
    document = _document(program)
    nodes = _rolling_nodes(document)
    assert len(nodes) == 1
    inputs = {item["input"] for item in nodes[0]["operator"]["spec"]["outputs"]}
    assert len(inputs) == 1
    assert "close" not in inputs
    assert "shared_state_groups=1" in program.explain(cf.Runtime())


def test_macd_materialization_is_independent_of_component_declaration_order() -> None:
    source = _source()
    first = cf.ts.macd_components(source["close"] + 1.0)
    second = cf.ts.macd_components(source["close"] + 2.0)
    program = cf.Program(
        "permuted",
        engine="sql",
        outputs={
            "one": source.with_columns(a=first.line, b=second.signal),
            "two": source.with_columns(c=second.histogram, d=first.signal),
        },
    )
    nodes = _rolling_nodes(_document(program))
    assert len(nodes) == 1
    declarations = nodes[0]["operator"]["spec"]["outputs"]
    assert len(declarations) == 4
    assert len({item["input"] for item in declarations}) == 2
    assert "shared_state_groups=2" in program.explain(cf.Runtime())


def test_macd_shared_dependency_is_emitted_once_before_component_forks() -> None:
    source = _source()
    average = cf.ts.ema(source["close"], span=3)
    parts = cf.ts.macd_components(average)
    program = cf.Program(
        "dependency",
        engine="sql",
        outputs={
            "line": source.with_columns(a=parts.line),
            "histogram": source.with_columns(b=parts.histogram),
            "average": source.with_columns(e=average),
        },
    )
    document = _document(program)
    nodes = _rolling_nodes(document)
    assert len(nodes) == 2
    upstream, downstream = nodes
    assert [item["kind"] for item in upstream["operator"]["spec"]["outputs"]] == [
        "ewma"
    ]
    assert {
        item["component"] for item in downstream["operator"]["spec"]["outputs"]
    } == {"line", "histogram"}
    assert _has_edge(document, upstream["id"], downstream["id"])
    assert _has_edge(document, upstream["id"], "average")
    assert not _has_edge(document, downstream["id"], "average")


def test_macd_shared_owner_precedes_unrelated_branch_state() -> None:
    source = _source()
    parts = cf.ts.macd_components(source["close"] + 1.0)
    program = cf.Program(
        "branch",
        engine="sql",
        outputs={
            "first": source.with_columns(
                a=parts.line, e=cf.ts.ema(source["close"] + 2.0, span=3)
            ),
            "second": source.with_columns(b=parts.histogram),
        },
    )
    document = _document(program)
    macd = _nodes_with_kind(document, "macd_component")
    assert len(macd) == 1
    assert {item["component"] for item in macd[0]["operator"]["spec"]["outputs"]} == {
        "line",
        "histogram",
    }
    ema = _nodes_with_kind(document, "ewma")
    assert len(ema) == 1
    assert _selected_names(document, "first") == [
        "entity",
        "time",
        "seq",
        "close",
        "a",
        "e",
    ]
    assert _selected_names(document, "second") == [
        "entity",
        "time",
        "seq",
        "close",
        "b",
    ]


def test_macd_shared_prefix_does_not_capture_an_unrelated_raw_output() -> None:
    source = _source()
    parts = cf.ts.macd_components(source["close"])
    program = cf.Program(
        "raw_fork",
        engine="sql",
        outputs={
            "first": source.with_columns(a=parts.line),
            "second": source.with_columns(b=parts.histogram),
            "raw": source,
        },
    )
    document = _document(program)
    assert len(_rolling_nodes(document)) == 1
    assert not any(
        edge["target_node"] == "raw" and "rolling" in edge["source_node"]
        for edge in document["graph"]["edges"]
    )


def test_macd_program_rejects_digest_collisions_before_shared_lowering() -> None:
    from dataclasses import replace

    source = _source()
    parts = cf.ts.macd_components(source["close"])
    collision = cf.ColumnExpr(
        replace(parts.histogram._node, digest=parts.line._node.digest)
    )
    with pytest.raises(ValueError, match="digest collision"):
        cf.Program(
            "collision",
            engine="sql",
            outputs={
                "first": source.with_columns(a=parts.line),
                "second": source.with_columns(b=collision),
            },
        )


def test_macd_shared_prefilter_has_one_writer_for_each_physical_input() -> None:
    source = _source()
    filtered = source.filter(source["close"] > 0.0)
    parts = cf.ts.macd_components(filtered["close"] + 1.0)
    program = cf.Program(
        "filtered_fork",
        engine="sql",
        outputs={
            "first": filtered.with_columns(
                a=parts.line, e=cf.ts.ema(filtered["close"] + 2.0, span=3)
            ),
            "second": filtered.with_columns(b=parts.histogram),
        },
    )
    document = _document(program)
    assert len(_nodes_with_kind(document, "macd_component")) == 1
    prefilters = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"].get("filter") is not None
    ]
    assert len(prefilters) == 1
    targets = [
        (edge["target_node"], edge["target_port"])
        for edge in document["graph"]["edges"]
    ]
    assert len(targets) == len(set(targets))


def test_macd_shared_planning_keeps_differently_filtered_contexts_separate() -> None:
    source = _source()
    positive = source.filter(source["close"] > 0.0)
    negative = source.filter(source["close"] < 0.0)
    first = cf.ts.macd_components(positive["close"])
    second = cf.ts.macd_components(negative["close"])
    program = cf.Program(
        "separate_contexts",
        engine="sql",
        outputs={
            "positive_line": positive.with_columns(a=first.line),
            "positive_hist": positive.with_columns(b=first.histogram),
            "negative_line": negative.with_columns(c=second.line),
            "negative_hist": negative.with_columns(d=second.histogram),
        },
    )
    assert len(_nodes_with_kind(_document(program), "macd_component")) == 2


def test_macd_shared_physical_ids_avoid_a_user_output_collision() -> None:
    source = _source()
    parts = cf.ts.macd_components(source["close"])
    first = source.with_columns(a=parts.line)
    second = source.with_columns(b=parts.histogram)
    ordinary = cf.Program(
        "ids", engine="sql", outputs={"first": first, "second": second}
    )
    reserved = _nodes_with_kind(_document(ordinary), "macd_component")[0]["id"]
    program = cf.Program(
        "reserved", engine="sql", outputs={reserved: first, "second": second}
    )
    document = _document(program)
    macd = _nodes_with_kind(document, "macd_component")
    assert len(macd) == 1
    assert macd[0]["id"] != reserved
    assert _selected_names(document, reserved) == [
        "entity",
        "time",
        "seq",
        "close",
        "a",
    ]


def test_macd_single_materialization_id_avoids_a_legal_program_output_name() -> None:
    source = _source()
    output = source.with_columns(line=cf.ts.macd_components(source["close"] + 1.0).line)
    original = _document(cf.Program("initial", engine="sql", outputs={"out": output}))
    owner = _nodes_with_kind(original, "macd_component")[0]["id"]
    reserved = next(
        edge["source_node"]
        for edge in original["graph"]["edges"]
        if edge["target_node"] == owner
    )
    document = _document(
        cf.Program("collision", engine="sql", outputs={reserved: output})
    )
    assert len(_nodes_with_kind(document, "macd_component")) == 1
    owner = _nodes_with_kind(document, "macd_component")[0]["id"]
    assert not _has_edge(document, reserved, owner)
    assert _selected_names(document, reserved) == [
        "entity",
        "time",
        "seq",
        "close",
        "line",
    ]
    ids = [node["id"] for node in document["graph"]["nodes"]]
    assert len(ids) == len(set(ids))


def _reachable(document: dict[str, object], source: str, target: str) -> bool:
    children: dict[str, list[str]] = {}
    for edge in document["graph"]["edges"]:
        children.setdefault(edge["source_node"], []).append(edge["target_node"])
    pending, seen = [source], set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current not in seen:
            seen.add(current)
            pending.extend(children.get(current, []))
    return False


@pytest.mark.parametrize(
    "kind,predicate", [("ewma", "e"), ("ewma", "close"), ("cross_section", "e")]
)
def test_macd_preserves_filters_after_upstream_state_before_its_inputs(
    kind: str, predicate: str
) -> None:
    source = _source()
    value = (
        cf.ts.ema(source["close"], span=3)
        if kind == "ewma"
        else cf.cs.rank(source["close"], group=cf.exact_time(source["time"]))
    )
    upstream = source.with_columns(e=value)
    positive = upstream.filter(upstream[predicate] > 0.0)
    negative = upstream.filter(upstream[predicate] < 0.0)
    first = cf.ts.macd_components(positive["close"])
    second = cf.ts.macd_components(negative["close"])
    program = cf.Program(
        "boundaries",
        engine="sql",
        outputs={
            "positive": positive.with_columns(line=first.line),
            "negative": negative.with_columns(histogram=second.histogram),
        },
    )
    assert program.analyze(cf.Runtime()).issues == ()
    document = _document(program)
    macd = _nodes_with_kind(document, "macd_component")
    assert len(macd) == 2
    filters = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"].get("filter") is not None
    ]
    assert len(filters) == 2
    for node in macd:
        assert (
            sum(_reachable(document, item["id"], node["id"]) for item in filters) == 1
        )
        assert not any(_reachable(document, node["id"], item["id"]) for item in filters)


def test_macd_computed_results_are_filtered_after_the_shared_owner() -> None:
    source = _source()
    parts = cf.ts.macd_components(source["close"])
    featured = source.with_columns(line=parts.line, histogram=parts.histogram)
    program = cf.Program(
        "post",
        engine="sql",
        outputs={
            "positive": featured.filter(featured["line"] > 0.0),
            "negative": featured.filter(featured["histogram"] < 0.0),
        },
    )
    document = _document(program)
    macd = _nodes_with_kind(document, "macd_component")
    assert len(macd) == 1
    filters = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"].get("filter") is not None
    ]
    assert len(filters) == 2
    assert all(_reachable(document, macd[0]["id"], node["id"]) for node in filters)


def test_macd_after_a_shared_stateful_filter_keeps_one_owner_and_ordered_schema() -> (
    None
):
    source = _source()
    upstream = source.with_columns(e=cf.ts.ema(source["close"], span=3))
    filtered = upstream.filter(upstream["e"] > 0.0)
    parts = cf.ts.macd_components(filtered["close"])
    program = cf.Program(
        "shared_filtered",
        engine="sql",
        outputs={
            "first": filtered.select(line=parts.line),
            "second": filtered.select(histogram=parts.histogram),
        },
    )
    document = _document(program)
    macd = _nodes_with_kind(document, "macd_component")
    assert len(macd) == 1
    fields = {field["name"]: field for field in macd[0]["input_ports"][0]["schema"]}
    assert fields["time"] == {
        "name": "time",
        "data_type": "timestamp[us, UTC]",
        "nullable": False,
    }
    assert fields["seq"] == {"name": "seq", "data_type": "uint64", "nullable": False}
    spec = macd[0]["operator"]["spec"]
    assert (spec["partition_by"], spec["event_time"], spec["sequence_by"]) == (
        ["entity"],
        "time",
        ["seq"],
    )
    filters = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"].get("filter") is not None
    ]
    assert len(filters) == 1
    assert _reachable(document, filters[0]["id"], macd[0]["id"])


def test_macd_constant_input_still_observes_its_evaluation_table_filter() -> None:
    source = _source()
    upstream = source.with_columns(e=cf.ts.ema(source["close"], span=3))
    filtered = upstream.filter(upstream["e"] > 0.0)
    result = cf.ts.macd_components(cf.lit(1.0)).line
    document = _document(
        cf.Program(
            "constant",
            engine="sql",
            outputs={"out": filtered.with_columns(result=result)},
        )
    )
    macd = _nodes_with_kind(document, "macd_component")
    filters = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"].get("filter") is not None
    ]
    assert len(macd) == len(filters) == 1
    assert _reachable(document, filters[0]["id"], macd[0]["id"])


def test_macd_boundary_batches_common_upstream_components_before_later_owner() -> None:
    source = _source()
    upstream = cf.ts.macd_components(source["close"])
    featured = source.with_columns(line=upstream.line)
    filtered = featured.filter(featured["line"] > 0.0)
    later = cf.ts.macd_components(filtered["close"], fast_span=1, slow_span=2)
    program = cf.Program(
        "macro_boundary",
        engine="sql",
        outputs={
            "later": filtered.select(later=later.histogram),
            "original": source.select(original=upstream.histogram),
        },
    )
    document = _document(program)
    macd = _nodes_with_kind(document, "macd_component")
    assert len(macd) == 2
    first = next(
        node
        for node in macd
        if node["operator"]["spec"]["outputs"][0]["fast_span"] == 12
    )
    second = next(
        node
        for node in macd
        if node["operator"]["spec"]["outputs"][0]["fast_span"] == 1
    )
    assert {item["component"] for item in first["operator"]["spec"]["outputs"]} == {
        "line",
        "histogram",
    }
    assert _reachable(document, first["id"], second["id"])


def _native_attachment(source: cf.TableExpr) -> cf.TableExpr:
    weights = cf.parameter(
        "weights", kind="array", backend="native", dtype="float64", shape=(1, 1)
    )
    values = cf.linalg.from_columns(source, columns=("close",), backend="native")
    return cf.table.attach_columns(
        source, cf.linalg.matmul(values, weights), names=("copy",)
    )


def _native_runtime() -> cf.Runtime:
    import calc_flow.array as cfa

    runtime = cf.Runtime()
    cfa.register_native(runtime)
    return runtime


@pytest.mark.parametrize("variant", ["base", "projection", "filtered", "derived"])
def test_macd_native_materialized_boundary_keeps_proven_column_meaning(
    variant: str,
) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    if variant == "filtered":
        source = source.filter(source["close"] > 0.0)
    target = _native_attachment(source)
    operand = source["close"]
    if variant == "projection":
        target = target.select("entity", "time", "seq", "close", "copy")
    if variant == "derived":
        target = target.with_columns(adjusted=source["close"] + 1.0)
        operand = target["adjusted"]
    parts = cf.ts.macd_components(operand)
    program = cf.Program(
        "native_boundary",
        engine="sql",
        outputs={
            "line": target.select(line=parts.line),
            "histogram": target.select(histogram=parts.histogram),
        },
    )
    runtime = _native_runtime()
    assert program.analyze(runtime).issues == ()
    document = lower_program_document(program, runtime, "batch")
    assert len(_nodes_with_kind(document, "macd_component")) == 1


@pytest.mark.parametrize("mismatch", ["definition", "filtered_rows"])
def test_macd_native_boundary_rejects_unproven_inherited_references(
    mismatch: str,
) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    if mismatch == "definition":
        declared = source.with_columns(adjusted=source["close"] + 1.0)
        actual = source.with_columns(adjusted=source["close"] + 2.0)
        operand = declared["adjusted"]
    else:
        declared = source.filter(source["close"] < 0.0)
        actual = source.filter(source["close"] > 0.0)
        operand = declared["close"]
    target = _native_attachment(actual)
    program = cf.Program(
        "native_mismatch",
        engine="sql",
        outputs={
            "out": target.with_columns(result=cf.ts.macd_components(operand).line),
        },
    )
    runtime = _native_runtime()
    assert any(
        issue.code == "schema_mismatch" for issue in program.analyze(runtime).issues
    )
    with pytest.raises(cf.CompileError, match="schema_mismatch"):
        lower_program_document(program, runtime, "batch")


def test_real_macd_named_outputs_match_legacy_composition_without_hidden_columns() -> (
    None
):
    import pyarrow as pa

    source = _source()
    parts = cf.ts.macd_components(
        source["close"], fast_span=2, slow_span=4, signal_span=2, signal_min_periods=2
    )
    program = cf.Program(
        "live",
        engine="sql",
        outputs={
            "line": source.select(line=parts.line),
            "histogram": source.select(histogram=parts.histogram),
        },
    )
    data = pa.table(
        {
            "entity": ["a"] * 4,
            "time": pa.array([0, 1, 2, 3], type=pa.timestamp("us", tz="UTC")),
            "seq": pa.array([0, 1, 2, 3], type=pa.uint64()),
            "close": [1.0, 2.0, None, 3.0],
        },
        schema=pa.schema(
            [
                pa.field("entity", pa.string(), nullable=False),
                pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("close", pa.float64()),
            ]
        ),
    )
    actual = program.collect({"quotes": data})
    line = cf.ts.macd(source["close"], fast_span=2, slow_span=4)
    signal = cf.ts.ema(line, span=2, min_periods=2)
    expected = cf.Program(
        "reference",
        engine="sql",
        outputs={
            "out": source.select(line=line, histogram=line - signal),
        },
    ).collect({"quotes": data})["out"]
    assert actual["line"].column_names == ["line"]
    assert actual["histogram"].column_names == ["histogram"]
    assert actual["line"]["line"].to_pylist() == expected["line"].to_pylist()
    assert (
        actual["histogram"]["histogram"].to_pylist()
        == expected["histogram"].to_pylist()
    )


def test_macd_subset_and_unused_container_emit_only_consumed_components() -> None:
    source = _source()
    parts = cf.ts.macd_components(source["close"])
    subset = cf.Program(
        "subset", engine="sql", outputs={"out": source.with_columns(h=parts.histogram)}
    )
    declarations = _rolling_nodes(_document(subset))[0]["operator"]["spec"]["outputs"]
    assert len(declarations) == 1
    assert declarations[0]["component"] == "histogram"
    unused = cf.Program("unused", engine="sql", outputs={"out": source})
    assert _rolling_nodes(_document(unused)) == []


def test_macd_mixed_named_outputs_use_layout_six_regardless_of_first_plan() -> None:
    source = _source()
    parts = cf.ts.macd_components(source["close"])
    program = cf.Program(
        "mixed",
        engine="sql",
        outputs={
            "first_old": source.with_columns(ema=cf.ts.ema(source["close"], span=3)),
            "second_rsi": source.with_columns(r=cf.ts.rsi(source["close"], period=2)),
            "third_macd": source.with_columns(h=parts.histogram),
        },
    )
    nodes = _rolling_nodes(_document(program))
    assert len(nodes) == 1
    assert nodes[0]["operator"]["spec"]["state_layout_version"] == 6
    assert {item["kind"] for item in nodes[0]["operator"]["spec"]["outputs"]} == {
        "ewma",
        "rsi",
        "macd_component",
    }


def test_macd_group_identity_keeps_all_options_and_ignores_component_and_alias() -> (
    None
):
    from calc_flow.symbolic.optimizer import _rolling_group_key

    declaration = {
        "kind": "macd_component",
        "primitive_version": 1,
        "input": "close",
        "component": "line",
        "output": "first",
        "fast_span": 12,
        "slow_span": 26,
        "signal_span": 9,
        "min_periods": 1,
        "signal_min_periods": 1,
    }
    key = _rolling_group_key(declaration)
    assert key is not None
    assert key == _rolling_group_key(
        {**declaration, "component": "histogram", "output": "second"}
    )
    for name in (
        "fast_span",
        "slow_span",
        "signal_span",
        "min_periods",
        "signal_min_periods",
        "primitive_version",
    ):
        assert key != _rolling_group_key({**declaration, name: declaration[name] + 1})
    assert key != _rolling_group_key({**declaration, "input": "other"})


def test_macd_late_output_accepts_three_components_as_one_stage() -> None:
    source = _source()
    parts = cf.ts.macd_components(source["close"])
    outputs = cf.with_late_output(
        source.with_columns(
            line=parts.line, signal=parts.signal, histogram=parts.histogram
        )
    )
    program = cf.Program(
        "late",
        engine="streaming",
        outputs={"out": outputs.output, "late": outputs.late},
    )
    assert program.analyze(cf.Runtime()).issues == ()
    nodes = _rolling_nodes(_document(program, "stream"))
    assert len(nodes) == 1
    assert {port["name"] for port in nodes[0]["output_ports"]} == {"output", "late"}


def test_macd_lowerer_rejects_foreign_definitions_before_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calc_flow.symbolic.lower import segments

    source = _source()
    foreign = source.filter(source["close"] > 0.0)
    output = source.with_columns(h=cf.ts.macd_components(foreign["close"]).histogram)

    def unexpected_inline(*_arguments: object) -> None:
        raise AssertionError("invalid MACD source reached inline")

    monkeypatch.setattr(segments, "_inline", unexpected_inline)
    with pytest.raises(cf.CompileError, match="same row set"):
        segments._resolve_table(output._node, "out")


@pytest.mark.parametrize("projection", [False, True])
def test_macd_owner_is_forwarded_through_native_attachment(projection: bool) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    parts = cf.ts.macd_components(source["close"] + 1.0)
    first = source.with_columns(line=parts.line)
    attached = _native_attachment(first)
    if projection:
        attached = attached.select("entity", "time", "seq", "close", "copy")
    program = cf.Program(
        "forward",
        engine="sql",
        outputs={
            "before": first.select("line"),
            "after": attached.with_columns(histogram=parts.histogram).select(
                "copy", "histogram"
            ),
        },
    )
    runtime = _native_runtime()
    assert program.analyze(runtime).issues == ()
    document = lower_program_document(program, runtime, "batch")
    owners = _nodes_with_kind(document, "macd_component")
    assert len(owners) == 1
    assert {item["component"] for item in owners[0]["operator"]["spec"]["outputs"]} == {
        "line",
        "histogram",
    }
    assert _selected_names(document, "before") == ["line"]
    assert _selected_names(document, "after") == ["copy", "histogram"]
    native = _native_nodes(document)
    assert len(native) == 1
    _assert_private_fields_forwarded(native[0])
    assert _reachable(document, owners[0]["id"], native[0]["id"])


def _native_nodes(document: dict[str, object]) -> list[dict[str, object]]:
    return [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "external"
    ]


def _assert_private_fields_forwarded(native: dict[str, object]) -> None:
    bridge_fields = {field["name"] for field in native["output_ports"][0]["schema"]}
    hidden = {name for name in bridge_fields if name.startswith("cf_macd_value_")}
    assert len(hidden) == 2
    table_input = next(
        port for port in native["input_ports"] if port["kind"] == "table"
    )
    assert hidden <= {field["name"] for field in table_input["schema"]}


def test_macd_parallel_native_boundaries_share_only_the_same_input_meaning() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    first = _native_attachment(source)
    second = _native_attachment(source.select("entity", "time", "seq", "close"))
    parts = cf.ts.macd_components(source["close"])
    program = cf.Program(
        "parallel",
        engine="sql",
        outputs={
            "first": first.select(a=parts.line),
            "second": second.select(b=parts.histogram),
        },
    )
    document = lower_program_document(program, _native_runtime(), "batch")
    assert len(_nodes_with_kind(document, "macd_component")) == 1
    assert _selected_names(document, "first") == ["a"]
    assert _selected_names(document, "second") == ["b"]


def test_macd_native_forwarding_keeps_different_filtered_owners_separate() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    positive = source.filter(source["close"] > 0.0)
    negative = source.filter(source["close"] < 0.0)
    first = _native_attachment(positive)
    second = _native_attachment(negative)
    program = cf.Program(
        "filtered",
        engine="sql",
        outputs={
            "first": first.select(a=cf.ts.macd_components(positive["close"]).line),
            "second": second.select(
                b=cf.ts.macd_components(negative["close"]).histogram
            ),
        },
    )
    document = lower_program_document(program, _native_runtime(), "batch")
    assert len(_nodes_with_kind(document, "macd_component")) == 2
    assert _selected_names(document, "first") == ["a"]
    assert _selected_names(document, "second") == ["b"]


def test_macd_native_forwarding_shares_its_upstream_state_once() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    ema = cf.ts.ema(source["close"], span=3)
    parts = cf.ts.macd_components(ema)
    first = source.with_columns(average=ema, line=parts.line)
    target = _native_attachment(first)
    program = cf.Program(
        "nested_forward",
        engine="sql",
        outputs={
            "first": first.select("average", "line"),
            "last": target.select(histogram=parts.histogram),
        },
    )
    document = lower_program_document(program, _native_runtime(), "batch")
    averages = _nodes_with_kind(document, "ewma")
    owners = _nodes_with_kind(document, "macd_component")
    assert len(averages) == len(owners) == 1
    assert _reachable(document, averages[0]["id"], owners[0]["id"])
    assert _selected_names(document, "first") == ["average", "line"]
    assert _selected_names(document, "last") == ["histogram"]


def test_macd_native_forwarding_uses_definitions_instead_of_only_field_names() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    first = source.with_columns(adjusted=source["close"] + 1.0)
    second = source.with_columns(adjusted=source["close"] + 2.0)
    a = cf.ts.macd_components(first["adjusted"])
    b = cf.ts.macd_components(second["adjusted"])
    before = first.with_columns(line=a.line)
    after = _native_attachment(before)
    different = _native_attachment(second)
    program = cf.Program(
        "definitions",
        engine="sql",
        outputs={
            "first": after.select(histogram=a.histogram),
            "second": different.select(line=b.line),
        },
    )
    document = lower_program_document(program, _native_runtime(), "batch")
    assert len(_nodes_with_kind(document, "macd_component")) == 2
    assert _selected_names(document, "first") == ["histogram"]
    assert _selected_names(document, "second") == ["line"]


def test_macd_forwarding_does_not_expose_private_columns_to_sql() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    parts = cf.ts.macd_components(source["close"])
    first = source.with_columns(line=parts.line)
    target = _native_attachment(first)
    sql = target.sql("SELECT * FROM input")
    program = cf.Program(
        "sql_boundary",
        engine="sql",
        outputs={
            "first": first.select("line"),
            "second": target.select(histogram=parts.histogram),
            "sql": sql,
        },
    )
    document = lower_program_document(program, _native_runtime(), "batch")
    assert len(_nodes_with_kind(document, "macd_component")) == 1
    assert _selected_names(document, "sql") == [
        "entity",
        "time",
        "seq",
        "close",
        "line",
        "copy",
    ]
    sql_node = next(
        node for node in document["graph"]["nodes"] if node["operator"]["kind"] == "sql"
    )
    assert [field["name"] for field in sql_node["input_ports"][0]["schema"]] == [
        "entity",
        "time",
        "seq",
        "close",
        "line",
        "copy",
    ]


def test_macd_forwarding_reserves_hidden_names_against_actual_user_fields() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    parts = cf.ts.macd_components(source["close"])
    first = source.with_columns(line=parts.line)
    attached = _native_attachment(first)
    runtime = _native_runtime()
    initial = cf.Program(
        "names",
        engine="sql",
        outputs={
            "before": first.select("line"),
            "after": attached.select(histogram=parts.histogram),
        },
    )
    initial_document = lower_program_document(initial, runtime, "batch")
    hidden = next(
        item["output"]
        for item in _nodes_with_kind(initial_document, "macd_component")[0]["operator"][
            "spec"
        ]["outputs"]
        if item["component"] == "line"
    )
    user = source.with_columns({hidden: source["close"] + 7.0})
    before = user.with_columns(line=parts.line)
    target = _native_attachment(before)
    program = cf.Program(
        "names",
        engine="sql",
        outputs={
            "before": before.select("line", hidden),
            "after": target.select(histogram=parts.histogram),
        },
    )
    document = lower_program_document(program, runtime, "batch")
    owners = _nodes_with_kind(document, "macd_component")
    assert len(owners) == 1
    assert hidden not in {
        item["output"] for item in owners[0]["operator"]["spec"]["outputs"]
    }
    assert _selected_names(document, "before") == ["line", hidden]
    assert _selected_names(document, "after") == ["histogram"]


def test_macd_forwarding_preserves_a_stateful_filtered_input_and_postfilter() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    average = source.with_columns(average=cf.ts.ema(source["close"], span=3))
    filtered = average.filter(average["close"] > 0.0)
    parts = cf.ts.macd_components(filtered["close"])
    first = filtered.with_columns(line=parts.line)
    attached = _native_attachment(first)
    result = attached.with_columns(histogram=parts.histogram)
    result = result.filter(result["histogram"] > 0.0)
    program = cf.Program(
        "filter_forward",
        engine="sql",
        outputs={
            "first": first.select("line"),
            "last": result.select("histogram"),
        },
    )
    document = lower_program_document(program, _native_runtime(), "batch")
    owners = _nodes_with_kind(document, "macd_component")
    assert len(owners) == 1
    filters = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "expression"
        and node["operator"]["filter"] is not None
    ]
    assert len(filters) == 2
    assert any(_reachable(document, node["id"], owners[0]["id"]) for node in filters)
    assert any(_reachable(document, owners[0]["id"], node["id"]) for node in filters)
    assert _selected_names(document, "first") == ["line"]
    assert _selected_names(document, "last") == ["histogram"]


def test_macd_native_forwarding_prunes_unconsumed_component_declarations() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    parts = cf.ts.macd_components(source["close"])
    first = source.with_columns(line=parts.line)
    first = first.select("entity", "time", "seq", "close")
    attached = _native_attachment(first)
    last = attached.with_columns(histogram=parts.histogram).select("copy")
    program = cf.Program("unused", engine="sql", outputs={"out": last})
    document = lower_program_document(program, _native_runtime(), "batch")
    assert _nodes_with_kind(document, "macd_component") == []
    assert _selected_names(document, "out") == ["copy"]


def test_macd_forwarding_keeps_the_explicit_late_owner_context_separate() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _source()
    parts = cf.ts.macd_components(source["close"])
    first = source.with_columns(line=parts.line)
    attached = _native_attachment(first)
    late = cf.with_late_output(first, allowed_lateness_micros=7)
    program = cf.Program(
        "late_forward",
        engine="streaming",
        outputs={
            "before": first.select("line"),
            "after": attached.select(histogram=parts.histogram),
            "main": late.output.select("line"),
            "late": late.late,
        },
    )
    runtime = _native_runtime()
    assert program.analyze(runtime, mode="stream").issues == ()
    document = lower_program_document(program, runtime, "stream")
    owners = _nodes_with_kind(document, "macd_component")
    assert len(owners) == 2
    plain = next(
        node
        for node in owners
        if node["operator"]["spec"]["late_policy"]["kind"] == "error"
    )
    paired = next(
        node
        for node in owners
        if node["operator"]["spec"]["late_policy"]["kind"] == "side_output"
    )
    assert plain["operator"]["spec"]["allowed_lateness_micros"] == 0
    assert paired["operator"]["spec"]["allowed_lateness_micros"] == 7
    assert {port["name"] for port in paired["output_ports"]} == {"output", "late"}
    assert _reachable(document, plain["id"], "before")
    assert _reachable(document, plain["id"], "after")
    assert _reachable(document, paired["id"], "main")
    assert _reachable(document, paired["id"], "late")
    assert not _reachable(document, plain["id"], paired["id"])
    assert _selected_names(document, "before") == ["line"]
    assert _selected_names(document, "after") == ["histogram"]
    assert _selected_names(document, "main") == ["line"]
    assert not any(
        name.startswith("cf_macd_value_") for name in _selected_names(document, "late")
    )
