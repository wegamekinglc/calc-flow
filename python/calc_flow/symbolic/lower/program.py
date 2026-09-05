"""The row-local program orchestrator and compile entry points.

Deterministic lowering of symbolic programs of symbolic programs to strict project-v3.

The lowerer resolves each declared table output into one fused row-local
segment, renders the segment as DataFusion SQL inside strict project-v3
``expression`` nodes, and hands the document to the existing Rust graph
compiler for final port, schema, topology, and fingerprint validation. No data
object, source, sink, or runner is accepted here, and no symbolic Python runs
while a compiled plan executes.
"""

from __future__ import annotations

from calc_flow.capabilities import RuntimeCapabilities
from calc_flow.pipeline import (
    BatchExecutionPlan,
    Runtime,
    StreamExecutionPlan,
    StreamRequirements,
    _canonical,
)
from calc_flow.symbolic import errors
from calc_flow.symbolic.analyzer import (
    _Analyzer,
    _require_mode,
    _run,
    _schema_fields,
)
from calc_flow.symbolic.domains import type_name
from calc_flow.symbolic.lower.planners import (
    _check_declared_inputs,
    _CrossSectionPlan,
    _LoweringProgram,
    _plan_cross_section,
    _plan_rolling,
    _share_cross_section_plans,
    _share_rolling_plans,
)
from calc_flow.symbolic.lower.segments import (
    _TABLE_OUTPUT_PRIMITIVES,
    _U64_MAX,
    _CompileCacheKey,
    _cstr,
    _expression_node,
    _find_cross_section,
    _find_rolling,
    _quote_identifier,
    _reject_primitive,
    _resolve_table,
    _Segment,
    _select_item,
    _sql,
)
from calc_flow.symbolic.lower.strategies import (
    _deduplicate_node_ids,
    _deduplicate_pure_expression_nodes,
    _lower_matrix_program,
    _lower_stream_join_program,
    _project_document,
    _required_segment_state_plan,
    _stream_join_nodes,
)
from calc_flow.symbolic.nodes import (
    build,
)
from calc_flow.symbolic.optimizer import expression_refs, extract_common
from calc_flow.symbolic.program import Program


# The lowerer keeps per-output segment staging in one deterministic pass:
# stage order, edge wiring, and id assignment are semantic, so the rolling,
# prefilter, and CSE stages stay in one place.
def _lower_program(
    program: Program | _LoweringProgram,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object]:
    # #lizard forgives
    matrix_project = _lower_matrix_program(
        program,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    if matrix_project is not None:
        return matrix_project
    _check_declared_inputs(program)
    segments = []
    for output_name, value in program.outputs:
        node = value._node
        path = f"outputs.{output_name}"
        if node.op.name not in _TABLE_OUTPUT_PRIMITIVES:
            _reject_primitive(path, node)
        segments.append((output_name, _resolve_table(node, path)))
    consumed = {segment.input_node.digest for _, segment in segments}
    multi_output_lineages = {
        digest
        for digest in consumed
        if sum(1 for _, segment in segments if segment.input_node.digest == digest) > 1
    }
    fanout = len(program.inputs) > 1 or bool(multi_output_lineages)
    plans = {
        output_name: _plan_rolling(
            output_name,
            segment,
            f"outputs.{output_name}",
            allowed_lateness_micros,
            late_policy,
        )
        for output_name, segment in segments
    }
    plans = _share_rolling_plans(segments, plans)
    rolling_digests = {
        segment.input_node.digest
        for (output_name, segment), plan in zip(segments, plans.values(), strict=True)
        if plan is not None
    }
    cross_plans: dict[str, _CrossSectionPlan | None] = {}
    cross_segments: list[tuple[str, _Segment]] = []
    for (output_name, segment), rolling in zip(segments, plans.values(), strict=True):
        if rolling is not None:
            # Cross-section planning runs over the rolling-rewritten
            # environment so a measured value may be a materialized rolling
            # output column.
            after_rolling = _Segment(
                segment.input_node,
                segment.fields,
                rolling.env,
                None,
                rolling.post_predicate,
            )
            cross_segments.append((output_name, after_rolling))
            cross_plans[output_name] = _plan_cross_section(
                output_name,
                after_rolling,
                f"outputs.{output_name}",
                allowed_lateness_micros,
                late_policy,
                rolling.output_fields,
            )
        else:
            cross_segments.append((output_name, segment))
            cross_plans[output_name] = _plan_cross_section(
                output_name,
                segment,
                f"outputs.{output_name}",
                allowed_lateness_micros,
                late_policy,
                None,
            )
    cross_plans = _share_cross_section_plans(
        cross_segments,
        {
            output_name: None
            if plans[output_name] is None
            else plans[output_name].node_id
            for output_name, _ in segments
        },
        cross_plans,
    )
    direct_cross_section_digests = {
        segment.input_node.digest
        for (output_name, segment), rolling, cross in zip(
            segments, plans.values(), cross_plans.values(), strict=True
        )
        if rolling is None and cross is not None
    }
    shared_plan_counts: dict[str, int] = {}
    for plan in (*plans.values(), *cross_plans.values()):
        if plan is not None:
            shared_plan_counts[plan.node_id] = (
                shared_plan_counts.get(plan.node_id, 0) + 1
            )
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    fanout_ids: dict[str, str] = {}
    if fanout:
        for value in program.inputs:
            input_node = value._node
            if input_node.digest not in consumed:
                continue
            input_name = _cstr(input_node.attr("name"))
            schema = _schema_fields(input_node.attr("schema"))
            pinned = (
                input_node.digest in rolling_digests
                or input_node.digest in direct_cross_section_digests
            )
            nodes.append(
                _expression_node(
                    input_name,
                    [_quote_identifier(field.name) for field in schema],
                    None,
                    schema,
                    schema if pinned else None,
                )
            )
            fanout_ids[input_node.digest] = input_name
    for output_name, segment in segments:
        rolling = plans[output_name]
        cross = cross_plans[output_name]
        env = dict(segment.env)
        input_field_names = [
            field.name for field in _schema_fields(segment.input_node.attr("schema"))
        ]
        upstream_id: str | None = None
        final_predicate = segment.predicate
        if (rolling is not None or cross is not None) and segment.predicate is not None:
            state_plan = _required_segment_state_plan(rolling, cross)
            prefilter_id = (
                f"{state_plan.node_id}__prefilter"
                if shared_plan_counts[state_plan.node_id] > 1
                else f"{output_name}__cf_prefilter"
            )
            input_fields = _schema_fields(segment.input_node.attr("schema"))
            nodes.append(
                _expression_node(
                    prefilter_id,
                    [_quote_identifier(name) for name in input_field_names],
                    _sql(segment.predicate),
                    input_fields,
                    input_fields,
                )
            )
            if fanout:
                edges.append(
                    {
                        "source_node": fanout_ids[segment.input_node.digest],
                        "source_port": "output",
                        "target_node": prefilter_id,
                        "target_port": "input",
                    }
                )
            upstream_id = prefilter_id
        if rolling is not None:
            for stage in rolling.stages:
                if stage.materialization_node is not None:
                    nodes.append(stage.materialization_node)
                    materialization_id = stage.materialization_node_id
                    if materialization_id is None:
                        raise RuntimeError("rolling materialization node has no id")
                    if upstream_id is not None:
                        edges.append(
                            {
                                "source_node": upstream_id,
                                "source_port": "output",
                                "target_node": materialization_id,
                                "target_port": "input",
                            }
                        )
                    elif fanout:
                        edges.append(
                            {
                                "source_node": fanout_ids[segment.input_node.digest],
                                "source_port": "output",
                                "target_node": materialization_id,
                                "target_port": "input",
                            }
                        )
                    upstream_id = materialization_id
                nodes.append(stage.node)
                if upstream_id is not None:
                    edges.append(
                        {
                            "source_node": upstream_id,
                            "source_port": "output",
                            "target_node": stage.node_id,
                            "target_port": "input",
                        }
                    )
                elif fanout:
                    edges.append(
                        {
                            "source_node": fanout_ids[segment.input_node.digest],
                            "source_port": "output",
                            "target_node": stage.node_id,
                            "target_port": "input",
                        }
                    )
                upstream_id = stage.node_id
            env = dict(rolling.env)
            input_field_names = list(rolling.input_field_names)
            final_predicate = rolling.post_predicate
        if cross is not None:
            if cross.materialization_node is not None:
                nodes.append(cross.materialization_node)
                materialization_id = cross.materialization_node_id
                if materialization_id is None:
                    raise RuntimeError("cross-section materialization node has no id")
                if upstream_id is not None:
                    edges.append(
                        {
                            "source_node": upstream_id,
                            "source_port": "output",
                            "target_node": materialization_id,
                            "target_port": "input",
                        }
                    )
                elif fanout:
                    edges.append(
                        {
                            "source_node": fanout_ids[segment.input_node.digest],
                            "source_port": "output",
                            "target_node": materialization_id,
                            "target_port": "input",
                        }
                    )
                upstream_id = materialization_id
            nodes.append(cross.node)
            if upstream_id is not None:
                edges.append(
                    {
                        "source_node": upstream_id,
                        "source_port": "output",
                        "target_node": cross.node_id,
                        "target_port": "input",
                    }
                )
            elif fanout:
                edges.append(
                    {
                        "source_node": fanout_ids[segment.input_node.digest],
                        "source_port": "output",
                        "target_node": cross.node_id,
                        "target_port": "input",
                    }
                )
            upstream_id = cross.node_id
            env = dict(cross.env)
            input_field_names = list(cross.input_field_names)
            final_predicate = cross.post_predicate
        elif rolling is None and segment.post_predicate is not None:
            final_predicate = (
                segment.post_predicate
                if segment.predicate is None
                else build("and", (segment.predicate, segment.post_predicate), {})
            )
        reserved = frozenset(env)
        fused = extract_common(
            tuple((field, env[field]) for field in segment.fields),
            final_predicate,
            reserved,
        )
        cse_order = [name for tier in fused.tiers for name, _ in tier]
        needed: set[str] = set()
        for _, tree in fused.selects:
            needed |= expression_refs(tree)
        if fused.predicate is not None:
            needed |= expression_refs(fused.predicate)
        tier_items: list[list[str]] = []
        for index in range(len(fused.tiers) - 1, -1, -1):
            tier = fused.tiers[index]
            defined = {name for name, _ in tier}
            passthrough = needed - defined
            items = [
                _quote_identifier(field)
                for field in input_field_names
                if field in passthrough
            ]
            items += [
                _quote_identifier(name) for name in cse_order if name in passthrough
            ]
            items += [
                f"{_sql(tree)} AS {_quote_identifier(name)}" for name, tree in tier
            ]
            tier_items.append(items)
            needed = set(passthrough)
            for _, tree in tier:
                needed |= expression_refs(tree)
        tier_items.reverse()
        stage_ids = [
            f"{output_name}__cf_cse_{index}" for index in range(1, len(fused.tiers) + 1)
        ] + [output_name]
        stage_selects = [
            *tier_items,
            [_select_item(field, tree) for field, tree in fused.selects],
        ]
        for position, (node_id, select) in enumerate(
            zip(stage_ids, stage_selects, strict=True)
        ):
            filter_sql = None
            if position == len(stage_ids) - 1 and fused.predicate is not None:
                filter_sql = _sql(fused.predicate)
            if position == 0:
                if upstream_id is not None:
                    edges.append(
                        {
                            "source_node": upstream_id,
                            "source_port": "output",
                            "target_node": node_id,
                            "target_port": "input",
                        }
                    )
                    input_schema = (
                        cross.output_fields
                        if cross is not None
                        else rolling.output_fields
                        if rolling is not None
                        else None
                    )
                elif fanout:
                    edges.append(
                        {
                            "source_node": fanout_ids[segment.input_node.digest],
                            "source_port": "output",
                            "target_node": node_id,
                            "target_port": "input",
                        }
                    )
                    input_schema = None
                else:
                    input_schema = _schema_fields(segment.input_node.attr("schema"))
            else:
                edges.append(
                    {
                        "source_node": stage_ids[position - 1],
                        "source_port": "output",
                        "target_node": node_id,
                        "target_port": "input",
                    }
                )
                input_schema = None
            nodes.append(_expression_node(node_id, select, filter_sql, input_schema))
    nodes, edges = _deduplicate_node_ids(nodes, edges)
    nodes, edges = _deduplicate_pure_expression_nodes(
        nodes,
        edges,
        frozenset(output_name for output_name, _ in program.outputs),
    )
    return _project_document(program.name, mode, nodes, edges)


def _require_runtime(runtime: object, entry: str, /) -> Runtime:
    if not isinstance(runtime, Runtime):
        raise TypeError(
            f"{entry} requires an explicit calc_flow Runtime; got {type_name(runtime)}"
        )
    return runtime


def _check_expression_capability(
    program: Program,
    runtime: Runtime,
    mode: str,
    /,
) -> tuple[_Analyzer, RuntimeCapabilities]:
    analyzer, capabilities = _run(program, runtime, mode)
    issues = analyzer.issues
    if issues:
        first = issues[0]
        errors.raise_compile(first.path, first.code, first.message)
    for operator in capabilities.operators:
        if operator.kind != "expression":
            continue
        _require_mode_support(program, operator, mode)
        _require_stream_facts(program, operator, mode)
        return analyzer, capabilities
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer the expression operator",
    )


def _require_mode_support(program: Program, operator: object, mode: str, /) -> None:
    if mode not in operator.modes:  # type: ignore[attr-defined]
        errors.raise_compile(
            program.name,
            errors.CAPABILITY_MISMATCH,
            f"the expression operator does not support {mode} mode in the"
            " selected capability snapshot",
        )


def _require_stream_facts(program: Program, operator: object, mode: str, /) -> None:
    if mode == "stream" and (
        operator.finality == "unproven"  # type: ignore[attr-defined]
        or not operator.microbatch_invariant  # type: ignore[attr-defined]
        or not operator.deterministic  # type: ignore[attr-defined]
        or not operator.replay_safe  # type: ignore[attr-defined]
    ):
        errors.raise_compile(
            program.name,
            errors.CAPABILITY_MISMATCH,
            "the expression operator does not prove stream lifecycle facts"
            " in the selected capability snapshot",
        )


def _program_needs_rolling(program: Program, /) -> bool:
    return any(True for _, value in program.outputs for _ in _find_rolling(value._node))


def _program_needs_cross_section(program: Program, /) -> bool:
    return any(
        True for _, value in program.outputs for _ in _find_cross_section(value._node)
    )


def _program_needs_stream_join(program: Program, /) -> bool:
    return bool(_stream_join_nodes(program))


def _stream_join_ports(operator: object, /) -> bool:
    inputs = tuple(
        (port.name, port.kind, port.required) for port in operator.input_ports
    )
    outputs = tuple(
        (port.name, port.kind, port.required) for port in operator.output_ports
    )
    return inputs == (("left", "table", True), ("right", "table", True)) and (
        outputs == (("output", "table", True),)
    )


def _positive_state_version(value: object, /) -> bool:
    return isinstance(value, int) and value > 0


def _stream_join_capability_facts(operator: object, mode: str, /) -> tuple[bool, ...]:
    return (
        operator.version == "1",
        mode in operator.modes,
        _stream_join_ports(operator),
        operator.requires_watermark,
        operator.stateful,
        operator.checkpoint_support == "checkpointed_stateful",
        _positive_state_version(operator.state_version),
        operator.deterministic,
        operator.replay_safe,
    )


def _check_stream_join_capability(
    program: Program,
    capabilities: RuntimeCapabilities,
    mode: str,
    /,
) -> None:
    for operator in capabilities.operators:
        if operator.kind != "stream_join":
            continue
        if not all(_stream_join_capability_facts(operator, mode)):
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                "the stream_join operator does not prove the required stream"
                " ports, watermark, checkpoint, determinism, and replay facts",
            )
        return
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer stream_join@1",
    )


# The cross-section gate mirrors the rolling gate over the frozen
# group-final stream lifecycle facts.
def _cross_section_checkpoint_capability(operator: object, /) -> bool:
    return (
        operator.stateful
        and operator.checkpoint_support == "checkpointed_stateful"
        and isinstance(operator.state_version, int)
        and operator.state_version > 0
    )


def _cross_section_stream_capability(operator: object, /) -> bool:
    return (
        operator.finality != "unproven"
        and operator.microbatch_invariant
        and _cross_section_checkpoint_capability(operator)
        and operator.deterministic
        and operator.replay_safe
    )


def _check_cross_section_capability(
    program: Program,
    capabilities: object,
    mode: str,
    /,
) -> None:
    for operator in capabilities.operators:
        if operator.kind != "cross_section":
            continue
        if mode not in operator.modes:
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                f"the cross-section operator does not support {mode} mode in"
                " the selected capability snapshot",
            )
        if mode == "stream" and not _cross_section_stream_capability(operator):
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                "the cross-section operator does not prove stream lifecycle"
                " facts in the selected capability snapshot",
            )
        return
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer the cross-section operator",
    )


# The capability gate conjoins the frozen stream lifecycle facts; every
# fact fails with the same stable capability_mismatch code.
def _check_rolling_capability(
    program: Program,
    capabilities: object,
    mode: str,
    /,
) -> None:
    # #lizard forgives
    for operator in capabilities.operators:
        if operator.kind != "rolling":
            continue
        if mode not in operator.modes:
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                f"the rolling operator does not support {mode} mode in the"
                " selected capability snapshot",
            )
        if mode == "stream" and (
            operator.finality == "unproven"
            or not operator.stateful
            or not operator.microbatch_invariant
            or operator.checkpoint_support != "checkpointed_stateful"
            or not isinstance(operator.state_version, int)
            or operator.state_version <= 0
            or not operator.deterministic
            or not operator.replay_safe
        ):
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                "the rolling operator does not prove stream lifecycle facts"
                " in the selected capability snapshot",
            )
        return
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer the rolling operator",
    )


def lower_program_document(
    program: Program,
    runtime: Runtime,
    mode: str,
    /,
    *,
    allowed_lateness_micros: int = 0,
    late_policy: str = "error",
) -> dict[str, object]:
    """Analyze and lower one program to its strict project-v3 document.

    The lateness arguments are validated whenever the program contains
    rolling or cross-section primitives; row-local programs do not consume
    them.
    """

    selected = _require_runtime(runtime, "lower_program_document")
    mode_value = _require_mode(mode)
    analyzer, capabilities = _check_expression_capability(program, selected, mode_value)
    if _program_needs_stream_join(program):
        _check_stream_join_capability(program, capabilities, mode_value)
    if _program_needs_rolling(program) or _program_needs_cross_section(program):
        _validate_lateness(allowed_lateness_micros, late_policy)
        if _program_needs_rolling(program):
            _check_rolling_capability(program, capabilities, mode_value)
        if _program_needs_cross_section(program):
            _check_cross_section_capability(program, capabilities, mode_value)
    join_project = _lower_stream_join_program(
        program,
        analyzer,
        mode_value,
        allowed_lateness_micros,
        late_policy,
    )
    if join_project is not None:
        return join_project
    return _lower_program(program, mode_value, allowed_lateness_micros, late_policy)


def _cache_graph_nodes(document: dict[str, object], /) -> list[dict[str, object]]:
    graph = document["graph"]
    return graph["nodes"]  # type: ignore[index,return-value]


def _cache_operator_versions(
    nodes: list[dict[str, object]], capabilities: RuntimeCapabilities, /
) -> tuple[tuple[str, str], ...]:
    operator_kinds = {
        node["operator"]["kind"]  # type: ignore[index]
        for node in nodes
        if node["operator"]["kind"] != "external"  # type: ignore[index]
    }
    return tuple(
        sorted(
            (operator.kind, operator.version)
            for operator in capabilities.operators
            if operator.kind in operator_kinds
        )
    )


def _cache_provider_versions(
    nodes: list[dict[str, object]], /
) -> tuple[tuple[str, str, str], ...]:
    versions: set[tuple[str, str, str]] = set()
    for node in nodes:
        operator = node["operator"]
        if operator["kind"] == "external":  # type: ignore[index]
            versions.add(  # type: ignore[arg-type]
                (operator["provider"], operator["name"], operator["version"])  # type: ignore[index]
            )
    return tuple(sorted(versions))


def _cache_udf_versions(
    nodes: list[dict[str, object]], /
) -> tuple[tuple[str, str, str], ...]:
    versions: set[tuple[str, str, str]] = set()
    for node in nodes:
        operator = node["operator"]
        for udf in operator.get("udfs", ()):  # type: ignore[union-attr]
            if isinstance(udf, dict):
                versions.add((udf["provider"], udf["name"], udf["version"]))  # type: ignore[arg-type]
    return tuple(sorted(versions))


def _compile_cache_key(
    program: Program,
    mode: str,
    document: dict[str, object],
    capabilities: RuntimeCapabilities,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> _CompileCacheKey:
    nodes = _cache_graph_nodes(document)
    return _CompileCacheKey(
        program_fingerprint=program.fingerprint,
        mode=mode,
        input_declarations=tuple(
            value._node.node_bytes.hex() for value in program.inputs
        ),
        capability_schema_version=capabilities.schema_version,
        capability_session_id=capabilities.scope.session_id,
        capability_revision=capabilities.scope.revision,
        operator_versions=_cache_operator_versions(nodes, capabilities),
        provider_versions=_cache_provider_versions(nodes),
        udf_versions=_cache_udf_versions(nodes),
        allowed_lateness_micros=allowed_lateness_micros,
        late_policy=late_policy,
    )


def compile_program_batch(program: Program, runtime: object, /) -> BatchExecutionPlan:
    """Lower one program to a strict project-v3 batch plan."""

    selected = _require_runtime(runtime, "compile_batch")
    document = lower_program_document(program, selected, "batch")
    capabilities = selected.capabilities()
    key = _compile_cache_key(program, "batch", document, capabilities, 0, "error")
    return selected._cached_symbolic_compile(
        key, lambda: selected.compile_batch_project(_canonical(document))
    )  # type: ignore[return-value]


def compile_program_stream(
    program: Program,
    runtime: object,
    allowed_lateness_micros: object,
    late_policy: object,
    /,
) -> StreamExecutionPlan:
    """Lower one program to a strict project-v3 continuous plan.

    The validated lateness arguments are written into every lowered rolling
    node; row-local programs are unaffected by them.
    """

    selected = _require_runtime(runtime, "compile_stream")
    _validate_lateness(allowed_lateness_micros, late_policy)
    document = lower_program_document(
        program,
        selected,
        "stream",
        allowed_lateness_micros=allowed_lateness_micros,
        late_policy=late_policy,
    )
    capabilities = selected.capabilities()
    key = _compile_cache_key(
        program,
        "stream",
        document,
        capabilities,
        allowed_lateness_micros,
        late_policy,
    )
    return selected._cached_symbolic_compile(
        key,
        lambda: selected._compile_stream_graph_project(
            _canonical(document), requirements=StreamRequirements()
        ),
    )  # type: ignore[return-value]


def _validate_lateness(allowed_lateness_micros: object, late_policy: object, /) -> None:
    if type(allowed_lateness_micros) is not int:
        raise TypeError(
            "allowed_lateness_micros must be an exact int; got"
            f" {type_name(allowed_lateness_micros)}"
        )
    if allowed_lateness_micros < 0:
        raise ValueError(
            "allowed_lateness_micros: invalid_literal: must be non-negative"
        )
    if allowed_lateness_micros > _U64_MAX:
        raise ValueError(
            "allowed_lateness_micros: invalid_literal: must fit the unsigned"
            " 64-bit microsecond range"
        )
    if type(late_policy) is not str:
        raise TypeError(f"late_policy must be a string; got {type_name(late_policy)}")
    if late_policy not in ("error", "drop"):
        raise ValueError(
            "late_policy: invalid_literal: must be 'error' or"
            f" 'drop'; got {late_policy!r}"
        )
