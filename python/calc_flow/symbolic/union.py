"""Exact-schema union declaration checks; native code owns table execution."""

from __future__ import annotations

import json

from calc_flow.symbolic.types import Field


def sql_queries(node):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import _cstr_seq

    return json.loads(
        _native._union_all_queries(
            node.attr("query").value, list(_cstr_seq(node.attr("aliases")))
        )
    )


def has_late_input(node):
    pending, visited = list(node.args), set()
    while pending:
        current = pending.pop()
        if current.digest in visited:
            continue
        if current.op.name == "late_rows":
            return True
        visited.add(current.digest)
        pending.extend(current.args)
    return False


def sql_plan(runtime, node, schemas):
    query = node.attr("query").value
    key = (
        "finite_union_v1",
        query,
        tuple(
            (name, schema.serialize().to_pybytes())
            for name, schema in sorted(schemas.items())
        ),
    )
    return runtime._cached_symbolic_compile(
        key, lambda: json.loads(runtime._inner._plan_union_all(query, schemas))
    )


def expand_query_graph(graph, nodes, paths, analyzers):
    plans = _analysis_plans(nodes, paths, analyzers)
    declarations = [
        {**node, "kind": "union", "query": None} if node["path"] in plans else node
        for node in graph["nodes"]
    ]
    edges = [edge for edge in graph["edges"] if edge["target_path"] not in plans]
    for path, (node, plan) in plans.items():
        extra_nodes, extra_edges = _arm_graph(path, node, plan, paths)
        declarations.extend(extra_nodes)
        edges.extend(extra_edges)
    return {**graph, "nodes": declarations, "edges": edges}


def _analysis_plans(nodes, paths, analyzers):
    return {
        paths[key]: (nodes[key], plan)
        for analyzer in analyzers
        for key, plan in analyzer._union_plans.items()
        if key in nodes
    }


def _arm_graph(path, node, plan, paths):
    from calc_flow.symbolic.analyzer import _cstr_seq
    from calc_flow.symbolic.query_analysis import _segment

    children = dict(zip(_cstr_seq(node.attr("aliases")), node.args))
    declarations, edges = [], []
    for index, arm in enumerate(plan["arms"]):
        arm_path = f"{path}.sql.arms[{index}]"
        declarations.append(
            dict(
                path=arm_path,
                kind="sql",
                query=arm["query"],
                schema=arm["schema"],
                ordering=dict(
                    kind="unspecified",
                    keys=[],
                    ties="unspecified",
                    temporal_identity=None,
                ),
                batch_support=None,
                stream_support=None,
            )
        )
        edges.extend(
            [
                dict(
                    path=f"{arm_path}.input",
                    source_path=paths[children[arm["alias"]].digest],
                    target_path=arm_path,
                    input_alias=arm["alias"],
                ),
                dict(
                    path=_segment(f"{path}.sql.tables", arm["alias"]),
                    source_path=arm_path,
                    target_path=path,
                    input_alias=f"input_{index}",
                ),
            ]
        )
    return declarations, edges


def cache_policy(program, runtime):
    pending = [value._node for _, value in program.outputs]
    visited = set()
    candidate = False
    while pending:
        node = pending.pop()
        if node.digest in visited:
            continue
        visited.add(node.digest)
        candidate |= node.op.name == "sql" and len(node.args) > 1
        pending.extend(node.args)
    if not candidate:
        return ()
    from calc_flow.symbolic.analyzer import _run

    analyzer, _ = _run(program, runtime, program.mode)
    return ("finite_union_v1",) if analyzer._union_plans else ()


def schema_mismatch(
    expected: tuple[Field, ...], found: tuple[Field, ...]
) -> tuple[str, str] | None:
    if len(expected) != len(found):
        return (
            ".schema",
            f"exact schema required: expected {len(expected)} fields, "
            f"found {len(found)}",
        )
    for index, (left, right) in enumerate(zip(expected, found)):
        for name in ("name", "data_type", "nullable"):
            a, b = getattr(left, name), getattr(right, name)
            if a != b:
                return (
                    f".schema[{index}].{name}",
                    f"exact schema required for field {index} {left.name!r}: "
                    f"expected {a!r}, found {b!r}",
                )
    return None
