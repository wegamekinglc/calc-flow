"""Native ASOF SQL declarations; no Python SQL parser or matching engine."""

from __future__ import annotations

import json


def native_plan(node):
    from calc_flow import _native
    from calc_flow.symbolic.analyzer import _cstr_seq

    return json.loads(
        _native._asof_join_query(
            node.attr("query").value, list(_cstr_seq(node.attr("aliases")))
        )
    )


def join_node(node, plan):
    from calc_flow.symbolic.analyzer import _cstr_seq
    from calc_flow.symbolic.asof import _canonical
    from calc_flow.symbolic.nodes import build

    tables = dict(zip(_cstr_seq(node.attr("aliases")), node.args))
    return build(
        "stream_asof_join",
        (tables[plan["left_alias"]], tables[plan["right_alias"]]),
        {"spec": _canonical(plan["spec"])},
    )
