from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import FrozenInstanceError

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.symbolic as symbolic

# Captured from ccda5564 before replacing the original host constructor.
IDENTITIES = {
    "error": {
        "project_sha256": (
            "c76b01e10f57335c12743e715f6fb21dbbd04d7e9acbb52f5841cbc7906898d1"
        ),
        "program_fingerprint": (
            "c57de6ac4c0bc06034402e43b66c0ba0374c08ba76caf10d2d49f731111f3da9"
        ),
        "batch_fingerprint": (
            "7f79252e8fd78013a994a4f92d09fb130c3f5bc0cc5b07d57b4277551ef1cb85"
        ),
        "stream_fingerprint": (
            "e6e74846373d400add03aea21a7ff4471c1c614abad6f2ec710d3a9d87e3d75a"
        ),
        "preparation_key_sha256": (
            "9dcfd01618dd6da5795f3077d9e552fc8f24b00b437d789c5d975f77587658b0"
        ),
    },
    "drop": {
        "project_sha256": (
            "7a5117a09d5fd6a10058e8b3414322f76b726966fb1ebf77034cc232a54e9a23"
        ),
        "program_fingerprint": (
            "0438452c9e1a4cadfcc16360bd97a6ca317a5ee35b5f36977edd4bc8230f0711"
        ),
        "batch_fingerprint": (
            "b0e8b3cc5343c14c1aa91fecf13a9a6fd47cbdc1545340f6b59fc77dd3d0cc45"
        ),
        "stream_fingerprint": (
            "0cd303ee9e2092295928bf912551a1cbeaa5cad5a1bab35f0a18977df1f26db0"
        ),
        "preparation_key_sha256": (
            "f7c92c5f63f9af2d8408142d6a24d4901fba0a6739ec0352f6dc85e23593bb2d"
        ),
    },
}


def test_sort_window_is_public_frozen_and_explicitly_typed():
    window = cf.SortWindow("ts", start_micros=-10, end_micros=10)
    assert symbolic.SortWindow is cf.SortWindow
    assert "SortWindow" in cf.__all__ and "SortWindow" in symbolic.__all__
    assert (window.event_time, window.start_micros, window.end_micros) == (
        "ts",
        -10,
        10,
    )
    assert window == cf.SortWindow("ts", start_micros=-10, end_micros=10)
    with pytest.raises(FrozenInstanceError):
        window.start_micros = 0
    assert not hasattr(window, "__dict__")
    assert list(inspect.signature(cf.SortWindow).parameters) == [
        "event_time",
        "start_micros",
        "end_micros",
    ]
    assert list(inspect.signature(cf.SortSpec.closed_window).parameters) == [
        "window",
        "state_budget",
        "offset",
        "sequence_by",
        "late_policy",
    ]


@pytest.mark.parametrize(
    "column,start,end,error,path",
    [
        (None, 0, 10, TypeError, "event_time"),
        ("", 0, 10, ValueError, "event_time"),
        ("ts", True, 10, TypeError, "start_micros"),
        ("ts", 0.0, 10, TypeError, "start_micros"),
        ("ts", 0, False, TypeError, "end_micros"),
        ("ts", 0, 10.0, TypeError, "end_micros"),
        ("ts", -(1 << 63) - 1, 10, ValueError, "start_micros"),
        ("ts", 0, 1 << 63, ValueError, "end_micros"),
        ("ts", 10, 10, ValueError, "end_micros"),
        ("ts", 11, 10, ValueError, "end_micros"),
    ],
)
def test_sort_window_rejects_types_and_bounds_at_its_own_path(
    column, start, end, error, path
):
    with pytest.raises(error, match=f"SortWindow.{path}"):
        cf.SortWindow(column, start_micros=start, end_micros=end)


def test_sort_window_signed_extremes_and_required_constructor_arguments():
    window = cf.SortWindow("ts", start_micros=-(1 << 63), end_micros=(1 << 63) - 1)
    assert window.start_micros == -(1 << 63) and window.end_micros == (1 << 63) - 1
    for other in ("ts", {}, ("ts", 0, 10), None):
        with pytest.raises(TypeError, match="window must be a SortWindow"):
            cf.SortSpec.closed_window(other, state_budget=cf.StateBudget(2, 8 << 20))
    with pytest.raises(TypeError, match="state_budget"):
        cf.SortSpec.closed_window(window)
    with pytest.raises(TypeError):
        cf.SortSpec.closed_window(
            "ts", start_micros=0, end_micros=10, state_budget=cf.StateBudget(2, 8 << 20)
        )


def expected_spec(policy):
    return dict(
        history=dict(
            kind="closed_window",
            event_time_column="ts",
            start_micros=-10,
            end_micros=10,
        ),
        ingress_policy="copy_visible_v2",
        state_budget=dict(max_rows=1000, max_bytes=8388608),
        offset=1,
        ties=dict(kind="sequence", columns=["seq"]),
        late_policy=dict(kind="error", scope="envelope")
        if policy == "error"
        else dict(kind="drop", metrics_version=1),
    )


def program(policy):
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us")),
            pa.field("x", pa.int64()),
            pa.field("seq", pa.uint64(), nullable=False),
        ]
    )
    t = cf.table_input("events", schema=schema)
    spec = cf.SortSpec.closed_window(
        cf.SortWindow("ts", start_micros=-10, end_micros=10),
        state_budget=cf.StateBudget(1000, 8 << 20),
        offset=1,
        sequence_by=["seq"],
        late_policy=policy,
    )
    out = t.top_k(1, [cf.SortKey("x")], spec=spec)
    return cf.Program("window_v2", mode="batch", outputs={"out": out}), t, spec


@pytest.mark.parametrize("policy", ["error", "drop"])
def test_sort_window_preserves_v1_wire_export_cache_and_plan_identities(policy):
    from calc_flow.symbolic.lower.program import _preparation_key
    from calc_flow.symbolic.order import cache_policy

    p, t, spec = program(policy)
    expected = expected_spec(policy)
    assert spec._wire == json.dumps(expected, sort_keys=True, separators=(",", ":"))
    project = p.to_project()
    operator = next(
        n["operator"]
        for n in project.model_dump()["graph"]["nodes"]
        if n["operator"]["kind"] == "bounded_order"
    )
    assert json.dumps(operator, sort_keys=True, separators=(",", ":")) == json.dumps(
        dict(
            kind="bounded_order",
            by=[dict(column="x", direction="ascending", nulls="last")],
            fetch=1,
            spec=expected,
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    sql = cf.sql(
        "SELECT * FROM CF_ORDER_WINDOW('a','ts',-10,10,1000,8388608,"
        f"'{policy}',ARRAY['seq']) ORDER BY x LIMIT 1 OFFSET 1",
        a=t,
    )
    sql_project = cf.Program(
        "sql_window", mode="batch", outputs={"out": sql}
    ).to_project()
    assert operator == next(
        n["operator"]
        for n in sql_project.model_dump()["graph"]["nodes"]
        if n["operator"]["kind"] == "bounded_order"
    )
    baseline = IDENTITIES[policy]
    assert (
        hashlib.sha256(project.canonical_json().encode()).hexdigest()
        == baseline["project_sha256"]
    )
    assert p.fingerprint == baseline["program_fingerprint"]
    key = list(_preparation_key(p, "batch", 0, "error"))
    key[3] = [value.hex() for value in key[3]]
    assert (
        hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
        == baseline["preparation_key_sha256"]
    )
    assert cache_policy(p) == (
        "bounded_order_v1",
        "copy_visible_v2",
        "order_accounting1",
        "analysis10",
    )
    runtime = cf.Runtime()
    plan = p.compile_batch(runtime)
    assert plan.fingerprint == baseline["batch_fingerprint"]
    assert program(policy)[0].compile_batch(runtime) is plan
    assert (
        p.for_mode("stream").compile_stream().fingerprint
        == baseline["stream_fingerprint"]
    )
