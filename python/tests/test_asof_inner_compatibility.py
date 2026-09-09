from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from calc_flow import JoinStateLimits, JoinTimeBounds, Runtime
from calc_flow.symbolic import Field, Program, table, table_input
from calc_flow.symbolic.lower import lower_program_document

_FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/asof-inner-compat-v1"


def _input(name: str, value_name: str):
    return table_input(
        name,
        schema=[
            Field("key", "int64", nullable=False),
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field(value_name, "float64", nullable=False),
        ],
        entity_by=["key"],
        event_time="ts",
        sequence_by=["sequence"],
    )


def _declaration(version: int):
    left = _input("left_events", "left_value")
    right = _input("right_events", "right_value")
    ordering = (
        {
            "output_entity_by": ["left__key"],
            "output_event_time": "left__ts",
            "output_sequence_by": ["left__sequence", "right__sequence"],
        }
        if version == 2
        else {}
    )
    joined = table.stream_join(
        left,
        right,
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=JoinTimeBounds(timedelta(seconds=5), timedelta(seconds=2)),
        limits=JoinStateLimits(1_000, 16 * 1024 * 1024, 10_000),
        **ordering,
    )
    program = Program(
        "legacy-inner", inputs=[left, right], outputs=[("matches", joined)]
    )
    return joined, program


@pytest.mark.parametrize("version", [1, 2])
def test_inner_symbolic_node_bytes_and_program_identity_remain_frozen(version):
    joined, program = _declaration(version)
    expected = json.loads((_FIXTURES / "symbolic-identities.json").read_text())
    identity = expected[str(version)]

    assert joined._node.op.version == version
    assert joined._node.node_bytes.hex() == identity["node_bytes"]
    assert joined.digest == identity["digest"]
    assert program.fingerprint == identity["program_fingerprint"]


@pytest.mark.parametrize("version", [1, 2])
def test_inner_symbolic_lowering_preserves_historical_project_bytes(version):
    _, program = _declaration(version)
    document = lower_program_document(program, Runtime(), "stream")
    actual = json.dumps(document, sort_keys=True, separators=(",", ":"))

    assert actual == (_FIXTURES / f"symbolic-v{version}-lowering.json").read_text()


def test_inner_connector_project_fingerprint_preserves_historical_identity():
    project = (_FIXTURES / "connector-project.json").read_text()
    plan = Runtime().compile_stream_project(project)
    expected = json.loads((_FIXTURES / "fingerprints.json").read_text())

    assert plan.fingerprint == expected["connector_project"]
