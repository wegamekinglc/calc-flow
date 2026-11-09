from __future__ import annotations

import json
from pathlib import Path

import pytest

import calc_flow as cf
from calc_flow.capabilities import _native_array_provider_capability


def _descriptor() -> dict[str, object]:
    path = (
        Path(__file__).with_name("fixtures")
        / "native_array_capabilities"
        / "rust_sequence_ir3.json"
    )
    return json.loads(path.read_text())


def test_actual_ir3_descriptor_keeps_schema_five_and_all_three_layouts() -> None:
    raw = _descriptor()
    capability = _native_array_provider_capability(raw)
    assert capability.state_layouts == (1, 2, 3)
    assert capability.options_json_schema == raw["options_json_schema"]
    assert capability.array_rules is not None
    assert {
        (rule.name, rule.version) for rule in capability.array_rules.shape_rules
    } == {
        ("native_sequence_vector", "1"),
        ("native_sequence_indicators", "2"),
        ("native_sequence_indicators", "3"),
    }
    assert cf.Runtime().capabilities().schema_version == 5


def test_rolling_inventory_advertises_the_implemented_layout_six() -> None:
    rolling = next(
        operator
        for operator in cf.Runtime().capabilities().operators
        if operator.kind == "rolling"
    )
    assert rolling.state_layouts == (1, 2, 3, 4, 5, 6)
    assert rolling.state_version == 1


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rule",
        "missing_layout",
        "missing_version",
        "cross_version_kernel",
        "default",
        "maximum",
    ],
)
def test_ir3_profile_rejects_incomplete_or_changed_rust_evidence(fault: str) -> None:
    descriptor = _descriptor()
    schema = json.loads(descriptor["options_json_schema"])
    if fault == "missing_rule":
        descriptor["array_rules"]["shape_rules"].pop()
    elif fault == "missing_layout":
        descriptor["state_layouts"].pop()
    elif fault == "missing_version":
        schema["oneOf"].pop()
    elif fault == "cross_version_kernel":
        schema["oneOf"][0]["properties"]["kernel"]["oneOf"].append(
            schema["oneOf"][2]["properties"]["kernel"]["oneOf"][-1]
        )
    else:
        macd = schema["oneOf"][2]["properties"]["kernel"]["oneOf"][-1]
        macd["properties"]["fast_span"][fault] = 1
    descriptor["options_json_schema"] = json.dumps(schema)
    with pytest.raises(ValueError):
        _native_array_provider_capability(descriptor)


def test_generated_project_schema_contains_the_table_macd_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "schemas/project-v3.schema.json").read_text())
    assert "MacdComponent" in schema["$defs"]
    assert "TableMacdComponent" not in schema["$defs"]
    branches = schema["$defs"]["RollingOutputSpec"]["oneOf"]
    macd = next(
        (
            branch
            for branch in branches
            if branch["properties"]["kind"]["const"] == "macd_component"
        ),
        None,
    )
    assert macd is not None
    assert macd["additionalProperties"] is False
    assert set(macd["required"]) == {
        "kind",
        "primitive_version",
        "input",
        "output",
        "component",
    }
    defaults = {
        "fast_span": 12,
        "slow_span": 26,
        "signal_span": 9,
        "min_periods": 1,
        "signal_min_periods": 1,
    }
    for name, default in defaults.items():
        assert macd["properties"][name]["default"] == default
        assert macd["properties"][name]["minimum"] == 1
