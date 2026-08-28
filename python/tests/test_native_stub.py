from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from calc_flow import _native


def test_execution_options_stub_uses_opaque_constructor_defaults() -> None:
    stub_path = Path(__file__).resolve().parents[1] / "calc_flow" / "_native.pyi"
    module = ast.parse(stub_path.read_text(encoding="utf-8"))
    execution_options = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ExecutionOptions"
    )
    constructor = next(
        node
        for node in execution_options.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    parameters = constructor.args.posonlyargs + constructor.args.args
    assert [parameter.arg for parameter in parameters] == [
        "self",
        "settings",
        "deadline",
    ]
    settings_default, deadline_default = constructor.args.defaults
    assert isinstance(settings_default, ast.Constant)
    assert settings_default.value is Ellipsis
    assert isinstance(deadline_default, ast.Constant)
    assert deadline_default.value is Ellipsis


def test_continuous_native_stub_matches_registered_runtime_surface() -> None:
    stub_path = Path(__file__).resolve().parents[1] / "calc_flow" / "_native.pyi"
    module = ast.parse(stub_path.read_text(encoding="utf-8"))
    classes = {
        node.name: node for node in module.body if isinstance(node, ast.ClassDef)
    }

    for name in (
        "StreamExecutionPlan",
        "_ManagedCheckpointRuntime",
        "_StreamingRunner",
        "_StreamingJob",
        "StreamingRuntimeError",
        "CheckpointPublicationUnknownError",
    ):
        assert name in classes
        assert hasattr(_native, name)
    assert str(inspect.signature(_native._ManagedCheckpointRuntime)) == "(directory, /)"
    assert str(inspect.signature(_native._StreamingRunner)) == (
        "(plan, sources, sinks, checkpoints, config, static_inputs)"
    )
    with pytest.raises(TypeError, match="positional-only"):
        _native._ManagedCheckpointRuntime(directory="checkpoint")


def test_native_stub_omits_legacy_continuous_classes() -> None:
    stub_path = Path(__file__).resolve().parents[1] / "calc_flow" / "_native.pyi"
    module = ast.parse(stub_path.read_text(encoding="utf-8"))
    classes = {
        node.name: node for node in module.body if isinstance(node, ast.ClassDef)
    }

    for name in (
        "_ContinuousStreamingRunner",
        "_FileCheckpointStore",
        "_MicroBatchRunner",
    ):
        assert name not in classes
    assert "_StreamingRunner" in classes
