from __future__ import annotations

import ast
from pathlib import Path


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
