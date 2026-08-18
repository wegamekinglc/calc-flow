from __future__ import annotations

import calc_flow
from calc_flow import _native


def test_native_package_reports_v3() -> None:
    assert calc_flow.__version__ == "3.0.0"
    assert _native.version() == "3.0.0"
    assert issubclass(calc_flow.CompileError, calc_flow.CalcFlowError)
    assert calc_flow.CalcFlowError.__module__ == "calc_flow._native"


def test_exception_hierarchy_is_exported() -> None:
    direct_errors = (
        calc_flow.ConfigError,
        calc_flow.CompileError,
        calc_flow.ExecutionError,
        calc_flow.CheckpointError,
    )
    assert all(issubclass(error, calc_flow.CalcFlowError) for error in direct_errors)
    assert issubclass(calc_flow.ProviderError, calc_flow.ExecutionError)
    assert issubclass(calc_flow.CancelledError, calc_flow.ExecutionError)
