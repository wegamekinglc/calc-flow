from __future__ import annotations

from calc_flow import _native as _native
from calc_flow.array import register_jax, register_numpy
from calc_flow.capabilities import (
    OperatorCapability,
    ProviderCapability,
    ProviderOption,
    ProviderOptionsSchema,
    ProviderPort,
    RuntimeCapabilities,
    RuntimeSessionScope,
    UdfCapability,
)
from calc_flow.config import ProjectDocument
from calc_flow.errors import (
    CalcFlowError,
    CancelledError,
    CheckpointError,
    CompileError,
    ConfigError,
    ExecutionError,
    ProviderError,
)
from calc_flow.pipeline import (
    ExecutionPlan,
    PipelineBuilder,
    Runtime,
    project_json_schema,
    validate_project_json,
)
from calc_flow.runtime import MicroBatchRunner, StreamingRunner
from calc_flow.store import FileCheckpointStore, FileProjectStore

__version__ = "2.0.0"
Batch = _native.Batch
RunResult = _native.RunResult

__all__ = [
    "Batch",
    "CalcFlowError",
    "CancelledError",
    "CheckpointError",
    "CompileError",
    "ConfigError",
    "ExecutionError",
    "FileCheckpointStore",
    "FileProjectStore",
    "ProviderError",
    "ProjectDocument",
    "ExecutionPlan",
    "PipelineBuilder",
    "OperatorCapability",
    "ProviderCapability",
    "ProviderOption",
    "ProviderOptionsSchema",
    "ProviderPort",
    "MicroBatchRunner",
    "Runtime",
    "RuntimeCapabilities",
    "RuntimeSessionScope",
    "RunResult",
    "StreamingRunner",
    "UdfCapability",
    "register_jax",
    "register_numpy",
    "project_json_schema",
    "validate_project_json",
    "__version__",
]
