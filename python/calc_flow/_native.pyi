import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import Literal, Never, Protocol, TypedDict, final

import pyarrow as pa

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)
type _JSONInput = (
    None | bool | int | float | str | list[_JSONInput] | Mapping[str, _JSONInput]
)
type StreamingFailureReasonCode = Literal[
    "join_state_limit_exceeded",
    "join_match_limit_exceeded",
    "join_counter_overflow",
    "join_time_conversion_failed",
]

class _ArrowCStream(Protocol):
    def __arrow_c_stream__(
        self, requested_schema: object | None = None, /
    ) -> object: ...

class CalcFlowError(Exception): ...

class ConfigError(CalcFlowError):
    @property
    def issues(self) -> tuple[dict[str, str], ...]: ...

class CompileError(CalcFlowError): ...
class ExecutionError(CalcFlowError): ...
class ProviderError(ExecutionError): ...
class CheckpointError(CalcFlowError): ...
class CancelledError(ExecutionError): ...

class StreamingRuntimeError(CalcFlowError):
    @property
    def category(self) -> str: ...
    @property
    def reason_code(self) -> StreamingFailureReasonCode | None: ...
    @property
    def message(self) -> str: ...
    @property
    def job_id(self) -> int | None: ...
    @property
    def epoch(self) -> int | None: ...
    @property
    def checkpoint_phase(self) -> str | None: ...
    @property
    def component_kind(self) -> str | None: ...
    @property
    def component_id(self) -> str | None: ...
    @property
    def diagnostic_id(self) -> int | None: ...
    @property
    def position(self) -> int: ...

class CheckpointPublicationUnknownError(StreamingRuntimeError): ...

@final
class ExecutionOptions:
    def __init__(
        self,
        settings: Mapping[str, _JSONInput] | None = ...,
        deadline: datetime | None = ...,
    ) -> None: ...
    @property
    def settings(self) -> dict[str, JSONValue]: ...
    @property
    def deadline(self) -> datetime | None: ...

@final
class ProviderContext:
    def __new__(cls) -> Never:
        """Engine-created; application construction is not public API."""
        ...
    @property
    def settings(self) -> dict[str, JSONValue]: ...
    @property
    def deadline(self) -> datetime | None: ...

class _ExecutionCancellation(Protocol):
    def cancel(self) -> None: ...

class Batch:
    @staticmethod
    def from_pyarrow(
        table: _ArrowCStream, metadata: Mapping[str, object] | None = None
    ) -> Batch: ...
    @staticmethod
    def from_array(
        array: object,
        *,
        backend: str,
        metadata: Mapping[str, object] | None = None,
    ) -> Batch: ...
    @staticmethod
    def _from_external(
        object: object, backend: str, len: int, metadata: Mapping[str, object]
    ) -> Batch:
        """Build an internal Python-provider batch; this is not a stable public API."""
        ...
    @staticmethod
    def _new_owned_numpy(shape: Sequence[int], dtype: str) -> tuple[object, object]:
        """Allocate private Rust-owned NumPy storage."""
        ...
    @staticmethod
    def _from_owned_array(
        array: object,
        *,
        backend: str,
        token: object | None,
        metadata: Mapping[str, object],
    ) -> Batch:
        """Adopt a trusted provider result; this is not a stable public API."""
        ...

    def to_pyarrow(self) -> pa.Table: ...
    @property
    def array(self) -> object: ...
    @property
    def backend(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def num_rows(self) -> int: ...
    @property
    def metadata(self) -> dict[str, object]: ...

class Runtime:
    def __init__(self) -> None: ...
    def register_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: object,
        *,
        accepts_context: bool = False,
    ) -> None: ...
    def _register_mapping_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: object,
        *,
        input_ports: Sequence[tuple[str, str]],
        output_ports: Sequence[tuple[str, str]],
        accepts_context: bool = False,
    ) -> None: ...
    def _register_stateless_stream_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: object,
        *,
        microbatch_invariant: bool,
        deterministic: bool,
        replay_safe: bool,
    ) -> None: ...
    def _register_stateless_stream_mapping_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: object,
        *,
        input_ports: Sequence[tuple[str, str]],
        output_ports: Sequence[tuple[str, str]],
        microbatch_invariant: bool,
        deterministic: bool,
        replay_safe: bool,
    ) -> None: ...
    def register_scalar_udf(
        self,
        *,
        provider: str,
        name: str,
        version: str,
        input_types: Sequence[str],
        return_type: str,
        volatility: str,
        function: Callable[..., object],
    ) -> None: ...
    def catalog(self) -> list[dict[str, object]]: ...
    def validation_report(self, project_json: str) -> dict[str, object]: ...
    def compile_project(self, project_json: str) -> ExecutionPlan: ...
    def compile_stream_project(
        self, project_json: str, delivery: Mapping[str, str]
    ) -> StreamExecutionPlan: ...
    def _compile_stream_graph_project(
        self, project_json: str, delivery: Mapping[str, str]
    ) -> StreamExecutionPlan: ...

class ExecutionPlan:
    @property
    def name(self) -> str: ...
    @property
    def fingerprint(self) -> str: ...
    def execute(
        self,
        inputs: dict[str, Batch],
        *,
        options: ExecutionOptions | None = None,
    ) -> RunResult: ...
    def execute_async(
        self,
        inputs: dict[str, Batch],
        *,
        options: ExecutionOptions | None = None,
    ) -> Awaitable[RunResult]: ...
    def _execute_async_cancellable(
        self,
        inputs: dict[str, Batch],
        *,
        options: ExecutionOptions | None = None,
    ) -> tuple[asyncio.Future[RunResult], _ExecutionCancellation]: ...
    def snapshot_async(self) -> Awaitable[dict[str, object]]: ...
    def restore_async(self, state_json: str) -> Awaitable[None]: ...
    def reset_async(self) -> Awaitable[None]: ...

class StreamExecutionPlan:
    @property
    def name(self) -> str: ...
    @property
    def fingerprint(self) -> str: ...
    @property
    def source_binding_ids(self) -> tuple[str, ...]: ...
    @property
    def static_input_ids(self) -> tuple[str, ...]: ...
    @property
    def sink_binding_ids(self) -> tuple[str, ...]: ...
    @property
    def requirements(self) -> dict[str, str]: ...

@final
class _ManagedCheckpointRuntime:
    def __init__(self, directory: str, /) -> None: ...

@final
class _StreamingRunner:
    def __init__(
        self,
        plan: StreamExecutionPlan,
        sources: Mapping[str, object],
        sinks: Mapping[str, Sequence[object]],
        checkpoints: _ManagedCheckpointRuntime,
        config: Mapping[str, int],
        static_inputs: Mapping[str, object],
    ) -> None: ...
    def start_async(self) -> Awaitable[_StreamingJob]: ...
    def _wait_start_cleanup_async(self) -> Awaitable[None]: ...
    def _release_roots(self) -> None: ...

@final
class _StreamingJob:
    @property
    def id(self) -> int: ...
    def status(self) -> dict[str, object]: ...
    def trigger_checkpoint_async(self) -> Awaitable[int]: ...
    def shutdown_async(self) -> Awaitable[dict[str, object]]: ...
    def cancel_async(self) -> Awaitable[dict[str, object]]: ...
    def wait_async(self) -> Awaitable[dict[str, object]]: ...
    def _release_roots(self) -> None: ...

class _FileProjectStore:
    def __init__(self, directory: str) -> None: ...
    def create(self, project_json: str) -> Awaitable[None]: ...
    def put(self, project_json: str) -> Awaitable[None]: ...
    def get(self, project_id: str) -> Awaitable[str]: ...
    def list(self) -> Awaitable[list[str]]: ...
    def delete(self, project_id: str) -> Awaitable[None]: ...

def import_project_json(document: bytes) -> str: ...
def import_project_yaml(document: bytes) -> str: ...
def export_project_json(project_json: str) -> str: ...
def export_project_yaml(project_json: str) -> str: ...

class _NodeTiming(TypedDict):
    duration_ns: int
    input_rows: dict[str, int]
    output_rows: dict[str, int]

class _DataFusionMetric(TypedDict):
    query_id: int
    node_id: str | None
    sql_parse_ns: int
    logical_planning_ns: int
    physical_planning_ns: int
    physical_planning_count: int
    planning_ns: int
    stream_open_ns: int
    execution_ns: int
    collect_ns: int
    output_rows: int
    logical_plan: str
    physical_plan: str

class RunResult:
    @property
    def outputs(self) -> dict[str, Batch]: ...
    @property
    def metadata(self) -> dict[str, str]: ...
    @property
    def node_timings(self) -> dict[str, _NodeTiming]: ...
    @property
    def datafusion_metrics(self) -> list[_DataFusionMetric]: ...

def project_json_schema() -> str: ...
def validate_project_json(project_json: str) -> str: ...
def version() -> str: ...
def registered_connectors() -> list[dict[str, object]]: ...
