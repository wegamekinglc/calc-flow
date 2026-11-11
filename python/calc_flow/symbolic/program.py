"""Immutable feature sets and reusable programs over expression declarations.

Programs own declarations and canonical fingerprints. Analysis, compilation,
project export and collection share the native runtime and existing lowering.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, cast, overload

from typing_extensions import Unpack

from calc_flow._compat import dataclass
from calc_flow.symbolic.domains import type_name
from calc_flow.symbolic.expr import ArrayExpr, ColumnExpr, Parameter, TableExpr
from calc_flow.symbolic.nodes import (
    _MAGIC,
    CSeq,
    CStr,
    Node,
    _text,
    _u64,
)
from calc_flow.symbolic.query_analysis import (
    AnalysisOptions,
    PortabilityRequest,
    UpdateSemantics,
    _segment,
)
from calc_flow.symbolic.types import CompileMode

if TYPE_CHECKING:
    import pyarrow as pa

    from calc_flow._native import ExecutionOptions, NativeArray
    from calc_flow.compute import TableData
    from calc_flow.config import ProjectDocument
    from calc_flow.pipeline import BatchExecutionPlan, Runtime, StreamExecutionPlan
    from calc_flow.runtime import StreamRuntimeConfig, WatermarkPolicy
    from calc_flow.stream import StreamInput, StreamOutput, StreamResults
    from calc_flow.symbolic.analyzer import AnalysisResult
    from calc_flow.symbolic.types import Field, LatePolicy

_PROGRAM_TAG = 0x21
_OMITTED = object()


def _engine_selection(engine: object, mode: object) -> Literal["sql", "streaming"]:
    if engine is _OMITTED and mode is _OMITTED:
        raise TypeError(
            "Program.mode: invalid_literal: provide exactly one of mode or engine"
        )
    if engine is not _OMITTED and mode is not _OMITTED:
        raise ValueError(
            "Program.mode: invalid_literal: mode and legacy engine "
            "cannot both be supplied"
        )
    field, value, allowed = (
        ("engine", engine, ("sql", "streaming"))
        if mode is _OMITTED
        else ("mode", mode, ("batch", "stream"))
    )
    return _checked_engine_value(field, value, allowed)


def _checked_engine_value(
    field: str, value: object, allowed: tuple[str, str]
) -> Literal["sql", "streaming"]:
    message = f"Program.{field}: invalid_literal: expected {allowed!r}"
    if type(value) is not str:
        raise TypeError(message)
    if value not in allowed:
        raise ValueError(message)
    return "sql" if value in ("sql", "batch") else "streaming"


def _check_schema_output_types(
    program: Program, runtime: Runtime, mode: CompileMode
) -> None:
    from calc_flow.symbolic.analyzer import _require_mode
    from calc_flow.symbolic.lower.program import _check_expression_capability
    from calc_flow.symbolic.query_analysis import _declaration_paths

    analyzer, _ = _check_expression_capability(program, runtime, _require_mode(mode))
    for name, value in program.outputs:
        path = _segment("outputs", name)
        fields = analyzer.table(value._node, path).schema
        _check_schema_fields(fields, path)
    paths, nodes, _ = _declaration_paths(program)
    for digest, node in nodes.items():
        if node.op.name in {"sql", "snapshot_sql"}:
            _check_schema_fields(
                analyzer.table(node, paths[digest]).schema, paths[digest]
            )


def _check_schema_fields(fields: Sequence[Field], path: str) -> None:
    from calc_flow.errors import CompileError
    from calc_flow.symbolic.types import check_table_field_type

    for index, field in enumerate(fields):
        try:
            check_table_field_type(field.data_type, f"{path}.schema[{index}].data_type")
        except ValueError as error:
            raise CompileError(str(error)) from error


class _StreamingExecuteOptions(TypedDict, total=False):
    config: StreamRuntimeConfig | None
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None


def _stream_controls(
    *,
    config: StreamRuntimeConfig | None = None,
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None = None,
) -> tuple[
    StreamRuntimeConfig | None, WatermarkPolicy | Mapping[str, WatermarkPolicy] | None
]:
    return config, watermarks


_STREAM_ONLY_OPS = frozenset(
    {
        "snapshot_sql",
        "stream_join",
        "stream_asof_join",
        "window_tumbling",
        "window_hopping",
        "late_output",
        "late_rows",
    }
)


def _validate_engine_node(
    engine: Literal["sql", "streaming"], node: Node, path: str
) -> None:
    if any(child.op.name == "snapshot_sql" for child in node.args):
        raise ValueError(
            f"{path}: unsupported_mode: snapshot_sql outputs must connect "
            "directly to sinks"
        )
    if engine == "streaming" and node.op.name == "sql":
        aliases = node.attr("aliases")
        if isinstance(aliases, CSeq) and len(aliases.items) != 1:
            raise ValueError(
                f"{path}: unsupported_mode: streaming SQL accepts one table alias"
            )
    if engine == "sql" and node.op.name in _STREAM_ONLY_OPS:
        raise ValueError(
            f"{path}: unsupported_mode: {node.op.name} requires the streaming engine"
        )


def _validate_engine_outputs(
    engine: Literal["sql", "streaming"],
    outputs: tuple[tuple[str, TableExpr | ArrayExpr], ...],
) -> None:
    """Reject graph shapes whose engine incompatibility is known at declaration."""

    visited: set[str] = set()

    def visit(node: Node, path: str) -> None:
        if node.digest in visited:
            return
        visited.add(node.digest)
        _validate_engine_node(engine, node, path)
        for child in node.args:
            visit(child, path)

    for name, value in outputs:
        visit(value._node, f"outputs.{name}")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FeatureSet:
    """An ordered immutable set of uniquely named column expressions."""

    _features: tuple[tuple[str, ColumnExpr], ...]

    def __init__(
        self,
        features: Sequence[tuple[str, ColumnExpr]] = (),
        /,
    ) -> None:
        copied: list[tuple[str, ColumnExpr]] = []
        seen: set[str] = set()
        for index, item in enumerate(features):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    f"features[{index}]: must be a (name, ColumnExpr) pair; got"
                    f" {type_name(item)}"
                )
            name, value = item
            if type(name) is not str:
                raise TypeError(
                    f"features[{index}].name: must be a string; got {type_name(name)}"
                )
            if not isinstance(value, ColumnExpr):
                raise TypeError(
                    f"features[{index}].value: must be a ColumnExpr; got"
                    f" {type_name(value)}"
                )
            if name in seen:
                raise ValueError(
                    f"features[{index}].name: duplicate_name: duplicate feature"
                    f" name {name!r}"
                )
            seen.add(name)
            copied.append((name, value))
        object.__setattr__(self, "_features", tuple(copied))

    @property
    def features(self) -> tuple[tuple[str, ColumnExpr], ...]:
        """The declared features in declaration order."""

        return self._features

    def with_feature(self, name: str, value: ColumnExpr, /) -> FeatureSet:
        """Return a new feature set with one feature appended."""

        return FeatureSet((*self._features, (name, value)))


def _node_name(node: Node, /) -> str:
    value = node.attr("name")
    if isinstance(value, CStr):
        return value.value
    raise TypeError("declaration node is missing its name attribute")


def _collect_program_nodes(
    inputs: tuple[TableExpr | ArrayExpr | Parameter[object], ...],
    outputs: tuple[tuple[str, TableExpr | ArrayExpr], ...],
    /,
) -> dict[str, tuple[bytes, Node]]:
    """Return every unique node reachable from a declared input or output."""

    nodes: dict[str, tuple[bytes, Node]] = {}

    def visit(node: Node) -> None:
        existing = nodes.get(node.digest)
        if existing is not None:
            if existing[0] != node.node_bytes:
                raise ValueError(
                    "Program.fingerprint: unresolved_type: digest collision for"
                    f" node {node.op.name}@{node.op.version}"
                )
            return
        nodes[node.digest] = (node.node_bytes, node)
        for child in node.args:
            visit(child)

    for value in inputs:
        visit(value._node)
    for _, value in outputs:
        visit(value._node)
    return nodes


def _program_fingerprint(
    name: str,
    inputs: tuple[TableExpr | ArrayExpr | Parameter[object], ...],
    outputs: tuple[tuple[str, TableExpr | ArrayExpr], ...],
    /,
) -> str:
    """Compute the frozen v1 program fingerprint over the declaration graph."""

    nodes = _collect_program_nodes(inputs, outputs)
    edges = _program_edges(nodes)
    node_records = sorted(
        ((bytes.fromhex(digest), record) for digest, record in nodes.items()),
        key=lambda item: item[0],
    )
    body = (
        bytes((_PROGRAM_TAG,))
        + _text(name)
        + _u64(len(inputs))
        + b"".join(_input_records(inputs))
        + _u64(len(outputs))
        + b"".join(
            _text(output_name) + bytes.fromhex(value._node.digest)
            for output_name, value in outputs
        )
        + _u64(len(node_records))
        + b"".join(
            digest + _u64(len(record[0])) + record[0] for digest, record in node_records
        )
        + _u64(len(edges))
        + b"".join(parent + _u64(index) + child for parent, index, child in edges)
    )
    return hashlib.sha256(_MAGIC + b"\x02" + _u64(len(body)) + body).hexdigest()


def _program_edges(
    nodes: dict[str, tuple[bytes, Node]],
    /,
) -> list[tuple[bytes, int, bytes]]:
    edges: list[tuple[bytes, int, bytes]] = []
    for _, node in nodes.values():
        for index, child in enumerate(node.args):
            edges.append(
                (
                    bytes.fromhex(node.digest),
                    index,
                    bytes.fromhex(child.digest),
                )
            )
    edges.sort()
    return edges


def _input_records(
    inputs: tuple[TableExpr | ArrayExpr | Parameter[object], ...],
    /,
) -> list[bytes]:
    return [
        _text(_node_name(value._node)) + bytes.fromhex(value._node.digest)
        for value in inputs
    ]


def _input_declaration(
    value: object, index: int
) -> TableExpr | ArrayExpr | Parameter[object]:
    if not isinstance(value, (TableExpr, ArrayExpr, Parameter)):
        raise TypeError(
            f"Program.inputs[{index}]: expected TableExpr | ArrayExpr | "
            f"Parameter[object]; got {type_name(value)}"
        )
    if isinstance(value, ArrayExpr) and value._node.op.name != "array_input":
        from calc_flow.symbolic.native_arrays import is_native_array

        error = ValueError if is_native_array(value._node) else TypeError
        raise error(
            f"Program.inputs[{index}]: ArrayExpr inputs must be "
            "declared array_input values"
        )
    if value._node.op.name not in ("table_input", "array_input", "parameter"):
        raise ValueError(
            f"Program.inputs[{index}]: invalid_literal: program inputs"
            " must be declared table_input, array_input or parameter values; got"
            f" {value._node.op.name}"
        )
    return value


def _validated_inputs(
    inputs: Sequence[TableExpr | ArrayExpr | Parameter[object]], /
) -> tuple[TableExpr | ArrayExpr | Parameter[object], ...]:
    declared: dict[str, str] = {}
    copied: list[TableExpr | ArrayExpr | Parameter[object]] = []
    for index, candidate in enumerate(inputs):
        value = _input_declaration(candidate, index)
        input_name = _node_name(value._node)
        root = "static_inputs" if isinstance(value, Parameter) else "inputs"
        if input_name in declared:
            raise ValueError(
                f"{root}.{input_name}: duplicate_name: duplicate input name"
                f" {input_name!r}"
            )
        declared[input_name] = root
        copied.append(value)
    return tuple(copied)


def _discovered_inputs(
    outputs: tuple[tuple[str, TableExpr | ArrayExpr], ...], /
) -> tuple[TableExpr | ArrayExpr | Parameter[object], ...]:
    visited: set[bytes] = set()
    roots: list[TableExpr | ArrayExpr | Parameter[object]] = []

    def visit(node: Node) -> None:
        if node.node_bytes in visited:
            return
        visited.add(node.node_bytes)
        if node.op.name == "table_input":
            roots.append(TableExpr(node))
        elif node.op.name == "array_input":
            roots.append(ArrayExpr(node))
        elif node.op.name == "parameter":
            roots.append(Parameter(node))
        for child in node.args:
            visit(child)

    for _, value in outputs:
        visit(value._node)
    return _validated_inputs(roots)


def _validated_output(index: int, item: object, /) -> tuple[str, TableExpr | ArrayExpr]:
    if not isinstance(item, tuple) or len(item) != 2:
        raise TypeError(
            f"Program.outputs[{index}]: must be a (name, TableExpr |"
            f" ArrayExpr) pair; got {type_name(item)}"
        )
    output_name, value = item
    if type(output_name) is not str:
        raise TypeError(
            f"Program.outputs[{index}].name: must be a string; got"
            f" {type_name(output_name)}"
        )
    if not output_name:
        raise ValueError(
            f"Program.outputs[{index}].name: invalid_literal: must be a"
            " non-empty string"
        )
    if not isinstance(value, (TableExpr, ArrayExpr)):
        raise TypeError(
            f"Program.outputs[{index}].value: expected TableExpr |"
            f" ArrayExpr; got {type_name(value)}"
        )
    return output_name, value


def _validated_outputs(
    outputs: Mapping[str, TableExpr | ArrayExpr]
    | Sequence[tuple[str, TableExpr | ArrayExpr]],
    /,
) -> tuple[tuple[str, TableExpr | ArrayExpr], ...]:
    copied: list[tuple[str, TableExpr | ArrayExpr]] = []
    names: set[str] = set()
    entries = outputs.items() if isinstance(outputs, Mapping) else outputs
    for index, item in enumerate(entries):
        output_name, value = _validated_output(index, item)
        if output_name in names:
            raise ValueError(
                f"outputs.{output_name}: duplicate_name: duplicate output"
                f" name {output_name!r}"
            )
        names.add(output_name)
        copied.append((output_name, value))
    return tuple(copied)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Program:
    """An immutable program with its engine chosen before compilation."""

    _name: str
    _inputs: tuple[TableExpr | ArrayExpr | Parameter[object], ...]
    _outputs: tuple[tuple[str, TableExpr | ArrayExpr], ...]
    _fingerprint: str
    _engine: Literal["sql", "streaming"]
    _inputs_inferred: bool
    _update_semantics: UpdateSemantics

    @overload
    def __init__(
        self,
        name: str,
        /,
        *,
        mode: CompileMode,
        inputs: Sequence[TableExpr | ArrayExpr | Parameter[object]] | None = None,
        outputs: Mapping[str, TableExpr | ArrayExpr]
        | Sequence[tuple[str, TableExpr | ArrayExpr]] = (),
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        /,
        *,
        engine: Literal["sql", "streaming"],
        inputs: Sequence[TableExpr | ArrayExpr | Parameter[object]] | None = None,
        outputs: Mapping[str, TableExpr | ArrayExpr]
        | Sequence[tuple[str, TableExpr | ArrayExpr]] = (),
    ) -> None: ...

    def __init__(
        self,
        name: str,
        /,
        *,
        engine: object = _OMITTED,
        mode: object = _OMITTED,
        inputs: Sequence[TableExpr | ArrayExpr | Parameter[object]] | None = None,
        outputs: Mapping[str, TableExpr | ArrayExpr]
        | Sequence[tuple[str, TableExpr | ArrayExpr]] = (),
    ) -> None:
        if type(name) is not str:
            raise TypeError(f"Program.name must be a string; got {type_name(name)}")
        if not name:
            raise ValueError(
                "Program.name: invalid_literal: must be a non-empty string"
            )
        engine = _engine_selection(engine, mode)
        copied_outputs = _validated_outputs(outputs)
        _validate_engine_outputs(engine, copied_outputs)
        copied_inputs = (
            _discovered_inputs(copied_outputs)
            if inputs is None
            else _validated_inputs(inputs)
        )
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_inputs", copied_inputs)
        object.__setattr__(self, "_outputs", copied_outputs)
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_inputs_inferred", inputs is None)
        object.__setattr__(self, "_update_semantics", "legacy_append")
        object.__setattr__(
            self,
            "_fingerprint",
            _program_fingerprint(name, copied_inputs, copied_outputs),
        )

    @property
    def name(self) -> str:
        """The declared program name."""

        return self._name

    @property
    def inputs(self) -> tuple[TableExpr | ArrayExpr | Parameter[object], ...]:
        """The declared inputs in declaration order."""

        return self._inputs

    @property
    def outputs(self) -> tuple[tuple[str, TableExpr | ArrayExpr], ...]:
        """The declared outputs in declaration order."""

        return self._outputs

    @property
    def fingerprint(self) -> str:
        """The runtime-independent v1 program fingerprint."""

        return self._fingerprint

    @property
    def engine(self) -> Literal["sql", "streaming"]:
        """The immutable engine selected when this program was constructed."""

        return self._engine

    @property
    def mode(self) -> CompileMode:
        """The immutable execution lifecycle, independent of frontend syntax."""
        return "batch" if self._engine == "sql" else "stream"

    @property
    def update_semantics(self) -> UpdateSemantics:
        """How complete relation events are interpreted by this declaration."""
        return self._update_semantics

    def with_update_semantics(self, semantics: UpdateSemantics, /) -> Program:
        """Copy the update interpretation without planning or reading data."""
        if type(semantics) is not str:
            raise TypeError("Program.update_semantics: expected a string")
        if semantics not in ("legacy_append", "latest_relation"):
            raise ValueError("Program.update_semantics: invalid_literal: unknown value")
        candidate = Program(
            self._name,
            engine=self._engine,
            inputs=None if self._inputs_inferred else self._inputs,
            outputs=self._outputs,
        )
        object.__setattr__(candidate, "_update_semantics", semantics)
        if semantics == "latest_relation":
            identity = b"program-update-semantics-v1\0" + bytes.fromhex(
                candidate.fingerprint
            )
            object.__setattr__(
                candidate, "_fingerprint", hashlib.sha256(identity).hexdigest()
            )
        return candidate

    def for_mode(self, mode: CompileMode, /) -> Program:
        """Copy declarations into a fresh wrapper and validate the target mode."""
        return Program(
            self._name,
            mode=mode,
            inputs=None if self._inputs_inferred else self._inputs,
            outputs=self._outputs,
        ).with_update_semantics(self._update_semantics)

    def with_input(
        self, value: TableExpr | ArrayExpr | Parameter[object], /
    ) -> Program:
        """Return a new program with one declared input appended."""

        return Program(
            self._name,
            engine=self._engine,
            inputs=(*self._inputs, value),
            outputs=self._outputs,
        ).with_update_semantics(self._update_semantics)

    def _require_engine(self, engine: Literal["sql", "streaming"], /) -> None:
        if self._engine != engine:
            raise RuntimeError(
                f"Program engine is {self._engine!r}; cannot use {engine!r}"
            )

    def execute(
        self,
        inputs: Mapping[str, StreamInput | TableData | NativeArray],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
        portability: PortabilityRequest | None = None,
        **streaming: Unpack[_StreamingExecuteOptions],
    ) -> dict[str, pa.Table] | StreamResults[StreamOutput]:
        """Execute through the selected engine.

        SQL returns named Arrow tables immediately. Streaming returns owned
        named events; consume it with ``async with`` and ``async for``.
        Streaming accepts ``config`` and ``watermarks`` keyword controls.
        """

        config, watermarks = _stream_controls(**streaming)
        if self._engine == "sql":
            if config is not None:
                raise ValueError("execute.config is available for the streaming engine")
            if watermarks is not None:
                raise ValueError(
                    "execute.watermarks is available for the streaming engine"
                )
            return self.collect(
                cast("Mapping[str, TableData | NativeArray]", inputs),
                runtime=runtime,
                options=options,
                portability=portability,
            )
        if self._engine == "streaming":
            if options is not None:
                raise ValueError("execute.options is available for the sql engine")
            return self.stream(
                inputs,
                runtime=runtime,
                config=config,
                watermarks=watermarks,
                portability=portability,
            )
        raise RuntimeError(f"Program has unsupported engine {self._engine!r}")

    def stream(
        self,
        inputs: Mapping[str, StreamInput | TableData | NativeArray],
        /,
        *,
        runtime: Runtime | None = None,
        config: StreamRuntimeConfig | None = None,
        watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None = None,
        portability: PortabilityRequest | None = None,
    ) -> StreamResults[StreamOutput]:
        """Yield named output events from one owned native streaming job.

        Watermark policies use logical input names. Ordered event-time iterables
        advance safe watermarks by default; explicit policies support disorder or
        iterable-provided watermarks without changing SourceBinding policies.
        """
        self._require_engine("streaming")
        from calc_flow.stream import _stream_program, _StreamRequest

        return _stream_program(
            _StreamRequest(self, inputs, runtime, config, watermarks, portability)
        )

    def output(self, name: str, value: TableExpr | ArrayExpr, /) -> Program:
        """Return a new program with one declared output appended."""

        return Program(
            self._name,
            engine=self._engine,
            inputs=None if self._inputs_inferred else self._inputs,
            outputs=(*self._outputs, (name, value)),
        ).with_update_semantics(self._update_semantics)

    def analyze(
        self,
        runtime: Runtime | None = None,
        /,
        *,
        mode: CompileMode | None = None,
        options: AnalysisOptions | None = None,
    ) -> AnalysisResult:
        """Analyze using the selected mode or an explicit diagnostic mode."""

        from calc_flow.symbolic.analyzer import analyze_program

        selected = _selected_runtime(runtime)
        with selected._registration_lock:
            return analyze_program(self, selected, self._analysis_mode(mode), options)

    def collect_schema(
        self,
        runtime: Runtime | None = None,
        /,
        *,
        mode: CompileMode | None = None,
    ) -> dict[str, pa.Schema]:
        """Plan exact named table schemas without reading data or starting jobs."""
        from calc_flow.compute import _require_blocking
        from calc_flow.symbolic import errors
        from calc_flow.symbolic.lower import lower_program_document
        from calc_flow.symbolic.lower.schema import infer_document_schemas

        _require_blocking("collect_schema")
        for name, value in self._outputs:
            if not isinstance(value, TableExpr):
                errors.raise_compile(
                    f"{_segment('outputs', name)}.schema",
                    "unsupported_type",
                    "collect_schema requires table outputs",
                )
        if not self._outputs:
            errors.raise_compile(
                "outputs", "unresolved_type", "at least one output is required"
            )
        selected = _selected_runtime(runtime)
        with selected._registration_lock:
            selected_mode = self._analysis_mode(mode)
            _check_schema_output_types(self, selected, selected_mode)
            document = lower_program_document(self, selected, selected_mode)
            schemas = infer_document_schemas(document, selected)
            return {name: schemas[name] for name, _ in self._outputs}

    def explain(
        self,
        runtime: Runtime | None = None,
        /,
        *,
        mode: CompileMode | None = None,
        options: AnalysisOptions | None = None,
    ) -> str:
        """Render deterministic query facts and comparison premises."""
        from calc_flow.symbolic.analyzer import explain_program

        selected = _selected_runtime(runtime)
        with selected._registration_lock:
            return explain_program(self, selected, self._analysis_mode(mode), options)

    def _analysis_mode(self, mode: CompileMode | None, /) -> CompileMode:
        return (
            ("batch" if self._engine == "sql" else "stream") if mode is None else mode
        )

    def _selected_mode(self, mode: CompileMode | None, /) -> CompileMode:
        selected = "batch" if self._engine == "sql" else "stream"
        if mode is not None and mode != selected:
            raise ValueError(
                f"Program engine {self._engine!r} requires mode {selected!r}"
            )
        return selected

    def compile_batch(
        self,
        runtime: Runtime | None = None,
        /,
        *,
        portability: PortabilityRequest | None = None,
    ) -> BatchExecutionPlan:
        """Lower this program to a strict project-v3 batch execution plan.

        Compilation is declaration processing only: it captures one immutable
        capability snapshot, lowers one strict project-v3 document, and invokes
        the Rust graph compiler for final validation. No data, source, sink,
        or runner is accepted.
        """
        self._require_engine("sql")

        from calc_flow.symbolic.lower import compile_program_batch
        from calc_flow.symbolic.lower.program import _require_runtime
        from calc_flow.symbolic.query_analysis import _gate

        selected = (
            _require_runtime(runtime, "compile_batch")
            if runtime is not None
            else _selected_runtime(None)
        )
        with selected._registration_lock:
            _gate(self, selected, portability)
            return compile_program_batch(self, selected)

    def compile_stream(
        self,
        runtime: Runtime | None = None,
        /,
        *,
        allowed_lateness_micros: int = 0,
        late_policy: LatePolicy = "error",
        portability: PortabilityRequest | None = None,
    ) -> StreamExecutionPlan:
        """Lower this program to a strict project-v3 continuous plan.

        The lateness arguments configure unmarked rolling/cross-section stages;
        with_late_output declarations retain their local policy and lateness.
        Row-local-only programs accept the same compile signature but have no
        stateful late-row surface.
        """
        self._require_engine("streaming")

        from calc_flow.symbolic.lower import compile_program_stream
        from calc_flow.symbolic.lower.program import _require_runtime
        from calc_flow.symbolic.query_analysis import _gate

        selected = (
            _require_runtime(runtime, "compile_stream")
            if runtime is not None
            else _selected_runtime(None)
        )
        with selected._registration_lock:
            _gate(self, selected, portability)
            return compile_program_stream(
                self, selected, allowed_lateness_micros, late_policy
            )

    def collect(
        self,
        inputs: Mapping[str, TableData | NativeArray],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
        portability: PortabilityRequest | None = None,
    ) -> dict[str, pa.Table]:
        """Execute a fresh plan and return Arrow tables by declaration name."""
        self._require_engine("sql")
        from calc_flow.compute import _collect

        return _collect(self, inputs, runtime, options, portability)

    def collect_async(
        self,
        inputs: Mapping[str, TableData | NativeArray],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
        portability: PortabilityRequest | None = None,
    ) -> Awaitable[dict[str, pa.Table]]:
        """Copy input mappings, capture Batch references and await execution.

        Arrow buffers are shared; keep their underlying storage read-only until
        execution completes.
        """
        self._require_engine("sql")
        from calc_flow.compute import _collect_async

        return _collect_async(self, inputs, runtime, options, portability)

    def to_project(
        self,
        runtime: Runtime | None = None,
        /,
        *,
        mode: CompileMode | None = None,
        allowed_lateness_micros: int = 0,
        late_policy: LatePolicy = "error",
    ) -> ProjectDocument:
        """Export a strict native project; logical aliases stay in Python."""
        from calc_flow.config import ProjectDocument
        from calc_flow.symbolic.lower import lower_program_document

        return ProjectDocument.model_validate(
            lower_program_document(
                self,
                _selected_runtime(runtime),
                self._selected_mode(mode),
                allowed_lateness_micros=allowed_lateness_micros,
                late_policy=late_policy,
            )
        )


def _selected_runtime(runtime: Runtime | None, /) -> Runtime:
    from calc_flow.pipeline import Runtime

    if runtime is not None and not isinstance(runtime, Runtime):
        raise TypeError("expected a calc_flow Runtime or None")
    return Runtime() if runtime is None else runtime
