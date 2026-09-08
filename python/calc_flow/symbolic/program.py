"""Immutable feature sets and reusable programs over expression declarations.

Programs own declarations and canonical fingerprints. Analysis, compilation,
project export and collection share the native runtime and existing lowering.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from calc_flow.symbolic.domains import type_name
from calc_flow.symbolic.expr import ArrayExpr, ColumnExpr, Parameter, TableExpr
from calc_flow.symbolic.nodes import (
    _MAGIC,
    CStr,
    Node,
    _text,
    _u64,
)
from calc_flow.symbolic.types import CompileMode

if TYPE_CHECKING:
    import pyarrow as pa

    from calc_flow._native import ExecutionOptions
    from calc_flow.compute import TableData
    from calc_flow.config import ProjectDocument
    from calc_flow.pipeline import BatchExecutionPlan, Runtime, StreamExecutionPlan
    from calc_flow.symbolic.analyzer import AnalysisResult
    from calc_flow.symbolic.types import LatePolicy

_PROGRAM_TAG = 0x21


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
    inputs: tuple[TableExpr | Parameter[object], ...],
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
    inputs: tuple[TableExpr | Parameter[object], ...],
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
    inputs: tuple[TableExpr | Parameter[object], ...],
    /,
) -> list[bytes]:
    return [
        _text(_node_name(value._node)) + bytes.fromhex(value._node.digest)
        for value in inputs
    ]


def _validated_inputs(
    inputs: Sequence[TableExpr | Parameter[object]], /
) -> tuple[TableExpr | Parameter[object], ...]:
    declared: dict[str, str] = {}
    copied: list[TableExpr | Parameter[object]] = []
    for index, value in enumerate(inputs):
        if not isinstance(value, (TableExpr, Parameter)):
            raise TypeError(
                f"Program.inputs[{index}]: expected TableExpr |"
                f" Parameter[object]; got {type_name(value)}"
            )
        if value._node.op.name not in ("table_input", "parameter"):
            raise ValueError(
                f"Program.inputs[{index}]: invalid_literal: program inputs"
                " must be declared table_input or parameter values; got"
                f" {value._node.op.name}"
            )
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
) -> tuple[TableExpr | Parameter[object], ...]:
    visited: set[bytes] = set()
    roots: list[TableExpr | Parameter[object]] = []

    def visit(node: Node) -> None:
        if node.node_bytes in visited:
            return
        visited.add(node.node_bytes)
        if node.op.name == "table_input":
            roots.append(TableExpr(node))
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
    """An immutable program of declared inputs, outputs, and expressions."""

    _name: str
    _inputs: tuple[TableExpr | Parameter[object], ...]
    _outputs: tuple[tuple[str, TableExpr | ArrayExpr], ...]
    _fingerprint: str

    def __init__(
        self,
        name: str,
        /,
        *,
        inputs: Sequence[TableExpr | Parameter[object]] | None = None,
        outputs: Mapping[str, TableExpr | ArrayExpr]
        | Sequence[tuple[str, TableExpr | ArrayExpr]] = (),
    ) -> None:
        if type(name) is not str:
            raise TypeError(f"Program.name must be a string; got {type_name(name)}")
        if not name:
            raise ValueError(
                "Program.name: invalid_literal: must be a non-empty string"
            )
        copied_outputs = _validated_outputs(outputs)
        copied_inputs = (
            _discovered_inputs(copied_outputs)
            if inputs is None
            else _validated_inputs(inputs)
        )
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_inputs", copied_inputs)
        object.__setattr__(self, "_outputs", copied_outputs)
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
    def inputs(self) -> tuple[TableExpr | Parameter[object], ...]:
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

    def with_input(self, value: TableExpr | Parameter[object], /) -> Program:
        """Return a new program with one declared input appended."""

        return Program(
            self._name,
            inputs=(*self._inputs, value),
            outputs=self._outputs,
        )

    def output(self, name: str, value: TableExpr | ArrayExpr, /) -> Program:
        """Return a new program with one declared output appended."""

        return Program(
            self._name,
            inputs=self._inputs,
            outputs=(*self._outputs, (name, value)),
        )

    def analyze(
        self, runtime: Runtime | None = None, /, *, mode: CompileMode = "batch"
    ) -> AnalysisResult:
        """Analyze this program against one immutable capability snapshot."""

        from calc_flow.symbolic.analyzer import analyze_program

        return analyze_program(self, _selected_runtime(runtime), mode)

    def explain(
        self, runtime: Runtime | None = None, /, *, mode: CompileMode = "batch"
    ) -> str:
        """Render deterministic analysis facts for this program."""

        from calc_flow.symbolic.analyzer import explain_program

        return explain_program(self, _selected_runtime(runtime), mode)

    def compile_batch(self, runtime: Runtime | None = None, /) -> BatchExecutionPlan:
        """Lower this program to a strict project-v3 batch execution plan.

        Compilation is declaration processing only: it captures one immutable
        capability snapshot, lowers one strict project-v3 document, and invokes
        the Rust graph compiler for final validation. No data, source, sink,
        or runner is accepted.
        """

        from calc_flow.symbolic.lower import compile_program_batch

        return compile_program_batch(self, _selected_runtime(runtime))

    def compile_stream(
        self,
        runtime: Runtime | None = None,
        /,
        *,
        allowed_lateness_micros: int = 0,
        late_policy: LatePolicy = "error",
    ) -> StreamExecutionPlan:
        """Lower this program to a strict project-v3 continuous plan.

        The lateness arguments are validated and serialized into every lowered
        rolling or cross-section stage. Row-local-only programs accept the same
        compile signature but have no stateful late-row surface.
        """

        from calc_flow.symbolic.lower import compile_program_stream

        return compile_program_stream(
            self, _selected_runtime(runtime), allowed_lateness_micros, late_policy
        )

    def collect(
        self,
        inputs: Mapping[str, TableData],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
    ) -> dict[str, pa.Table]:
        """Execute a fresh plan and return Arrow tables by declaration name."""
        from calc_flow.compute import _collect

        return _collect(self, inputs, runtime, options)

    def collect_async(
        self,
        inputs: Mapping[str, TableData],
        /,
        *,
        runtime: Runtime | None = None,
        options: ExecutionOptions | None = None,
    ) -> Awaitable[dict[str, pa.Table]]:
        """Snapshot named inputs now and await independent native execution."""
        from calc_flow.compute import _collect_async

        return _collect_async(self, inputs, runtime, options)

    def to_project(
        self,
        runtime: Runtime | None = None,
        /,
        *,
        mode: CompileMode = "batch",
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
                mode,
                allowed_lateness_micros=allowed_lateness_micros,
                late_policy=late_policy,
            )
        )


def _selected_runtime(runtime: Runtime | None, /) -> Runtime:
    from calc_flow.pipeline import Runtime

    return Runtime() if runtime is None else runtime
