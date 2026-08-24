"""Immutable feature sets and programs over symbolic declarations.

``FeatureSet`` and ``Program`` are the program-level declaration values of the
frozen public surface. A program owns its declared inputs and outputs, the
``calc_flow.symbolic.declaration.v1`` program fingerprint, and the declaration
processing entry points ``analyze``/``explain``. Compilation to execution plans
is a later lowering stage and is deliberately absent here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
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
    from calc_flow.pipeline import Runtime
    from calc_flow.symbolic.analyzer import AnalysisResult

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
    node_records = sorted(
        ((bytes.fromhex(digest), record) for digest, record in nodes.items()),
        key=lambda item: item[0],
    )
    body = (
        bytes((_PROGRAM_TAG,))
        + _text(name)
        + _u64(len(inputs))
        + b"".join(
            _text(_node_name(value._node)) + bytes.fromhex(value._node.digest)
            for value in inputs
        )
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


def _validated_outputs(
    outputs: Sequence[tuple[str, TableExpr | ArrayExpr]], /
) -> tuple[tuple[str, TableExpr | ArrayExpr], ...]:
    copied: list[tuple[str, TableExpr | ArrayExpr]] = []
    names: set[str] = set()
    for index, item in enumerate(outputs):
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
        inputs: Sequence[TableExpr | Parameter[object]] = (),
        outputs: Sequence[tuple[str, TableExpr | ArrayExpr]] = (),
    ) -> None:
        if type(name) is not str:
            raise TypeError(f"Program.name must be a string; got {type_name(name)}")
        if not name:
            raise ValueError(
                "Program.name: invalid_literal: must be a non-empty string"
            )
        copied_inputs = _validated_inputs(inputs)
        copied_outputs = _validated_outputs(outputs)
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

    def analyze(self, runtime: Runtime, /, *, mode: CompileMode) -> AnalysisResult:
        """Analyze this program against one immutable capability snapshot."""

        from calc_flow.symbolic.analyzer import analyze_program

        return analyze_program(self, runtime, mode)

    def explain(self, runtime: Runtime, /, *, mode: CompileMode) -> str:
        """Render deterministic analysis facts for this program."""

        from calc_flow.symbolic.analyzer import explain_program

        return explain_program(self, runtime, mode)
