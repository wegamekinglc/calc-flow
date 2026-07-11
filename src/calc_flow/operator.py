from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from calc_flow.batch import Batch, BatchKind
from calc_flow.context import RunContext
from calc_flow.engine.array import JaxEngine, NumpyEngine
from calc_flow.expression import sql_projection
from calc_flow.udf import UdfReference

_PORT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class Port:
    """A named, typed operator input or output."""

    name: str
    kind: BatchKind
    required: bool = True
    schema: pa.Schema | None = None

    def __post_init__(self) -> None:
        if not _PORT_NAME_RE.fullmatch(self.name):
            msg = f"invalid port name {self.name!r}"
            raise ValueError(msg)
        if self.schema is not None and self.kind is not BatchKind.TABLE:
            msg = "only table ports may declare an Arrow schema"
            raise ValueError(msg)

    def validate(self, batch: Batch, *, endpoint: str) -> None:
        if not isinstance(batch, Batch):
            msg = f"{endpoint} requires a Batch"
            raise TypeError(msg)
        if batch.kind is not self.kind:
            msg = (
                f"{endpoint} requires a {self.kind.value} batch, got {batch.kind.value}"
            )
            raise TypeError(msg)
        if self.schema is not None and not batch.table_payload.schema.equals(
            self.schema, check_metadata=True
        ):
            msg = f"{endpoint} produced an unexpected Arrow schema"
            raise TypeError(msg)

    def fingerprint_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "required": self.required,
            "schema": self.schema.serialize().to_pybytes().hex()
            if self.schema is not None
            else None,
        }


ProcessFunction = Callable[[Mapping[str, Batch], RunContext], Mapping[str, Batch]]


def _udf_references(values: Iterable[UdfReference]) -> tuple[UdfReference, ...]:
    references = tuple(values)
    if not all(isinstance(reference, UdfReference) for reference in references):
        msg = "operator UDFs must be UdfReference values"
        raise TypeError(msg)
    return references


class Operator(ABC):
    """Base class for a graph node with named input and output ports."""

    def __init__(
        self,
        name: str,
        *,
        input_ports: Iterable[Port] = (Port("input", BatchKind.TABLE),),
        output_ports: Iterable[Port] = (Port("output", BatchKind.TABLE),),
    ) -> None:
        if not name:
            msg = "operator name must not be empty"
            raise ValueError(msg)
        self.name = name
        self.input_ports = tuple(input_ports)
        self.output_ports = tuple(output_ports)
        self._validate_unique_ports(self.input_ports, "input")
        self._validate_unique_ports(self.output_ports, "output")

    @staticmethod
    def _validate_unique_ports(ports: tuple[Port, ...], direction: str) -> None:
        names = [port.name for port in ports]
        if len(names) != len(set(names)):
            msg = f"operator has duplicate {direction} port names"
            raise ValueError(msg)

    @abstractmethod
    def process(
        self, inputs: Mapping[str, Batch], context: RunContext
    ) -> Mapping[str, Batch]: ...

    def reset(self) -> None:
        """Reset internal state after recovery or a requested reset."""
        return None

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-compatible internal state for checkpointing."""
        return {}

    def restore(self, state: dict[str, Any]) -> None:
        """Restore internal state from a checkpoint."""
        return None

    def configuration(self) -> Mapping[str, Any]:
        """Return stable configuration included in pipeline fingerprints."""
        return {}

    def datafusion_udfs(self) -> tuple[UdfReference, ...]:
        return ()

    def array_udfs(self) -> tuple[UdfReference, ...]:
        return ()

    def fingerprint_data(self) -> dict[str, Any]:
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "inputs": [port.fingerprint_data() for port in self.input_ports],
            "outputs": [port.fingerprint_data() for port in self.output_ports],
            "configuration": dict(self.configuration()),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class StatelessOperator(Operator):
    """A pure graph transform constructed from a mapping-based callable."""

    def __init__(
        self,
        name: str,
        fn: ProcessFunction | None = None,
        *,
        input_ports: Iterable[Port] = (Port("input", BatchKind.TABLE),),
        output_ports: Iterable[Port] = (Port("output", BatchKind.TABLE),),
        fingerprint_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name, input_ports=input_ports, output_ports=output_ports)
        self._fn = fn
        self._fingerprint_config = dict(fingerprint_config or {})

    def process(
        self, inputs: Mapping[str, Batch], context: RunContext
    ) -> Mapping[str, Batch]:
        if self._fn is not None:
            return self._fn(inputs, context)
        msg = f"{type(self).__name__} must override process or provide fn"
        raise NotImplementedError(msg)

    def configuration(self) -> Mapping[str, Any]:
        if self._fn is None:
            return self._fingerprint_config
        return {
            "callable": (
                f"{getattr(self._fn, '__module__', '')}."
                f"{getattr(self._fn, '__qualname__', type(self._fn).__qualname__)}"
            ),
            **self._fingerprint_config,
        }


class StatefulOperator(Operator):
    """An operator that maintains JSON-compatible state across batches."""

    def __init__(
        self,
        name: str,
        *,
        input_ports: Iterable[Port] = (Port("input", BatchKind.TABLE),),
        output_ports: Iterable[Port] = (Port("output", BatchKind.TABLE),),
    ) -> None:
        super().__init__(name, input_ports=input_ports, output_ports=output_ports)
        self._state: dict[str, Any] = {}

    @abstractmethod
    def process(
        self, inputs: Mapping[str, Batch], context: RunContext
    ) -> Mapping[str, Batch]: ...

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def restore(self, state: dict[str, Any]) -> None:
        self._state = deepcopy(state)

    def reset(self) -> None:
        self._state.clear()


class ExpressionOperator(StatelessOperator):
    """Projection, calculation, or filter over one DataFusion table input."""

    def __init__(
        self,
        name: str,
        expression: str | None = None,
        *,
        select: Iterable[str] | None = None,
        filter_expression: str | None = None,
        udfs: Iterable[UdfReference] = (),
        input_schema: pa.Schema | None = None,
        output_schema: pa.Schema | None = None,
    ) -> None:
        projections = tuple(select or ())
        if (expression is None) == (not projections):
            msg = "provide exactly one expression or non-empty select list"
            raise ValueError(msg)
        super().__init__(
            name,
            input_ports=(Port("input", BatchKind.TABLE, schema=input_schema),),
            output_ports=(Port("output", BatchKind.TABLE, schema=output_schema),),
        )
        self.expression = expression
        self.select = projections
        self.filter_expression = filter_expression
        self.udfs = _udf_references(udfs)

    def process(
        self, inputs: Mapping[str, Batch], context: RunContext
    ) -> Mapping[str, Batch]:
        context.check_cancelled()
        if self.expression is not None:
            query = sql_projection(self.expression, "__input__")
        else:
            query = f"SELECT {', '.join(self.select)} FROM __input__"
        if self.filter_expression is not None:
            query = f"{query} WHERE ({self.filter_expression})"
        result = context.datafusion.sql(
            query, {"__input__": inputs["input"]}, node_id=context.node_id
        )
        context.check_cancelled()
        return {"output": result}

    def configuration(self) -> Mapping[str, Any]:
        return {
            "expression": self.expression,
            "select": list(self.select),
            "filter_expression": self.filter_expression,
            "udfs": [reference.to_dict() for reference in self.udfs],
        }

    def datafusion_udfs(self) -> tuple[UdfReference, ...]:
        return self.udfs


class SqlOperator(StatelessOperator):
    """A multi-input table query executed by the run's DataFusion session."""

    def __init__(
        self,
        name: str,
        query: str,
        *,
        inputs: Iterable[str],
        udfs: Iterable[UdfReference] = (),
        input_schemas: Mapping[str, pa.Schema] | None = None,
        output_schema: pa.Schema | None = None,
    ) -> None:
        aliases = tuple(inputs)
        if not aliases:
            msg = "SqlOperator requires at least one input alias"
            raise ValueError(msg)
        schemas = input_schemas or {}
        unknown_schemas = set(schemas) - set(aliases)
        if unknown_schemas:
            msg = f"schemas declared for unknown SQL inputs: {sorted(unknown_schemas)}"
            raise ValueError(msg)
        super().__init__(
            name,
            input_ports=tuple(
                Port(alias, BatchKind.TABLE, schema=schemas.get(alias))
                for alias in aliases
            ),
            output_ports=(Port("output", BatchKind.TABLE, schema=output_schema),),
        )
        self.query = query
        self.aliases = aliases
        self.udfs = _udf_references(udfs)

    def process(
        self, inputs: Mapping[str, Batch], context: RunContext
    ) -> Mapping[str, Batch]:
        context.check_cancelled()
        result = context.datafusion.sql(
            self.query,
            {alias: inputs[alias] for alias in self.aliases},
            node_id=context.node_id,
        )
        context.check_cancelled()
        return {"output": result}

    def configuration(self) -> Mapping[str, Any]:
        return {
            "query": self.query,
            "inputs": list(self.aliases),
            "udfs": [reference.to_dict() for reference in self.udfs],
        }

    def datafusion_udfs(self) -> tuple[UdfReference, ...]:
        return self.udfs


class ArrayExpressionOperator(StatelessOperator):
    """A registered-UDF-aware expression over one NumPy or JAX array batch."""

    def __init__(
        self,
        name: str,
        expression: str,
        *,
        backend: str,
        udfs: Iterable[UdfReference] = (),
    ) -> None:
        if backend == "numpy":
            engine = NumpyEngine()
        elif backend == "jax":
            engine = JaxEngine()
        else:
            msg = "array expression backend must be 'numpy' or 'jax'"
            raise ValueError(msg)
        super().__init__(
            name,
            input_ports=(Port("input", BatchKind.ARRAY),),
            output_ports=(Port("output", BatchKind.ARRAY),),
        )
        self.expression = expression
        self.backend = backend
        self.udfs = _udf_references(udfs)
        self._engine = engine

    def process(
        self, inputs: Mapping[str, Batch], context: RunContext
    ) -> Mapping[str, Batch]:
        context.check_cancelled()
        functions = context.udfs.array_functions(self.udfs)
        result = self._engine.evaluate(self.expression, inputs["input"], udfs=functions)
        context.check_cancelled()
        return {"output": result}

    def configuration(self) -> Mapping[str, Any]:
        return {
            "expression": self.expression,
            "backend": self.backend,
            "udfs": [reference.to_dict() for reference in self.udfs],
        }

    def array_udfs(self) -> tuple[UdfReference, ...]:
        return self.udfs
