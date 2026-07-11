from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import pyarrow as pa

_UDF_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class UdfKind(StrEnum):
    DATAFUSION_SCALAR = "datafusion_scalar"
    ARRAY = "array"


class UdfVolatility(StrEnum):
    IMMUTABLE = "immutable"
    STABLE = "stable"
    VOLATILE = "volatile"


class UdfError(RuntimeError):
    """Base class for UDF registration, resolution, and execution failures."""


class DuplicateUdfError(UdfError):
    """Raised when the same UDF kind, name, and version is registered twice."""


class UnknownUdfError(UdfError):
    """Raised when a pipeline references an unregistered UDF version."""


class UdfVersionConflictError(UdfError):
    """Raised when one DataFusion run requests two versions of one SQL name."""


class UdfExecutionError(UdfError):
    """Raised when a registered implementation violates its declared contract."""


def _validate_identity(name: str, version: str) -> None:
    if not isinstance(name, str) or not isinstance(version, str):
        msg = "UDF name and version must be strings"
        raise TypeError(msg)
    if not _UDF_NAME_RE.fullmatch(name):
        msg = "UDF names must be lowercase SQL identifiers"
        raise ValueError(msg)
    if not _VERSION_RE.fullmatch(version):
        msg = f"invalid UDF version {version!r}"
        raise ValueError(msg)


def _as_input_field(value: pa.DataType | pa.Field, index: int) -> pa.Field:
    if isinstance(value, pa.Field):
        return value
    if isinstance(value, pa.DataType):
        return pa.field(f"arg{index}", value)
    msg = "UDF input fields must be pyarrow.Field or pyarrow.DataType values"
    raise TypeError(msg)


def _as_return_field(value: pa.DataType | pa.Field) -> pa.Field:
    if isinstance(value, pa.Field):
        return value
    if isinstance(value, pa.DataType):
        return pa.field("result", value)
    msg = "UDF return field must be a pyarrow.Field or pyarrow.DataType"
    raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class UdfReference:
    """Serializable reference to trusted code already installed in a registry."""

    name: str
    version: str

    def __post_init__(self) -> None:
        _validate_identity(self.name, self.version)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UdfReference:
        if set(data) != {"name", "version"}:
            msg = "UDF references require exactly name and version"
            raise ValueError(msg)
        name = data["name"]
        version = data["version"]
        if not isinstance(name, str) or not isinstance(version, str):
            msg = "UDF reference name and version must be strings"
            raise TypeError(msg)
        return cls(name=name, version=version)


@dataclass(frozen=True, slots=True)
class DataFusionScalarUdf:
    name: str
    version: str
    input_fields: tuple[pa.Field, ...]
    return_field: pa.Field
    volatility: UdfVolatility
    description: str
    implementation: Callable[..., Any]
    kind: UdfKind = UdfKind.DATAFUSION_SCALAR

    @property
    def reference(self) -> UdfReference:
        return UdfReference(self.name, self.version)

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "volatility": self.volatility.value,
            "parameters": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in self.input_fields
            ],
            "return": {
                "name": self.return_field.name,
                "type": str(self.return_field.type),
                "nullable": self.return_field.nullable,
            },
        }

    def invoke(self, *arguments: Any) -> pa.Array | pa.Scalar:
        if len(arguments) != len(self.input_fields):
            msg = (
                f"DataFusion UDF {self.name}@{self.version} expects "
                f"{len(self.input_fields)} arguments, got {len(arguments)}"
            )
            raise UdfExecutionError(msg)
        for argument, field in zip(arguments, self.input_fields, strict=True):
            if not isinstance(argument, pa.Array | pa.ChunkedArray | pa.Scalar):
                msg = (
                    f"DataFusion UDF {self.name}@{self.version} received "
                    "a non-Arrow argument"
                )
                raise UdfExecutionError(msg)
            if not argument.type.equals(field.type):
                msg = (
                    f"DataFusion UDF {self.name}@{self.version} received "
                    f"{argument.type} for {field.name}; expected {field.type}"
                )
                raise UdfExecutionError(msg)
            null_count = (
                argument.null_count
                if isinstance(argument, pa.Array | pa.ChunkedArray)
                else int(not argument.is_valid)
            )
            if not field.nullable and null_count:
                msg = (
                    f"DataFusion UDF {self.name}@{self.version} received nulls "
                    f"for non-nullable input {field.name}"
                )
                raise UdfExecutionError(msg)

        try:
            result = self.implementation(*arguments)
        except Exception as error:
            msg = f"DataFusion UDF {self.name}@{self.version} failed"
            raise UdfExecutionError(msg) from error

        if not isinstance(result, pa.Array | pa.ChunkedArray | pa.Scalar):
            msg = (
                f"DataFusion UDF {self.name}@{self.version} must return "
                "an Arrow array or scalar"
            )
            raise UdfExecutionError(msg)
        if not result.type.equals(self.return_field.type):
            msg = (
                f"DataFusion UDF {self.name}@{self.version} returned {result.type}; "
                f"expected {self.return_field.type}"
            )
            raise UdfExecutionError(msg)
        null_count = (
            result.null_count
            if isinstance(result, pa.Array | pa.ChunkedArray)
            else int(not result.is_valid)
        )
        if not self.return_field.nullable and null_count:
            msg = (
                f"DataFusion UDF {self.name}@{self.version} returned nulls for "
                "a non-nullable field"
            )
            raise UdfExecutionError(msg)

        input_lengths = [
            len(argument)
            for argument in arguments
            if isinstance(argument, pa.Array | pa.ChunkedArray)
        ]
        if input_lengths and isinstance(result, pa.Array | pa.ChunkedArray):
            expected_length = input_lengths[0]
            if any(length != expected_length for length in input_lengths):
                msg = (
                    f"DataFusion UDF {self.name}@{self.version} inputs differ in length"
                )
                raise UdfExecutionError(msg)
            if len(result) != expected_length:
                msg = (
                    f"DataFusion UDF {self.name}@{self.version} returned {len(result)} "
                    f"values for {expected_length} input rows"
                )
                raise UdfExecutionError(msg)

        if isinstance(result, pa.ChunkedArray):
            return result.combine_chunks()
        return result


@dataclass(frozen=True, slots=True)
class ArrayUdf:
    name: str
    version: str
    argument_count: int
    description: str
    implementation: Callable[..., Any]
    kind: UdfKind = UdfKind.ARRAY

    @property
    def reference(self) -> UdfReference:
        return UdfReference(self.name, self.version)

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "argument_count": self.argument_count,
        }

    def invoke(self, *arguments: Any, namespace: Any) -> Any:
        if len(arguments) != self.argument_count:
            msg = (
                f"array UDF {self.name}@{self.version} expects "
                f"{self.argument_count} arguments, got {len(arguments)}"
            )
            raise UdfExecutionError(msg)
        try:
            result = self.implementation(*arguments)
        except Exception as error:
            msg = f"array UDF {self.name}@{self.version} failed"
            raise UdfExecutionError(msg) from error

        array_namespace = getattr(result, "__array_namespace__", None)
        if not callable(array_namespace):
            msg = (
                f"array UDF {self.name}@{self.version} must return an Array API object"
            )
            raise UdfExecutionError(msg)
        result_namespace = array_namespace()
        if getattr(result_namespace, "__name__", None) != getattr(
            namespace, "__name__", None
        ):
            msg = f"array UDF {self.name}@{self.version} changed array backend"
            raise UdfExecutionError(msg)
        return result


class UdfRegistrySnapshot:
    """Read-only registry view captured when a pipeline is compiled."""

    __slots__ = ("_array", "_datafusion", "_frozen")

    def __init__(
        self,
        datafusion: Mapping[tuple[str, str], DataFusionScalarUdf] | None = None,
        array: Mapping[tuple[str, str], ArrayUdf] | None = None,
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(
            self, "_datafusion", MappingProxyType(dict(datafusion or {}))
        )
        object.__setattr__(self, "_array", MappingProxyType(dict(array or {})))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            msg = "UdfRegistrySnapshot is immutable"
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    @property
    def datafusion_specs(self) -> tuple[DataFusionScalarUdf, ...]:
        return tuple(self._datafusion.values())

    def resolve_datafusion(self, reference: UdfReference) -> DataFusionScalarUdf:
        try:
            return self._datafusion[(reference.name, reference.version)]
        except KeyError as error:
            msg = f"unknown DataFusion UDF {reference.name}@{reference.version}"
            raise UnknownUdfError(msg) from error

    def resolve_array(self, reference: UdfReference) -> ArrayUdf:
        try:
            return self._array[(reference.name, reference.version)]
        except KeyError as error:
            msg = f"unknown array UDF {reference.name}@{reference.version}"
            raise UnknownUdfError(msg) from error

    def array_functions(
        self, references: Iterable[UdfReference]
    ) -> Mapping[str, ArrayUdf]:
        functions: dict[str, ArrayUdf] = {}
        for reference in references:
            specification = self.resolve_array(reference)
            existing = functions.get(reference.name)
            if existing is not None and existing.version != reference.version:
                msg = (
                    f"array expression requests both {reference.name}@"
                    f"{existing.version} and {reference.name}@{reference.version}"
                )
                raise UdfVersionConflictError(msg)
            functions[reference.name] = specification
        return MappingProxyType(functions)

    def select(
        self,
        *,
        datafusion: Iterable[UdfReference] = (),
        array: Iterable[UdfReference] = (),
    ) -> UdfRegistrySnapshot:
        selected_datafusion: dict[tuple[str, str], DataFusionScalarUdf] = {}
        versions: dict[str, str] = {}
        for reference in datafusion:
            specification = self.resolve_datafusion(reference)
            existing_version = versions.get(reference.name)
            if existing_version is not None and existing_version != reference.version:
                msg = (
                    f"DataFusion run requests both {reference.name}@{existing_version} "
                    f"and {reference.name}@{reference.version}"
                )
                raise UdfVersionConflictError(msg)
            versions[reference.name] = reference.version
            selected_datafusion[(reference.name, reference.version)] = specification

        selected_array = {
            (reference.name, reference.version): self.resolve_array(reference)
            for reference in array
        }
        return UdfRegistrySnapshot(selected_datafusion, selected_array)

    def catalog(self) -> tuple[dict[str, Any], ...]:
        entries = [
            *(
                specification.catalog_entry()
                for specification in self._datafusion.values()
            ),
            *(specification.catalog_entry() for specification in self._array.values()),
        ]
        return tuple(
            sorted(
                entries,
                key=lambda entry: (entry["kind"], entry["name"], entry["version"]),
            )
        )


class UdfRegistry:
    """Mutable application registry for trusted, vectorized UDF implementations."""

    def __init__(self) -> None:
        self._datafusion: dict[tuple[str, str], DataFusionScalarUdf] = {}
        self._array: dict[tuple[str, str], ArrayUdf] = {}

    def datafusion_scalar(
        self,
        *,
        name: str,
        version: str,
        input_fields: Sequence[pa.DataType | pa.Field],
        return_field: pa.DataType | pa.Field,
        volatility: UdfVolatility | str = UdfVolatility.IMMUTABLE,
        description: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        _validate_identity(name, version)
        if not isinstance(description, str):
            msg = "UDF description must be a string"
            raise TypeError(msg)
        fields = tuple(
            _as_input_field(value, index) for index, value in enumerate(input_fields)
        )
        if not fields:
            msg = "DataFusion scalar UDFs require at least one input field"
            raise ValueError(msg)
        field_names = [field.name for field in fields]
        if len(field_names) != len(set(field_names)):
            msg = "DataFusion UDF input field names must be unique"
            raise ValueError(msg)
        output = _as_return_field(return_field)
        try:
            resolved_volatility = UdfVolatility(volatility)
        except ValueError as error:
            msg = f"invalid UDF volatility {volatility!r}"
            raise ValueError(msg) from error

        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(function):
                msg = "UDF implementation must be callable"
                raise TypeError(msg)
            key = (name, version)
            if key in self._datafusion:
                msg = f"DataFusion UDF {name}@{version} is already registered"
                raise DuplicateUdfError(msg)
            self._datafusion[key] = DataFusionScalarUdf(
                name=name,
                version=version,
                input_fields=fields,
                return_field=output,
                volatility=resolved_volatility,
                description=description,
                implementation=function,
            )
            return function

        return decorator

    def array(
        self,
        *,
        name: str,
        version: str,
        argument_count: int,
        description: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        _validate_identity(name, version)
        if not isinstance(description, str):
            msg = "UDF description must be a string"
            raise TypeError(msg)
        if isinstance(argument_count, bool) or not isinstance(argument_count, int):
            msg = "array UDF argument_count must be an integer"
            raise TypeError(msg)
        if argument_count <= 0:
            msg = "array UDF argument_count must be greater than 0"
            raise ValueError(msg)

        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(function):
                msg = "UDF implementation must be callable"
                raise TypeError(msg)
            key = (name, version)
            if key in self._array:
                msg = f"array UDF {name}@{version} is already registered"
                raise DuplicateUdfError(msg)
            self._array[key] = ArrayUdf(
                name=name,
                version=version,
                argument_count=argument_count,
                description=description,
                implementation=function,
            )
            return function

        return decorator

    def snapshot(self) -> UdfRegistrySnapshot:
        return UdfRegistrySnapshot(self._datafusion, self._array)

    def catalog(self) -> tuple[dict[str, Any], ...]:
        return self.snapshot().catalog()
