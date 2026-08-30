from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Never

from calc_flow import _native
from calc_flow.capabilities import (
    CapabilityRule,
    ProviderArrayRules,
    ProviderOption,
    ProviderOptionsSchema,
)

if TYPE_CHECKING:
    from calc_flow.pipeline import Runtime

_MAX_AST_NODES = 128
_MAX_AST_DEPTH = 24
_MAX_EXPRESSION_LENGTH = 4096
_MAX_INTEGER_MAGNITUDE = 2**63 - 1
_MAX_POWER_EXPONENT_MAGNITUDE = 64
_MAX_RESHAPE_RANK = 16
_MAX_RESHAPE_DIMENSION = 1_000_000
_MAX_RESHAPE_ELEMENTS = 10_000_000
_MAX_OPERATION_ELEMENTS = 10_000_000
_TABLE_MATMUL_INPUT_PORTS = (("table", "table"), ("weights", "array"))
_TABLE_MATMUL_OUTPUT_PORTS = (("output", "array"),)
_SYMBOLIC_MATRIX_INPUT_PORTS = (("input", "table"), ("weights", "array"))
_SYMBOLIC_MATRIX_OUTPUT_PORTS = (("output", "table"),)
_EXPRESSION_OPTIONS_SCHEMA = ProviderOptionsSchema(
    fields=(ProviderOption("expression", "string", required=True),)
)
_NUMPY_STREAM_ARRAY_RULES = ProviderArrayRules(
    supported_dtypes=(
        "bool",
        "complex128",
        "complex64",
        "float32",
        "float64",
        "int16",
        "int32",
        "int64",
        "int8",
        "uint16",
        "uint32",
        "uint64",
        "uint8",
    ),
    safe_dtype_rule=CapabilityRule("array_api_safe_dtype", "1"),
    shape_rules=(CapabilityRule("elementwise_broadcast", "1"),),
)
_JAX_STREAM_ARRAY_RULES = ProviderArrayRules(
    supported_dtypes=(
        "bool",
        "complex64",
        "float32",
        "int16",
        "int32",
        "int8",
        "uint16",
        "uint32",
        "uint8",
    ),
    safe_dtype_rule=CapabilityRule("array_api_safe_dtype", "1"),
    shape_rules=(CapabilityRule("elementwise_broadcast", "1"),),
)
_NUMPY_MATRIX_STREAM_ARRAY_RULES = ProviderArrayRules(
    supported_dtypes=_NUMPY_STREAM_ARRAY_RULES.supported_dtypes,
    safe_dtype_rule=_NUMPY_STREAM_ARRAY_RULES.safe_dtype_rule,
    shape_rules=(
        CapabilityRule("elementwise_broadcast", "1"),
        CapabilityRule("table_matmul_static_rhs", "1"),
    ),
)
_JAX_MATRIX_STREAM_ARRAY_RULES = ProviderArrayRules(
    supported_dtypes=_JAX_STREAM_ARRAY_RULES.supported_dtypes,
    safe_dtype_rule=_JAX_STREAM_ARRAY_RULES.safe_dtype_rule,
    shape_rules=(
        CapabilityRule("elementwise_broadcast", "1"),
        CapabilityRule("table_matmul_static_rhs", "1"),
    ),
)

_ALLOWED_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.MatMult: operator.matmul,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_UNARY_FUNCTIONS = {"sum", "mean", "max", "min", "transpose"}
_ALLOWED_FUNCTIONS = _UNARY_FUNCTIONS | {"reshape"}


def _array_error(message: str) -> ValueError:
    return ValueError(f"invalid array expression: {message}")


def _parse_expression(expression: object) -> ast.Expression:
    if not isinstance(expression, str) or not expression.strip():
        raise _array_error("expression must be a non-empty string")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise _array_error(
            f"expression length limit is {_MAX_EXPRESSION_LENGTH} characters"
        )
    return _parse_valid_expression(expression)


@lru_cache(maxsize=256)
def _parse_valid_expression(expression: str) -> ast.Expression:
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise _array_error("syntax is invalid") from error
    if not isinstance(parsed, ast.Expression):
        raise _array_error("syntax is invalid")
    nodes = list(ast.walk(parsed))
    if len(nodes) > _MAX_AST_NODES:
        raise _array_error(f"node limit is {_MAX_AST_NODES}")
    if _ast_depth(parsed) > _MAX_AST_DEPTH:
        raise _array_error(f"depth limit is {_MAX_AST_DEPTH}")
    _validate_node(parsed.body)
    return parsed


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(child) for child in children)


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.Name):
        if node.id != "x":
            raise _array_error(f"unknown name {node.id!r}")
        return
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise _array_error("constants must be finite numbers")
        if type(node.value) is int and abs(node.value) > _MAX_INTEGER_MAGNITUDE:
            raise _array_error(
                f"integer constant magnitude limit is {_MAX_INTEGER_MAGNITUDE}"
            )
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise _array_error("constants must be finite numbers")
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
        _validate_node(node.left)
        if isinstance(node.op, ast.Pow):
            exponent = _numeric_literal(node.right)
            if exponent is None:
                raise _array_error("power exponent must be a finite numeric literal")
            if abs(exponent) > _MAX_POWER_EXPONENT_MAGNITUDE:
                raise _array_error(
                    f"power exponent magnitude limit is {_MAX_POWER_EXPONENT_MAGNITUDE}"
                )
        _validate_node(node.right)
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        _validate_node(node.operand)
        return
    if isinstance(node, ast.Call):
        _validate_call(node)
        return
    raise _array_error(f"unsupported syntax {type(node).__name__}")


def _numeric_literal(node: ast.AST) -> int | float | None:
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
        return None
    if isinstance(node.value, float) and not math.isfinite(node.value):
        return None
    return sign * node.value


def _validate_call(node: ast.Call) -> None:
    if not isinstance(node.func, ast.Name):
        raise _array_error("functions must be called directly")
    name = node.func.id
    if name not in _ALLOWED_FUNCTIONS:
        raise _array_error(f"unknown function {name!r}")
    if node.keywords:
        raise _array_error("function keyword arguments are unsupported")
    expected = 1 if name in _UNARY_FUNCTIONS else 2
    if len(node.args) != expected:
        suffix = "" if expected == 1 else "s"
        raise _array_error(f"{name} expects {expected} argument{suffix}")
    _validate_node(node.args[0])
    if name == "reshape":
        _reshape_shape(node.args[1])


def _reshape_shape(node: ast.AST) -> tuple[int, ...]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise _array_error("reshape shape must be a tuple or list")
    if len(node.elts) > _MAX_RESHAPE_RANK:
        raise _array_error(f"reshape rank limit is {_MAX_RESHAPE_RANK}")
    dimensions: list[int] = []
    for item in node.elts:
        value: object
        if isinstance(item, ast.Constant):
            value = item.value
        elif (
            isinstance(item, ast.UnaryOp)
            and isinstance(item.op, ast.USub)
            and isinstance(item.operand, ast.Constant)
            and type(item.operand.value) is int
        ):
            value = -item.operand.value
        else:
            raise _array_error("reshape dimensions must be integers")
        if type(value) is not int:
            raise _array_error("reshape dimensions must be integers")
        dimensions.append(value)
    if dimensions.count(-1) > 1:
        raise _array_error("reshape shape allows at most one -1")
    if -1 in dimensions and 0 in dimensions:
        raise _array_error("reshape cannot combine a zero dimension with -1")
    if any(dimension < -1 for dimension in dimensions):
        raise _array_error("reshape dimensions must be non-negative or -1")
    if any(dimension > _MAX_RESHAPE_DIMENSION for dimension in dimensions):
        raise _array_error(f"reshape dimension limit is {_MAX_RESHAPE_DIMENSION}")
    known_elements = math.prod(dimension for dimension in dimensions if dimension != -1)
    if known_elements > _MAX_RESHAPE_ELEMENTS:
        raise _array_error(f"reshape output limit is {_MAX_RESHAPE_ELEMENTS} elements")
    return tuple(dimensions)


def _validate_reshape_result(value: object) -> None:
    dimensions = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    if any(dimension > _MAX_RESHAPE_DIMENSION for dimension in dimensions):
        raise _array_error(f"reshape dimension limit is {_MAX_RESHAPE_DIMENSION}")
    if math.prod(dimensions) > _MAX_RESHAPE_ELEMENTS:
        raise _array_error(f"reshape output limit is {_MAX_RESHAPE_ELEMENTS} elements")


def _evaluate(
    node: ast.AST,
    value: object,
    namespace: Any,
    validate_result: Callable[[object], None] | None = None,
) -> object:
    if isinstance(node, ast.Name):
        # The input batch is validated at construction; only operation
        # results pass through validate_result.
        return value
    if isinstance(node, ast.Constant):
        result = node.value
    elif isinstance(node, ast.BinOp):
        function = _ALLOWED_BINARY[type(node.op)]
        left = _evaluate(node.left, value, namespace, validate_result)
        right = _evaluate(node.right, value, namespace, validate_result)
        if isinstance(node.op, ast.MatMult):
            _validate_matmul_output_size(left, right)
        else:
            _validate_broadcast_output_size(left, right)
        result = function(left, right)
    elif isinstance(node, ast.UnaryOp):
        function = _ALLOWED_UNARY[type(node.op)]
        result = function(_evaluate(node.operand, value, namespace, validate_result))
    elif isinstance(node, ast.Call):
        name = node.func.id  # type: ignore[union-attr]
        function = getattr(namespace, name)
        arguments = [_evaluate(node.args[0], value, namespace, validate_result)]
        if name == "reshape":
            arguments.append(_reshape_shape(node.args[1]))
        result = function(*arguments)
        if name == "reshape":
            _validate_reshape_result(result)
    else:
        raise AssertionError("validated array expression contained an unsupported node")
    if validate_result is not None:
        validate_result(result)
    return result


def _validate_options(options: Mapping[str, object]) -> ast.Expression:
    unknown = set(options) - {"expression", "udfs"}
    if unknown:
        raise _array_error(f"unsupported options: {', '.join(sorted(unknown))}")
    udfs = options.get("udfs", [])
    if udfs:
        raise _array_error("custom array UDFs are unavailable")
    return _parse_expression(options.get("expression"))


def _validate_stream_options(options: Mapping[str, object]) -> None:
    parsed = _validate_options(options)
    if not any(
        isinstance(node, ast.Name) and node.id == "x" for node in ast.walk(parsed)
    ):
        raise _array_error("stream expressions must depend on the input rows")
    if any(isinstance(node, (ast.Call, ast.MatMult)) for node in ast.walk(parsed)):
        raise _array_error(
            "stream expressions support only row-axis-independent "
            "elementwise operations"
        )


def _validate_provider_options(
    provider: str, name: str, version: str, options: Mapping[str, object]
) -> None:
    if provider in {"numpy", "jax"} and name == "expression" and version == "1":
        _validate_options(options)


def _table_matmul_columns(options: Mapping[str, object]) -> tuple[str, ...]:
    unknown = set(options) - {"columns"}
    if unknown:
        raise ValueError(
            "invalid table_matmul options: unsupported options: "
            + ", ".join(sorted(unknown))
        )
    columns = options.get("columns")
    if isinstance(columns, (str, bytes)) or not isinstance(columns, list):
        raise ValueError("invalid table_matmul options: columns must be a JSON array")
    if not columns:
        raise ValueError(
            "invalid table_matmul options: columns must contain at least one name"
        )
    if not all(isinstance(column, str) and column for column in columns):
        raise ValueError(
            "invalid table_matmul options: columns must contain non-empty strings"
        )
    if len(set(columns)) != len(columns):
        raise ValueError("invalid table_matmul options: columns must be unique")
    return tuple(columns)


def _table_matmul_inputs(
    inputs: Mapping[str, _native.Batch],
    backend: str,
) -> tuple[_native.Batch, _native.Batch]:
    copied = dict(inputs)
    expected = {"table", "weights"}
    missing = expected - set(copied)
    unexpected = set(copied) - expected
    if missing:
        raise ValueError(
            "invalid table_matmul inputs: missing required inputs: "
            + ", ".join(sorted(missing))
        )
    if unexpected:
        raise ValueError(
            "invalid table_matmul inputs: unsupported inputs: "
            + ", ".join(sorted(unexpected))
        )
    table_batch = copied["table"]
    weights_batch = copied["weights"]
    if not isinstance(table_batch, _native.Batch) or table_batch.kind != "table":
        received = getattr(table_batch, "kind", type(table_batch).__name__)
        raise TypeError(
            f"invalid table_matmul table: expected a table batch, received {received}"
        )
    if not isinstance(weights_batch, _native.Batch) or weights_batch.kind != "array":
        received = getattr(weights_batch, "kind", type(weights_batch).__name__)
        raise TypeError(
            "invalid table_matmul weights: "
            f"expected an array batch, received {received}"
        )
    if weights_batch.backend != backend:
        raise ValueError(
            "invalid table_matmul weights.backend: "
            f"expected {backend}, received {weights_batch.backend}"
        )
    return table_batch, weights_batch


def _validated_table_dtypes(
    table: object,
    columns: tuple[str, ...],
) -> tuple[object, ...]:
    import numpy as np
    import pyarrow as pa

    if table.num_rows <= 0:
        raise ValueError("invalid table_matmul table.rows: expected at least one row")
    dtypes: list[object] = []
    for name in columns:
        indices = table.schema.get_all_field_indices(name)
        if not indices:
            raise ValueError(
                f"invalid table_matmul columns: selected column {name!r} is missing"
            )
        if len(indices) != 1:
            raise ValueError(
                f"invalid table_matmul columns: selected column {name!r} is ambiguous"
            )
        column = table.column(indices[0])
        if column.null_count:
            raise ValueError(
                f"invalid table_matmul columns: selected column {name!r} contains nulls"
            )
        data_type = column.type
        if not (pa.types.is_integer(data_type) or pa.types.is_floating(data_type)):
            raise TypeError(
                "invalid table_matmul columns: "
                f"selected column {name!r} has unsupported Arrow dtype {data_type}"
            )
        family = (
            "int"
            if pa.types.is_signed_integer(data_type)
            else "uint"
            if pa.types.is_unsigned_integer(data_type)
            else "float"
        )
        dtypes.append(np.dtype(f"{family}{data_type.bit_width}"))
    return tuple(dtypes)


def _validated_weights(
    weights_batch: _native.Batch,
    backend: str,
    column_count: int,
) -> object:
    weights = weights_batch.array
    if backend == "numpy":
        import numpy as np

        if type(weights) is not np.ndarray:
            raise TypeError(
                "invalid table_matmul weights: NumPy weights must be an ndarray"
            )
    shape = getattr(weights, "shape", ())
    if len(shape) != 2:
        raise ValueError(
            "invalid table_matmul weights.rank: "
            f"expected rank two, received rank {len(shape)}"
        )
    if shape[0] != column_count:
        raise ValueError(
            "invalid table_matmul weights.shape[0]: "
            f"expected {column_count}, received {shape[0]}"
        )
    if shape[1] <= 0:
        raise ValueError(
            "invalid table_matmul weights.shape[1]: expected a positive output width"
        )
    return weights


def _common_matrix_dtype(
    backend: str,
    namespace: object,
    table_dtypes: tuple[object, ...],
    weights: object,
) -> object:
    import numpy as np

    weight_dtype = np.dtype(weights.dtype)
    involved = ", ".join(
        [*(np.dtype(dtype).name for dtype in table_dtypes), weight_dtype.name]
    )
    if backend == "jax":
        import jax

        x64_dtypes = frozenset(
            np.dtype(dtype)
            for dtype in (np.int64, np.uint64, np.float64, np.complex128)
        )
        if not jax.config.x64_enabled and any(
            np.dtype(source) in x64_dtypes for source in (*table_dtypes, weight_dtype)
        ):
            raise TypeError(
                "invalid table_matmul dtype: "
                f"JAX x64 is disabled for [{involved}]; enable the required dtype "
                "or choose a lossless supported dtype"
            )
    try:
        result_dtype = np.dtype(namespace.result_type(*table_dtypes, weight_dtype))
    except (TypeError, ValueError) as error:
        raise TypeError(
            "invalid table_matmul dtype: "
            f"{backend} cannot promote Arrow and weight dtypes [{involved}]"
        ) from error
    sources = (*table_dtypes, weight_dtype)
    if not all(
        namespace.can_cast(source, result_dtype, casting="safe") for source in sources
    ):
        raise TypeError(
            "invalid table_matmul dtype: "
            f"common dtype {result_dtype.name} is lossy for [{involved}]"
        )
    try:
        _validate_numpy_dtype(result_dtype)
    except ValueError as error:
        raise TypeError(
            "invalid table_matmul dtype: "
            f"common dtype {result_dtype.name} is unsupported for [{involved}]"
        ) from error
    return result_dtype


def _numpy_table_matrix(
    table: object,
    columns: tuple[str, ...],
    dtype: object,
) -> object:
    import numpy as np

    matrix, _token = _native.Batch._new_owned_numpy(
        (table.num_rows, len(columns)),
        np.dtype(dtype).name,
    )
    for column_index, name in enumerate(columns):
        offset = 0
        for chunk in table[name].chunks:
            values = chunk.to_numpy(zero_copy_only=True)
            next_offset = offset + len(values)
            np.copyto(
                matrix[offset:next_offset, column_index],
                values,
                casting="safe",
            )
            offset = next_offset
    return matrix


def _owned_numpy(value: object) -> object:
    import numpy as np

    array = np.asarray(value)
    _validate_numpy_dtype(array.dtype)
    immutable_bytes = array.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=array.dtype).reshape(array.shape)


def _validate_numpy_dtype(dtype: object) -> None:
    import numpy as np

    normalized = np.dtype(dtype)
    allowed_dtypes = frozenset(
        np.dtype(scalar_type)
        for scalar_type in (
            np.bool_,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
    )
    if normalized not in allowed_dtypes or not normalized.isnative:
        raise ValueError(
            f"NumPy arrays require a NumPy Array API dtype; received {normalized}"
        )


def _validate_python_operation_scalar(value: object) -> bool:
    if type(value) is int:
        if abs(value) > _MAX_INTEGER_MAGNITUDE:
            raise _array_error(
                f"integer constant magnitude limit is {_MAX_INTEGER_MAGNITUDE}"
            )
        return True
    return type(value) in (float, complex)


def _result_shape(value: object) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in getattr(value, "shape", ()))


def _validate_operation_output_size(value: object) -> None:
    _validate_operation_shape(_result_shape(value))


def _validate_operation_shape(shape: tuple[int, ...]) -> None:
    if math.prod(shape) > _MAX_OPERATION_ELEMENTS:
        raise _array_error(
            f"operation output limit is {_MAX_OPERATION_ELEMENTS} elements"
        )


def _broadcast_shape(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...] | None:
    dimensions: list[int] = []
    for offset in range(1, max(len(left), len(right)) + 1):
        left_dim = left[-offset] if offset <= len(left) else 1
        right_dim = right[-offset] if offset <= len(right) else 1
        if left_dim == 1:
            dimensions.append(right_dim)
        elif right_dim == 1 or left_dim == right_dim:
            dimensions.append(left_dim)
        else:
            return None
    return tuple(reversed(dimensions))


def _matmul_batch_shape(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...] | None:
    return _broadcast_shape(
        left[:-2] if len(left) > 1 else (),
        right[:-2] if len(right) > 1 else (),
    )


def _matmul_result_axes(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    if len(left) == 1:
        return () if len(right) == 1 else (right[-1],)
    if len(right) == 1:
        return (left[-2],)
    return (left[-2], right[-1])


def _matmul_output_shape(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...] | None:
    if not left or not right:
        return None
    right_inner = right[-2] if len(right) > 1 else right[-1]
    if left[-1] != right_inner:
        return None
    batch = _matmul_batch_shape(left, right)
    if batch is None:
        return None
    return (*batch, *_matmul_result_axes(left, right))


def _validate_broadcast_output_size(left: object, right: object) -> None:
    shape = _broadcast_shape(_result_shape(left), _result_shape(right))
    if shape is not None:
        _validate_operation_shape(shape)


def _validate_matmul_output_size(left: object, right: object) -> None:
    shape = _matmul_output_shape(_result_shape(left), _result_shape(right))
    if shape is not None:
        _validate_operation_shape(shape)


def _validate_numpy_operation_result(value: object) -> None:
    import numpy as np

    if _validate_python_operation_scalar(value):
        return
    if type(value) is np.ndarray or isinstance(value, np.generic):
        _validate_numpy_dtype(value.dtype)
        _validate_operation_output_size(value)
        return
    raise TypeError("NumPy provider operations must produce arrays or numeric scalars")


def _validate_jax_operation_result(value: object) -> None:
    import jax

    if _validate_python_operation_scalar(value):
        return
    if isinstance(value, jax.Array):
        _validate_operation_output_size(value)
        return
    raise TypeError("JAX provider operations must produce arrays or numeric scalars")


def _owned_jax(value: object) -> object:
    import jax
    import jax.numpy as jnp

    array = jnp.asarray(value)
    if not isinstance(array, jax.Array):
        raise TypeError("JAX batches require a jax.Array payload")
    return array


def _prepare_array(value: object, backend: str) -> tuple[object, int]:
    if backend == "numpy":
        owned = _owned_numpy(value)
    elif backend == "jax":
        owned = _owned_jax(value)
    else:
        owned = value
    shape = getattr(owned, "shape", None)
    length = int(shape[0]) if shape else 1
    if length < 0:
        raise ValueError("array length must not be negative")
    return owned, length


@dataclass(frozen=True, slots=True)
class _ArrayProvider:
    backend: str
    namespace: object

    def validate(self, options: Mapping[str, object]) -> None:
        _validate_options(options)

    def validate_stream(self, options: Mapping[str, object]) -> None:
        _validate_stream_options(options)

    def __call__(
        self, batch: _native.Batch, options: Mapping[str, object]
    ) -> _native.Batch:
        if batch.backend != self.backend:
            raise TypeError(
                f"provider requires backend {self.backend}, received {batch.backend}"
            )
        parsed = _validate_options(options)
        validate_result = {
            "jax": _validate_jax_operation_result,
            "numpy": _validate_numpy_operation_result,
        }[self.backend]
        result = _evaluate(parsed.body, batch.array, self.namespace, validate_result)
        if self.backend == "jax":
            import jax

            result = _owned_jax(result)
            if not isinstance(result, jax.Array):
                raise TypeError("JAX provider output must remain a jax.Array")
        return _native.Batch.from_array(
            result, backend=self.backend, metadata=batch.metadata
        )


def _jax_table_matmul(
    table: object,
    columns: tuple[str, ...],
    dtype: object,
    weights: object,
) -> tuple[object, None]:
    import jax
    import jax.numpy as jnp

    host = _numpy_table_matrix(table, columns, dtype)
    dense = jax.device_put(host, device=weights.device)
    expected_dtype = jnp.dtype(dtype)
    if dense.dtype != expected_dtype:
        raise ValueError(
            "invalid table_matmul dtype: JAX changed "
            f"{expected_dtype} to {dense.dtype}; enable the required dtype "
            "or choose a lossless supported dtype"
        )
    result = jnp.matmul(dense, weights)
    if not isinstance(result, jax.Array):
        raise TypeError("table_matmul JAX result must remain a jax.Array")
    if result.dtype != expected_dtype:
        raise ValueError(
            "invalid table_matmul dtype: JAX changed "
            f"{expected_dtype} to {result.dtype}; enable the required dtype "
            "or choose a lossless supported dtype"
        )
    return result, None


@dataclass(frozen=True, slots=True)
class _TableMatmulProvider:
    backend: str
    namespace: object

    def validate(self, options: Mapping[str, object]) -> None:
        _table_matmul_columns(options)

    def __call__(
        self,
        inputs: Mapping[str, _native.Batch],
        options: Mapping[str, object],
    ) -> dict[str, _native.Batch]:
        columns = _table_matmul_columns(options)
        table_batch, weights_batch = _table_matmul_inputs(inputs, self.backend)
        table = table_batch.to_pyarrow()
        table_dtypes = _validated_table_dtypes(table, columns)
        weights = _validated_weights(weights_batch, self.backend, len(columns))
        result_dtype = _common_matrix_dtype(
            self.backend,
            self.namespace,
            table_dtypes,
            weights,
        )
        if self.backend != "numpy":
            output, token = _jax_table_matmul(
                table,
                columns,
                result_dtype,
                weights,
            )
        else:
            import numpy as np

            dense = _numpy_table_matrix(table, columns, result_dtype)
            output, token = _native.Batch._new_owned_numpy(
                (table.num_rows, weights.shape[1]),
                np.dtype(result_dtype).name,
            )
            np.matmul(dense, weights, out=output)

        metadata = table_batch.metadata
        metadata.update(
            {
                "backend": self.backend,
                "columns": list(columns),
                "operation": "table_matmul",
            }
        )
        return {
            "output": _native.Batch._from_owned_array(
                output,
                backend=self.backend,
                token=token,
                metadata=metadata,
            )
        }


_SYMBOLIC_BINARY = {
    "add": operator.add,
    "and": operator.and_,
    "eq": operator.eq,
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lt": operator.lt,
    "mul": operator.mul,
    "ne": operator.ne,
    "or": operator.or_,
    "sub": operator.sub,
    "truediv": operator.truediv,
}
_SYMBOLIC_UNARY = {"neg": operator.neg, "not": operator.invert}


def _symbolic_typed_operand(
    backend: str,
    data_type: str,
    *,
    matrix: bool,
) -> object:
    import numpy as np

    if backend == "numpy":
        namespace = np
    elif backend == "jax":
        import jax
        import jax.numpy as jnp

        if not jax.config.x64_enabled and np.dtype(data_type) in {
            np.dtype("int64"),
            np.dtype("uint64"),
            np.dtype("float64"),
            np.dtype("complex128"),
        }:
            raise TypeError(f"{backend} cannot represent dtype {data_type!r}")
        namespace = jnp
    else:
        raise TypeError(f"unsupported symbolic array backend {backend!r}")
    shape = (1, 1) if matrix else (1,)
    value = namespace.ones(shape, dtype=data_type)
    if np.dtype(value.dtype).name != data_type:
        raise TypeError(f"{backend} cannot represent dtype {data_type!r}")
    return value


def _symbolic_result_dtype(
    backend: str,
    operation: str,
    left: str | bool | int | float,
    right: str | bool | int | float | None = None,
) -> str:
    """Return the selected provider's v1 dtype for one symbolic primitive."""

    import numpy as np

    matrix = operation == "matmul"
    left_value = (
        _symbolic_typed_operand(backend, left, matrix=matrix)
        if isinstance(left, str)
        else left
    )
    right_value = (
        _symbolic_typed_operand(backend, right, matrix=matrix)
        if isinstance(right, str)
        else right
    )
    typed_values = tuple(
        value for value in (left_value, right_value) if hasattr(value, "dtype")
    )
    if len(typed_values) == 2:
        if backend == "numpy":
            namespace = np
        else:
            import jax.numpy as jnp

            namespace = jnp
        common = namespace.result_type(
            typed_values[0].dtype,
            typed_values[1].dtype,
        )
        if not all(
            namespace.can_cast(value.dtype, common, casting="safe")
            for value in typed_values
        ):
            raise TypeError("the provider has no safe common dtype")
    if operation in _SYMBOLIC_UNARY:
        result = _SYMBOLIC_UNARY[operation](left_value)
    elif operation == "matmul":
        result = operator.matmul(left_value, right_value)
    else:
        result = _SYMBOLIC_BINARY[operation](left_value, right_value)
    return np.dtype(result.dtype).name


def _raise_invalid_symbolic_node(node: Mapping[str, object]) -> Never:
    raise ValueError(
        f"invalid symbolic_matrix expression: unsupported node {node.get('op')!r}"
    )


def _validate_symbolic_leaf(node: dict[str, object], _depth: int) -> None:
    if set(node) != {"op"}:
        _raise_invalid_symbolic_node(node)


def _validate_symbolic_literal(node: dict[str, object], _depth: int) -> None:
    if set(node) != {"op", "value"}:
        _raise_invalid_symbolic_node(node)
    literal = node["value"]
    if type(literal) not in (bool, int, float):
        raise ValueError("invalid symbolic_matrix expression: literal must be finite")
    if type(literal) is float and not math.isfinite(literal):
        raise ValueError("invalid symbolic_matrix expression: literal must be finite")


def _validate_symbolic_unary(node: dict[str, object], depth: int) -> None:
    if set(node) != {"op", "value"}:
        _raise_invalid_symbolic_node(node)
    _validated_symbolic_tree(node["value"], depth=depth + 1)


def _validate_symbolic_binary(node: dict[str, object], depth: int) -> None:
    if set(node) != {"left", "op", "right"}:
        _raise_invalid_symbolic_node(node)
    _validated_symbolic_tree(node["left"], depth=depth + 1)
    _validated_symbolic_tree(node["right"], depth=depth + 1)


_SYMBOLIC_NODE_VALIDATORS: dict[str, Callable[[dict[str, object], int], None]] = {
    "input": _validate_symbolic_leaf,
    "literal": _validate_symbolic_literal,
    "weights": _validate_symbolic_leaf,
    **{
        operation: _validate_symbolic_binary
        for operation in (*_SYMBOLIC_BINARY, "matmul")
    },
    **{operation: _validate_symbolic_unary for operation in _SYMBOLIC_UNARY},
}


def _validated_symbolic_tree(value: object, *, depth: int = 0) -> dict[str, object]:
    if depth > _MAX_AST_DEPTH:
        raise ValueError("invalid symbolic_matrix expression: depth limit exceeded")
    if not isinstance(value, Mapping):
        raise ValueError("invalid symbolic_matrix expression: node must be a mapping")
    node = dict(value)
    validator = _SYMBOLIC_NODE_VALIDATORS.get(node.get("op"))
    if validator is None:
        _raise_invalid_symbolic_node(node)
    validator(node, depth)
    return node


def _raise_invalid_symbolic_names(field: str) -> Never:
    raise ValueError(
        f"invalid symbolic_matrix {field}: expected unique non-empty strings"
    )


def _symbolic_name(value: object, field: str) -> str:
    if type(value) is not str:
        _raise_invalid_symbolic_names(field)
    if not value:
        _raise_invalid_symbolic_names(field)
    return value


def _symbolic_names(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _raise_invalid_symbolic_names(field)
    if not value:
        _raise_invalid_symbolic_names(field)
    names = tuple(_symbolic_name(name, field) for name in value)
    if len(set(names)) != len(names):
        _raise_invalid_symbolic_names(field)
    return names


def _symbolic_matrix_options(
    options: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, object]]:
    if set(options) != {"columns", "expression", "names"}:
        raise ValueError(
            "invalid symbolic_matrix options: expected columns, expression, and names"
        )
    columns = _symbolic_names(options["columns"], "columns")
    names = _symbolic_names(options["names"], "names")
    return columns, names, _validated_symbolic_tree(options["expression"])


def _symbolic_matrix_inputs(
    inputs: Mapping[str, _native.Batch], backend: str
) -> tuple[_native.Batch, _native.Batch]:
    if set(inputs) != {"input", "weights"}:
        raise ValueError("invalid symbolic_matrix inputs: expected input and weights")
    table_batch = inputs["input"]
    weights_batch = inputs["weights"]
    if table_batch.kind != "table":
        raise ValueError("invalid symbolic_matrix input: expected a table batch")
    if weights_batch.kind != "array" or weights_batch.backend != backend:
        raise ValueError(f"invalid symbolic_matrix weights.backend: expected {backend}")
    return table_batch, weights_batch


def _evaluate_symbolic_tree(
    node: Mapping[str, object],
    namespace: object,
    dense: object,
    weights: object,
) -> object:
    operation = node["op"]
    if operation == "input":
        return dense
    if operation == "weights":
        return weights
    if operation == "literal":
        return node["value"]
    if operation in _SYMBOLIC_UNARY:
        result = _SYMBOLIC_UNARY[operation](
            _evaluate_symbolic_tree(node["value"], namespace, dense, weights)
        )
    else:
        left = _evaluate_symbolic_tree(node["left"], namespace, dense, weights)
        right = _evaluate_symbolic_tree(node["right"], namespace, dense, weights)
        if operation == "matmul":
            _validate_matmul_output_size(left, right)
            result = namespace.matmul(left, right)
        else:
            _validate_broadcast_output_size(left, right)
            result = _SYMBOLIC_BINARY[operation](left, right)
    _validate_operation_output_size(result)
    return result


def _symbolic_tree_dtype(
    node: Mapping[str, object],
    backend: str,
    input_dtype: str,
    weights_dtype: str,
) -> str | bool | int | float:
    operation = node["op"]
    if operation == "input":
        return input_dtype
    if operation == "weights":
        return weights_dtype
    if operation == "literal":
        literal = node["value"]
        if type(literal) in (bool, int, float):
            return literal
        raise TypeError("invalid symbolic_matrix literal dtype")
    child = "value" if operation in _SYMBOLIC_UNARY else "left"
    left = _symbolic_tree_dtype(
        node[child],
        backend,
        input_dtype,
        weights_dtype,
    )
    right = (
        None
        if operation in _SYMBOLIC_UNARY
        else _symbolic_tree_dtype(
            node["right"],
            backend,
            input_dtype,
            weights_dtype,
        )
    )
    return _symbolic_result_dtype(backend, operation, left, right)


@dataclass(frozen=True, slots=True)
class _SymbolicMatrixProvider:
    backend: str
    namespace: object

    def validate(self, options: Mapping[str, object]) -> None:
        _symbolic_matrix_options(options)

    def validate_stream(self, options: Mapping[str, object]) -> None:
        _symbolic_matrix_options(options)

    def __call__(
        self,
        inputs: Mapping[str, _native.Batch],
        options: Mapping[str, object],
    ) -> dict[str, _native.Batch]:
        import numpy as np
        import pyarrow as pa

        columns, names, expression = _symbolic_matrix_options(options)
        table_batch, weights_batch = _symbolic_matrix_inputs(inputs, self.backend)
        table = table_batch.to_pyarrow()
        table_dtypes = _validated_table_dtypes(table, columns)
        weights = _validated_weights(weights_batch, self.backend, len(columns))
        result_dtype = _common_matrix_dtype(
            self.backend, self.namespace, table_dtypes, weights
        )
        expected_dtype = _symbolic_tree_dtype(
            expression,
            self.backend,
            np.dtype(result_dtype).name,
            np.dtype(weights.dtype).name,
        )
        if not isinstance(expected_dtype, str):
            raise TypeError("invalid symbolic_matrix output.dtype: unresolved dtype")
        host = _numpy_table_matrix(table, columns, result_dtype)
        if self.backend == "jax":
            import jax

            dense = jax.device_put(host, device=weights.device)
        else:
            dense = host
        result = _evaluate_symbolic_tree(
            expression,
            self.namespace,
            dense,
            weights,
        )
        result_shape = _result_shape(result)
        if len(result_shape) != 2:
            raise ValueError(
                "invalid symbolic_matrix output.rank: expected rank two; "
                f"received {len(result_shape)}"
            )
        if result_shape[0] != table.num_rows:
            raise ValueError(
                "invalid symbolic_matrix output.rows: expected "
                f"{table.num_rows}, received {result_shape[0]}"
            )
        if result_shape[1] != len(names):
            raise ValueError(
                "invalid symbolic_matrix output.width: expected "
                f"{len(names)}, received {result_shape[1]}"
            )
        output_host = np.asarray(result)
        actual_dtype = np.dtype(output_host.dtype).name
        if actual_dtype != expected_dtype:
            raise TypeError(
                "invalid symbolic_matrix output.dtype: expected "
                f"{expected_dtype}, received {actual_dtype}"
            )
        attached = table
        for index, name in enumerate(names):
            attached = attached.append_column(name, pa.array(output_host[:, index]))
        metadata = table_batch.metadata
        metadata.update(
            {
                "backend": self.backend,
                "copy_bytes": {
                    "array_to_table": int(output_host.nbytes),
                    "table_to_array": int(host.nbytes),
                    "weights": int(
                        weights_batch.metadata.get("static_placement_bytes", 0)
                    ),
                },
                "operation": "symbolic_matrix",
                "provider_calls": 1,
            }
        )
        return {"output": _native.Batch.from_pyarrow(attached, metadata)}


def register_numpy(runtime: Runtime) -> None:
    import numpy as np

    expression = _ArrayProvider("numpy", np)
    runtime.register_provider(
        "numpy",
        "expression",
        "1",
        expression,
        options_schema=_EXPRESSION_OPTIONS_SCHEMA,
    )
    runtime._register_stateless_stream_provider(
        "numpy",
        "expression",
        "1",
        expression,
        microbatch_invariant=True,
        deterministic=True,
        replay_safe=True,
        supports_static_inputs=False,
        array_rules=_NUMPY_STREAM_ARRAY_RULES,
    )
    runtime._register_mapping_provider(
        "numpy",
        "table_matmul",
        "1",
        _TableMatmulProvider("numpy", np),
        input_ports=_TABLE_MATMUL_INPUT_PORTS,
        output_ports=_TABLE_MATMUL_OUTPUT_PORTS,
    )
    symbolic_matrix = _SymbolicMatrixProvider("numpy", np)
    runtime._register_mapping_provider(
        "numpy",
        "symbolic_matrix",
        "1",
        symbolic_matrix,
        input_ports=_SYMBOLIC_MATRIX_INPUT_PORTS,
        output_ports=_SYMBOLIC_MATRIX_OUTPUT_PORTS,
    )
    runtime._register_stateless_stream_mapping_provider(
        "numpy",
        "symbolic_matrix",
        "1",
        symbolic_matrix,
        microbatch_invariant=True,
        deterministic=True,
        replay_safe=True,
        supports_static_inputs=True,
        array_rules=_NUMPY_MATRIX_STREAM_ARRAY_RULES,
    )


def register_jax(runtime: Runtime) -> None:
    import jax.numpy as jnp

    expression = _ArrayProvider("jax", jnp)
    runtime.register_provider(
        "jax",
        "expression",
        "1",
        expression,
        options_schema=_EXPRESSION_OPTIONS_SCHEMA,
    )
    runtime._register_stateless_stream_provider(
        "jax",
        "expression",
        "1",
        expression,
        microbatch_invariant=True,
        deterministic=True,
        replay_safe=True,
        supports_static_inputs=False,
        array_rules=_JAX_STREAM_ARRAY_RULES,
    )
    runtime._register_mapping_provider(
        "jax",
        "table_matmul",
        "1",
        _TableMatmulProvider("jax", jnp),
        input_ports=_TABLE_MATMUL_INPUT_PORTS,
        output_ports=_TABLE_MATMUL_OUTPUT_PORTS,
    )
    symbolic_matrix = _SymbolicMatrixProvider("jax", jnp)
    runtime._register_mapping_provider(
        "jax",
        "symbolic_matrix",
        "1",
        symbolic_matrix,
        input_ports=_SYMBOLIC_MATRIX_INPUT_PORTS,
        output_ports=_SYMBOLIC_MATRIX_OUTPUT_PORTS,
    )
    runtime._register_stateless_stream_mapping_provider(
        "jax",
        "symbolic_matrix",
        "1",
        symbolic_matrix,
        microbatch_invariant=True,
        deterministic=True,
        replay_safe=True,
        supports_static_inputs=True,
        array_rules=_JAX_MATRIX_STREAM_ARRAY_RULES,
    )
