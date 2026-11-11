"""Table and native-array collection through the single native runtime."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar, Union, cast

import pyarrow as pa

from calc_flow._compat import TypeAliasType, dataclass
from calc_flow._native import Batch, ExecutionOptions, NativeArray, RunResult
from calc_flow._native_array import ArrayExecutionOptions
from calc_flow.pipeline import BatchExecutionPlan, Runtime, _canonical
from calc_flow.symbolic import errors
from calc_flow.symbolic.expr import ArrayExpr, Parameter, TableExpr, table_input
from calc_flow.symbolic.lower.bindings import _BatchBindings
from calc_flow.symbolic.native_arrays import is_native_array
from calc_flow.symbolic.nodes import CDType, CInt, CSeq, CStr
from calc_flow.symbolic.program import Program, _node_name, _selected_runtime
from calc_flow.symbolic.query_analysis import PortabilityRequest, _gate

TableData = TypeAliasType("TableData", Union[pa.Table, pa.RecordBatch, Batch])
T = TypeVar("T")


def _require_blocking(entry: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    message = f"{entry}() cannot run inside an event loop"
    if entry != "collect_array":
        message += f"; use {entry}_async()"
    raise RuntimeError(message)


def _arrow_table(data: TableData, path: str) -> pa.Table | pa.RecordBatch:
    if isinstance(data, Batch):
        if data.kind != "table":
            raise TypeError(f"{path}: expected table input, got {data.kind} Batch")
        return data.to_pyarrow()
    if isinstance(data, (pa.Table, pa.RecordBatch)):
        return data
    raise TypeError(
        f"{path}: expected Arrow Table, RecordBatch or table Batch; "
        f"got {type(data).__name__}"
    )


def _table_batch(data: TableData, path: str) -> Batch:
    table = _arrow_table(data, path)
    if table.schema.metadata is not None or any(
        field.metadata is not None for field in table.schema
    ):
        # Symbolic schemas omit Arrow metadata; rewrap the same column buffers.
        schema = pa.schema([field.remove_metadata() for field in table.schema])
        if table.num_columns:
            normalized = pa.Table.from_arrays(table.columns, schema=schema)
        else:
            # RecordBatch metadata replacement preserves zero-column row counts.
            batches = table.to_batches() if isinstance(table, pa.Table) else [table]
            normalized = pa.Table.from_batches(
                [batch.replace_schema_metadata(None) for batch in batches],
                schema=schema,
            )
        return Batch.from_pyarrow(
            normalized, metadata=data.metadata if isinstance(data, Batch) else None
        )
    return data if isinstance(data, Batch) else Batch.from_pyarrow(table)


def _input_kind(value: TableExpr | ArrayExpr | Parameter[object]) -> str:
    if isinstance(value, Parameter):
        return "static parameter"
    return "array input" if isinstance(value, ArrayExpr) else "table input"


def _check_input_names(
    provided: Mapping[str, object],
    expected: Mapping[str, T],
    root: str,
    describe: Callable[[T], str],
) -> None:
    for name, value in expected.items():
        if name not in provided:
            raise ValueError(f"{root}.{name}: missing {describe(value)}")
    for name in provided:
        if name not in expected:
            raise ValueError(f"{root}.{name}: unexpected input name")


def _validated_inputs(
    inputs: Mapping[str, TableData | NativeArray],
    expected: Mapping[str, TableExpr | ArrayExpr | Parameter[object]],
) -> dict[str, TableData | NativeArray]:
    if not isinstance(inputs, Mapping):
        raise TypeError("collect.inputs: expected a mapping by declared input name")
    copied = dict(inputs)
    _check_input_names(copied, expected, "inputs", _input_kind)
    return copied


def _input_batch(
    data: TableData | NativeArray,
    value: TableExpr | ArrayExpr | Parameter[object],
    path: str,
) -> Batch:
    if _is_native_declaration(value):
        return _native_input_batch(data, value, path)
    if isinstance(value, Parameter) and value.kind == "array":
        if not isinstance(data, Batch) or data.kind != "array":
            raise TypeError(
                f"{path}: expected array Batch; use Batch.from_array "
                "and register its provider"
            )
        return data
    return _table_batch(cast(TableData, data), path)


def _is_native_declaration(value: TableExpr | ArrayExpr | Parameter[object]) -> bool:
    if isinstance(value, ArrayExpr):
        return True
    backend = value._node.attr("backend")
    return (
        isinstance(value, Parameter)
        and value.kind == "array"
        and isinstance(backend, CStr)
        and backend.value == "native"
    )


def _native_input_batch(
    data: TableData | NativeArray,
    declaration: TableExpr | ArrayExpr | Parameter[object],
    path: str,
) -> Batch:
    if isinstance(data, NativeArray):
        batch = Batch.from_array(data, backend="native")
    elif isinstance(data, Batch):
        batch = data
    else:
        raise TypeError(f"{path}.backend: expected NativeArray or native array Batch")
    if batch.kind != "array" or batch.backend != "native":
        raise TypeError(f"{path}.backend: expected a native array payload")
    array = batch.array
    if not isinstance(array, NativeArray):
        raise TypeError(f"{path}.backend: expected an actual native array payload")
    _validate_native_descriptor(array, declaration, path)
    return batch


def _validate_native_descriptor(
    array: NativeArray,
    declaration: TableExpr | ArrayExpr | Parameter[object],
    path: str,
) -> None:
    dtype = declaration._node.attr("dtype")
    if not isinstance(dtype, CDType) or array.dtype != dtype.name:
        raise ValueError(f"{path}.dtype: native dtype differs from its declaration")
    shape = declaration._node.attr("shape")
    if not isinstance(shape, CSeq) or len(shape.items) != len(array.shape):
        raise ValueError(f"{path}.shape: native rank differs from its declaration")
    for index, dimension in enumerate(shape.items):
        if isinstance(dimension, CInt) and dimension.value != array.shape[index]:
            raise ValueError(
                f"{path}.shape: native extent differs from its declaration"
            )


def _collect_batches(
    program: Program, inputs: Mapping[str, TableData | NativeArray]
) -> dict[str, Batch]:
    expected = {_node_name(value._node): value for value in program.inputs}
    copied = _validated_inputs(inputs, expected)
    return {
        name: _input_batch(copied[name], value, f"inputs.{name}")
        for name, value in expected.items()
    }


def _prepare_collect(
    program: Program,
    inputs: Mapping[str, TableData | NativeArray],
    runtime: Runtime | None,
    portability: PortabilityRequest | None = None,
) -> tuple[BatchExecutionPlan, dict[str, Batch], dict[str, str]]:
    if portability is None and program.update_semantics == "legacy_append":
        batches = _collect_batches(program, inputs)
        return _prepare_collect_batches(program, batches, _selected_runtime(runtime))
    selected = _selected_runtime(runtime)
    with selected._registration_lock:
        _gate(program, selected, portability)
        batches = _collect_batches(program, inputs)
        return _prepare_collect_batches(program, batches, selected)


@dataclass(frozen=True, slots=True)
class _LoweredCollect:
    """Immutable lowering template shared by repeated collects of one program."""

    project_json: str
    input_names: tuple[tuple[str, tuple[str, ...]], ...]
    output_names: tuple[tuple[str, str], ...]


def _lower_collect(
    program: Program,
    selected: Runtime,
    array_options: ArrayExecutionOptions | None = None,
) -> _LoweredCollect:
    from calc_flow.symbolic.lower.program import lower_program_document

    bindings = _BatchBindings()
    document = lower_program_document(
        program, selected, "batch", _bindings=bindings, _array_options=array_options
    )
    input_names, output_names = bindings.names()
    return _LoweredCollect(
        _canonical(document),
        tuple(input_names.items()),
        tuple(output_names.items()),
    )


def _lowered_collect(program: Program, selected: Runtime) -> _LoweredCollect:
    from calc_flow.symbolic.union import cache_policy

    # Registrations clear the runtime cache, so the key needs only the program.
    key = (
        "collect",
        program.fingerprint,
        tuple(value._node.node_bytes for value in program.inputs),
        *cache_policy(program, selected),
    )
    return cast(
        _LoweredCollect,
        selected._cached_symbolic_compile(
            key, lambda: _lower_collect(program, selected)
        ),
    )


def _prepare_collect_batches(
    program: Program,
    batches: Mapping[str, Batch],
    selected: Runtime,
    _lowered: _LoweredCollect | None = None,
) -> tuple[BatchExecutionPlan, dict[str, Batch], dict[str, str]]:
    with selected._registration_lock:
        lowered = _lowered_collect(program, selected) if _lowered is None else _lowered
        input_names = dict(lowered.input_names)
        output_names = dict(lowered.output_names)
        for name in batches:
            if name not in input_names:
                errors.raise_compile(
                    f"inputs.{name}",
                    errors.INVALID_LITERAL,
                    "unconsumed explicit input; remove it from Program.inputs",
                )
        # Each call still compiles a fresh native plan from the immutable template.
        plan = selected.compile_batch_project(lowered.project_json)
    physical = {
        endpoint: batches[name]
        for name, endpoints in input_names.items()
        for endpoint in endpoints
    }
    return plan, physical, {name: output_names[name] for name, _ in program.outputs}


def _tables(result: RunResult, outputs: dict[str, str]) -> dict[str, pa.Table]:
    # Each RunResult.outputs read rebuilds every Batch wrapper; read it once.
    batches = result.outputs
    return {name: batches[port].to_pyarrow() for name, port in outputs.items()}


def _collect_array(
    expression: ArrayExpr,
    inputs: Mapping[str, TableData | NativeArray],
    runtime: Runtime,
    options: ExecutionOptions | None,
    array_options: ArrayExecutionOptions | None,
) -> NativeArray:
    _require_blocking("collect_array")
    if not is_native_array(expression._node):
        raise TypeError("collect_array requires a native ArrayExpr")
    _validate_options(options)
    if not isinstance(runtime, Runtime):
        raise TypeError("collect_array requires an explicit calc_flow Runtime")
    if array_options is not None and not isinstance(
        array_options, ArrayExecutionOptions
    ):
        raise TypeError("array_options must be ArrayExecutionOptions or None")
    limits = ArrayExecutionOptions() if array_options is None else array_options
    program = Program("collect_array", engine="sql", outputs={"output": expression})
    batches = _collect_batches(program, inputs)
    key = (
        "collect_array",
        program.fingerprint,
        tuple(value._node.node_bytes for value in program.inputs),
        limits.max_elements,
        limits.max_allocation_bytes,
    )
    with runtime._registration_lock:
        lowered = cast(
            _LoweredCollect,
            runtime._cached_symbolic_compile(
                key, lambda: _lower_collect(program, runtime, limits)
            ),
        )
        plan, bound, outputs = _prepare_collect_batches(
            program, batches, runtime, lowered
        )
    return _native_output(
        plan.execute(bound, options=options).outputs[outputs["output"]]
    )


def _native_output(output: Batch) -> NativeArray:
    if output.kind != "array" or output.backend != "native":
        raise TypeError("collect_array.output: expected a native array Batch")
    array = output.array
    if not isinstance(array, NativeArray):
        raise TypeError("collect_array.output: expected an actual native array payload")
    return array


def _collect(
    program: Program,
    inputs: Mapping[str, TableData | NativeArray],
    runtime: Runtime | None,
    options: ExecutionOptions | None,
    portability: PortabilityRequest | None = None,
) -> dict[str, pa.Table]:
    _require_blocking("collect")
    _require_table_outputs(program)
    plan, bound, outputs = _prepare_collect(program, inputs, runtime, portability)
    _validate_options(options)
    return _tables(plan.execute(bound, options=options), outputs)


def _as_input_mapping(
    program: Program, inputs: T | Mapping[str, T], error: str
) -> Mapping[str, T]:
    if isinstance(inputs, Mapping):
        return dict(inputs)
    if len(program.inputs) != 1 or not isinstance(program.inputs[0], TableExpr):
        raise ValueError(error)
    return {_node_name(program.inputs[0]._node): inputs}


def _table_inputs(
    program: Program, inputs: TableData | Mapping[str, TableData | NativeArray]
) -> Mapping[str, TableData | NativeArray]:
    return _as_input_mapping(
        program,
        inputs,
        "collect: multiple inputs require a mapping by declared input name",
    )


def compute(
    data: TableData,
    build: Callable[[TableExpr], TableExpr],
    /,
    *,
    runtime: Runtime | None = None,
    options: ExecutionOptions | None = None,
) -> pa.Table:
    """Build from Arrow fields and independently compute the result.

    The inferred input has no ordering. For temporal expressions, declare
    ordering with ``table_input`` and execute through ``TableExpr.collect``.
    """
    _require_blocking("compute")
    program, batch = _build_program(data, build)
    return _collect(program, {"input": batch}, runtime, options)["output"]


def _build_program(
    data: TableData,
    build: Callable[[TableExpr], TableExpr],
) -> tuple[Program, Batch]:
    batch = _table_batch(data, "compute.data")
    source = table_input("input", schema=batch.to_pyarrow().schema)
    if not callable(build):
        raise TypeError("compute.build: expected a callable returning TableExpr")
    output = build(source)
    if not isinstance(output, TableExpr):
        if inspect.iscoroutine(output):
            output.close()
        raise TypeError(
            f"compute.build: expected TableExpr, got {type(output).__name__}; "
            "use a synchronous expression builder"
        )
    return Program("compute", engine="sql", outputs={"output": output}), batch


def _collect_async(
    program: Program,
    inputs: Mapping[str, TableData | NativeArray],
    runtime: Runtime | None,
    options: ExecutionOptions | None,
    portability: PortabilityRequest | None = None,
) -> Awaitable[dict[str, pa.Table]]:
    _require_table_outputs(program)
    if portability is not None or program.update_semantics == "latest_relation":
        selected = _selected_runtime(runtime)
        _validate_options(options)
        captured = dict(inputs)

        async def strict_execute() -> dict[str, pa.Table]:
            plan, bound, outputs = await _run_blocking_compile(
                _prepare_collect, program, captured, selected, portability
            )
            return await _execute_collect(plan, bound, outputs, options)

        return strict_execute()
    batches = _collect_batches(program, inputs)
    selected = _selected_runtime(runtime)
    _validate_options(options)

    async def execute() -> dict[str, pa.Table]:
        plan, bound, outputs = await _run_blocking_compile(
            _prepare_collect_batches, program, batches, selected
        )
        return await _execute_collect(plan, bound, outputs, options)

    return execute()


def _require_table_outputs(program: Program) -> None:
    if any(isinstance(value, ArrayExpr) for _, value in program.outputs):
        raise TypeError(
            "collect returns tables; use ArrayExpr.collect_array for one array result"
        )


async def _run_blocking_compile(compile_plan: Callable[..., T], *args: object) -> T:
    task = asyncio.create_task(asyncio.to_thread(compile_plan, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        from calc_flow.runtime import _finish_cleanup

        async def drain() -> None:
            await asyncio.gather(task, return_exceptions=True)

        await _finish_cleanup(drain())
        raise


def _validate_options(options: ExecutionOptions | None) -> None:
    if options is not None and type(options) is not ExecutionOptions:
        raise TypeError("options must be a calc_flow.ExecutionOptions or None")


async def _execute_collect(
    plan: BatchExecutionPlan,
    bound: dict[str, Batch],
    outputs: dict[str, str],
    options: ExecutionOptions | None,
) -> dict[str, pa.Table]:
    return _tables(await plan.execute_async(bound, options=options), outputs)


def _collect_table_async(
    program: Program,
    inputs: Mapping[str, TableData | NativeArray],
    runtime: Runtime | None,
    options: ExecutionOptions | None,
) -> Awaitable[pa.Table]:
    batches = _collect_batches(program, inputs)
    selected = _selected_runtime(runtime)
    _validate_options(options)

    async def execute() -> pa.Table:
        plan, bound, outputs = await _run_blocking_compile(
            _prepare_collect_batches, program, batches, selected
        )
        return (await _execute_collect(plan, bound, outputs, options))["output"]

    return execute()


def compute_async(
    data: TableData,
    build: Callable[[TableExpr], TableExpr],
    /,
    *,
    runtime: Runtime | None = None,
    options: ExecutionOptions | None = None,
) -> Awaitable[pa.Table]:
    """Capture the input Batch and await cancellation-aware native execution.

    Arrow buffers are shared. Keep their underlying storage read-only until
    execution completes; this does not copy the table contents.

    The inferred input has no ordering. For temporal expressions, declare
    ordering with ``table_input`` and await ``TableExpr.collect_async``.
    """
    program, batch = _build_program(data, build)
    return _collect_table_async(program, {"input": batch}, runtime, options)
