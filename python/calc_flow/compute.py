"""Arrow convenience execution through expression lowering and the native runtime."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence

import pyarrow as pa

from calc_flow._native import Batch, ExecutionOptions, RunResult
from calc_flow.pipeline import BatchExecutionPlan, Runtime, _canonical
from calc_flow.symbolic import errors
from calc_flow.symbolic.expr import Parameter, TableExpr, table_input
from calc_flow.symbolic.lower.bindings import _BatchBindings
from calc_flow.symbolic.program import Program, _node_name, _selected_runtime

type TableData = pa.Table | pa.RecordBatch | Batch


def _require_blocking(entry: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{entry}() cannot run inside an event loop; use {entry}_async()"
    )


def _table_batch(data: TableData, path: str) -> Batch:
    if isinstance(data, Batch):
        if data.kind != "table":
            raise TypeError(f"{path}: expected table input, got {data.kind} Batch")
        return data
    if isinstance(data, (pa.Table, pa.RecordBatch)):
        return Batch.from_pyarrow(data)
    raise TypeError(
        f"{path}: expected Arrow Table, RecordBatch or table Batch; "
        f"got {type(data).__name__}"
    )


def _prepare_collect(
    program: Program, inputs: Mapping[str, TableData], runtime: Runtime | None
) -> tuple[BatchExecutionPlan, dict[str, Batch], dict[str, str]]:
    from calc_flow.symbolic.lower.program import lower_program_document

    if not isinstance(inputs, Mapping):
        raise TypeError("collect.inputs: expected a mapping by declared input name")
    copied = dict(inputs)
    expected = {_node_name(value._node): value for value in program.inputs}
    for name, value in expected.items():
        if name not in copied:
            kind = "static parameter" if isinstance(value, Parameter) else "table input"
            raise ValueError(f"inputs.{name}: missing {kind}")
    for name in copied:
        if name not in expected:
            raise ValueError(f"inputs.{name}: unexpected input name")
    batches: dict[str, Batch] = {}
    for name, value in expected.items():
        data = copied[name]
        if isinstance(value, Parameter) and value.kind == "array":
            if not isinstance(data, Batch) or data.kind != "array":
                raise TypeError(
                    f"inputs.{name}: expected array Batch; use Batch.from_array "
                    "and register its provider"
                )
            batches[name] = data
        else:
            batches[name] = _table_batch(data, f"inputs.{name}")
    selected = _selected_runtime(runtime)
    bindings = _BatchBindings()
    document = lower_program_document(program, selected, "batch", _bindings=bindings)
    input_names, output_names = bindings.names()
    for name in expected:
        if name not in input_names:
            errors.raise_compile(
                f"inputs.{name}",
                errors.INVALID_LITERAL,
                "unconsumed explicit input; remove it from Program.inputs",
            )
    plan = selected.compile_batch_project(_canonical(document))
    physical = {
        endpoint: batches[name]
        for name, endpoints in input_names.items()
        for endpoint in endpoints
    }
    return plan, physical, {name: output_names[name] for name, _ in program.outputs}


def _tables(result: RunResult, outputs: dict[str, str]) -> dict[str, pa.Table]:
    return {name: result.outputs[port].to_pyarrow() for name, port in outputs.items()}


def _collect(
    program: Program,
    inputs: Mapping[str, TableData],
    runtime: Runtime | None,
    options: ExecutionOptions | None,
) -> dict[str, pa.Table]:
    _require_blocking("collect")
    plan, bound, outputs = _prepare_collect(program, inputs, runtime)
    return _tables(plan.execute(bound, options=options), outputs)


def _table_inputs(
    program: Program, inputs: TableData | Mapping[str, TableData]
) -> Mapping[str, TableData]:
    if isinstance(inputs, Mapping):
        return dict(inputs)
    if len(program.inputs) != 1 or not isinstance(program.inputs[0], TableExpr):
        raise ValueError(
            "collect: multiple inputs require a mapping by declared input name"
        )
    return {_node_name(program.inputs[0]._node): inputs}


def compute(
    data: TableData,
    build: Callable[[TableExpr], TableExpr],
    /,
    *,
    entity_by: Sequence[str] = (),
    event_time: str | None = None,
    sequence_by: Sequence[str] = (),
    runtime: Runtime | None = None,
    options: ExecutionOptions | None = None,
) -> pa.Table:
    """Build from the exact Arrow schema and independently compute the result."""
    _require_blocking("compute")
    program, batch = _build_program(data, build, entity_by, event_time, sequence_by)
    return _collect(program, {"input": batch}, runtime, options)["output"]


def _build_program(
    data: TableData,
    build: Callable[[TableExpr], TableExpr],
    entity_by: Sequence[str],
    event_time: str | None,
    sequence_by: Sequence[str],
) -> tuple[Program, Batch]:
    batch = _table_batch(data, "compute.data")
    source = table_input(
        "input",
        schema=batch.to_pyarrow().schema,
        entity_by=entity_by,
        event_time=event_time,
        sequence_by=sequence_by,
    )
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
    return Program("compute", outputs={"output": output}), batch


def _collect_async(
    program: Program,
    inputs: Mapping[str, TableData],
    runtime: Runtime | None,
    options: ExecutionOptions | None,
) -> Awaitable[dict[str, pa.Table]]:
    plan, bound, outputs = _prepare_collect(program, inputs, runtime)
    _validate_options(options)
    return _execute_collect(plan, bound, outputs, options)


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
    inputs: Mapping[str, TableData],
    runtime: Runtime | None,
    options: ExecutionOptions | None,
) -> Awaitable[pa.Table]:
    plan, bound, outputs = _prepare_collect(program, inputs, runtime)
    _validate_options(options)

    async def execute() -> pa.Table:
        return (await _execute_collect(plan, bound, outputs, options))["output"]

    return execute()


def compute_async(
    data: TableData,
    build: Callable[[TableExpr], TableExpr],
    /,
    *,
    entity_by: Sequence[str] = (),
    event_time: str | None = None,
    sequence_by: Sequence[str] = (),
    runtime: Runtime | None = None,
    options: ExecutionOptions | None = None,
) -> Awaitable[pa.Table]:
    """Snapshot inputs now and await cancellation-aware native execution."""
    program, batch = _build_program(data, build, entity_by, event_time, sequence_by)
    return _collect_table_async(program, {"input": batch}, runtime, options)
