"""Prepared fixtures and observable boundaries for the fixed Join/ASOF task.

Importing this module does not import calc_flow or load its native library.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

from scripts.toolkit import fingerprint_json, sha256_file, write_json

FINAL_FRONTIER_POLL_SECONDS = 0.001


async def await_asof_frontier(
    job, micros: int, *, output_node_id: str | None = None
) -> dict:
    """Observe ASOF publication and the optional completed output projection cut."""

    while True:
        status = job.status()
        if status["state"] not in ("running", "draining"):
            raise RuntimeError(f"ASOF terminated before final frontier: {status}")
        joins = tuple(status["stream_asof_joins"].values())
        if len(joins) != 1:
            raise ValueError("ASOF acceptance requires exactly one operator status")
        frontier = joins[0].get("output_watermark_micros")
        if frontier is not None and frontier >= micros:
            completed = (
                micros
                if output_node_id is None
                else job._inner._benchmark_dispatch_frontier(output_node_id)
            )
            if completed is not None and completed >= micros:
                return status
        await asyncio.sleep(FINAL_FRONTIER_POLL_SECONDS)


def final_projection_node(project: dict) -> str:
    """Identify the maintained graph's one terminal output expression."""
    graph = project["graph"]
    upstream = {edge["source_node"] for edge in graph["edges"]}
    terminal = [node for node in graph["nodes"] if node["id"] not in upstream]
    if len(terminal) != 1 or terminal[0]["operator"]["kind"] != "expression":
        raise ValueError("matched ASOF requires one terminal output projection")
    return terminal[0]["id"]


def validate_retained_status(status: dict, rows: int, dimension_rows: int) -> None:
    """Require the entire quote stream to remain resident at the measured cut."""

    left, right = status["left"], status["right"]
    if (
        left["retained_rows"] != rows
        or right["retained_rows"] != dimension_rows
        or left["retained_bytes"] <= 0
        or right["retained_bytes"] <= 0
        or status["emitted_match_rows"] != rows
        or right["watermark_micros"] is not None
        or any(
            side["evicted_rows"] or side["idle"] or side["ended"]
            for side in (left, right)
        )
    ):
        raise ValueError("retained Join did not preserve every quote and dimension row")


def validate_one_to_one_output(actual: pa.Table, expected: pa.Table) -> dict:
    """Check every deterministic output once with a bounded identity bitmap."""

    count = expected.num_rows
    if set(actual.column_names) != {"sequence", "value"} or actual.num_rows != count:
        raise ValueError("acceptance output columns or row count differ")
    if actual.column_names != expected.column_names or any(
        actual[name].type != expected[name].type for name in expected.column_names
    ):
        raise ValueError("acceptance output Arrow types or column order differ")
    expected_ids = expected["sequence"].to_numpy()
    if not np.array_equal(expected_ids, np.arange(count, dtype=np.uint64)):
        raise ValueError("acceptance oracle must enumerate every sequence in order")
    reference = expected["value"].to_numpy()
    seen = np.zeros(count, dtype=np.bool_)
    for batch in actual.select(("sequence", "value")).to_batches(max_chunksize=64_000):
        sequence, values = batch.column(0), batch.column(1)
        if (
            sequence.null_count
            or values.null_count
            or not pa.types.is_integer(sequence.type)
        ):
            raise ValueError("acceptance output has nulls or invalid sequence type")
        indices = sequence.to_numpy(zero_copy_only=False)
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("acceptance output sequence is out of bounds")
        if len(np.unique(indices)) != len(indices) or np.any(seen[indices]):
            raise ValueError("acceptance output contains duplicate sequence")
        if not np.array_equal(
            values.to_numpy(zero_copy_only=False), reference[indices]
        ):
            raise ValueError("acceptance output value differs from the full oracle")
        seen[indices] = True
    if not bool(np.all(seen)):
        raise ValueError("acceptance output is incomplete")
    return {"passed": True, "rows": count, "max_abs_error": 0.0, "bitmap_bytes": count}


def _validate_fixture_events(inputs: dict, events: dict, batch_rows: int) -> None:
    if set(events) != {"preload", "processing", "terminal", "barriers"}:
        raise ValueError("fixture requires exact phase and barrier sections")
    if events["preload"]:
        raise ValueError("matched ASOF fixture must start with empty state")
    if any(not isinstance(events[phase], list) for phase in events):
        raise ValueError("fixture event sections must be lists")
    visited = Counter()
    ids = set()
    marks = {}
    ended = set()
    for phase in ("processing", "terminal"):
        for index, event in enumerate(events[phase]):
            name, kind = event.get("input"), event.get("kind")
            if name not in inputs:
                raise ValueError("fixture event has an unknown input")
            event_id = event.get("event_id")
            if event_id != f"{phase}/{index}" or event_id in ids:
                raise ValueError("fixture event IDs must identify their exact position")
            ids.add(event_id)
            if phase == "terminal":
                if kind != "end" or name in ended:
                    raise ValueError("terminal events must end each input exactly once")
                ended.add(name)
                continue
            if kind == "end":
                raise ValueError("EOF is forbidden in processing events")
            if type(event.get("chunk_id")) is not int or event["chunk_id"] < 0:
                raise ValueError("processing events require a chunk ID")
            if kind == "data":
                indices = event.get("batches")
                if (
                    not isinstance(indices, list)
                    or not indices
                    or any(
                        type(item) is not int or not 0 <= item < len(inputs[name])
                        for item in indices
                    )
                    or sum(inputs[name][item].num_rows for item in indices) > batch_rows
                ):
                    raise ValueError(
                        "fixture data event has invalid record-batch group"
                    )
                visited.update((name, item) for item in indices)
            elif kind == "watermark":
                value = event.get("micros")
                if type(value) is not int or (name in marks and value < marks[name]):
                    raise ValueError(
                        "fixture watermarks must be monotonic integer microseconds"
                    )
                marks[name] = value
            else:
                raise ValueError("matched processing only admits data and watermarks")
    expected = Counter(
        (name, index)
        for name, batches in inputs.items()
        for index in range(len(batches))
    )
    if visited != expected or ended != set(inputs):
        raise ValueError("fixture must use every input batch and end each source once")
    barriers = events["barriers"]
    processing = events["processing"]
    if len(processing) != 4 * len(barriers):
        raise ValueError("every matched ASOF chunk requires one complete barrier")
    rows = 0
    for chunk, barrier in enumerate(barriers):
        group = processing[4 * chunk : 4 * chunk + 4]
        order = [(event["input"], event["kind"]) for event in group]
        if order != [
            ("reference.input", "data"),
            ("quotes.input", "data"),
            ("reference.input", "watermark"),
            ("quotes.input", "watermark"),
        ]:
            raise ValueError(
                "fixture chunks must match the maintained submission order"
            )
        if any(event["chunk_id"] != chunk for event in group):
            raise ValueError("fixture chunk IDs must be consecutive")
        rows += sum(
            inputs["quotes.input"][index].num_rows for index in group[1]["batches"]
        )
        if barrier != {
            "chunk_id": chunk,
            "after_events": [event["event_id"] for event in group],
            "output_rows": rows,
        }:
            raise ValueError(
                "fixture barrier must follow its complete chunk at the exact row count"
            )


def _validate_join_fixture(
    inputs: dict, events: dict, rows: int, batch_rows: int
) -> None:
    if set(inputs) != {"left", "right"} or len(events.get("preload", [])) != 2:
        raise ValueError(
            "sealed Join requires left quotes and a preloaded right dimension"
        )
    if set(events) != {"preload", "processing", "terminal", "barriers"}:
        raise ValueError(
            "sealed Join fixture requires exact phase and barrier sections"
        )
    if sum(batch.num_rows for batch in inputs["left"]) != rows:
        raise ValueError("sealed Join quote rows differ from the declared workload")
    if not 1 <= sum(batch.num_rows for batch in inputs["right"]) <= 64:
        raise ValueError("sealed Join dimension must use at most 64 fixture entities")
    visited = Counter()
    for phase, binding in (("preload", "right"), ("processing", "left")):
        records = events[phase]
        if not records or len(records) % 2:
            raise ValueError("sealed Join messages must alternate data and watermark")
        last_watermark = None
        for index, record in enumerate(records):
            expected_kind = "data" if index % 2 == 0 else "watermark"
            if (
                record.get("input") != binding
                or record.get("kind") != expected_kind
                or record.get("event_id") != f"{phase}/{index}"
                or record.get("chunk_id") != index // 2
            ):
                raise ValueError("sealed Join fixture has inconsistent event identity")
            if expected_kind == "data":
                batches = record.get("batches")
                if (
                    not isinstance(batches, list)
                    or not batches
                    or any(
                        type(item) is not int or not 0 <= item < len(inputs[binding])
                        for item in batches
                    )
                    or sum(inputs[binding][item].num_rows for item in batches)
                    > batch_rows
                ):
                    raise ValueError("sealed Join fixture has an invalid batch group")
                visited.update((binding, item) for item in batches)
            else:
                micros = record.get("micros")
                if type(micros) is not int or (
                    last_watermark is not None and micros < last_watermark
                ):
                    raise ValueError("sealed Join requires monotonic watermarks")
                last_watermark = micros
    expected = Counter(
        (name, index)
        for name, batches in inputs.items()
        for index in range(len(batches))
    )
    if visited != expected:
        raise ValueError("sealed Join must use every record batch exactly once")
    if events["terminal"] != [
        {"kind": "end", "input": name, "event_id": f"terminal/{index}"}
        for index, name in enumerate(("right", "left"))
    ]:
        raise ValueError("sealed Join must end each input exactly once")
    barriers = events["barriers"]
    if len(events["processing"]) != 2 * len(barriers):
        raise ValueError("sealed Join native chunks require complete barriers")
    total = 0
    for chunk, barrier in enumerate(barriers):
        group = events["processing"][2 * chunk : 2 * chunk + 2]
        total += sum(inputs["left"][index].num_rows for index in group[0]["batches"])
        if barrier != {
            "chunk_id": chunk,
            "after_events": [event["event_id"] for event in group],
            "output_rows": total,
        }:
            raise ValueError("sealed Join native barrier row count differs")


def _write_ipc(path: Path, batches: list[pa.RecordBatch]) -> dict:
    if not batches:
        raise ValueError("fixture IPC file requires at least one record batch")
    with (
        pa.OSFile(str(path), "wb") as sink,
        pa.ipc.new_file(sink, batches[0].schema) as writer,
    ):
        for batch in batches:
            writer.write_batch(batch)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "rows": sum(batch.num_rows for batch in batches),
    }


def _partial_order(events: dict) -> dict:
    phases = {}
    for phase in ("preload", "processing", "terminal"):
        by_input = {}
        for event in events[phase]:
            by_input.setdefault(event["input"], []).append(event)
        phases[phase] = by_input
    return {**phases, "barriers": events.get("barriers", [])}


def export_matched_fixture(
    root: Path,
    *,
    project: dict,
    inputs: dict[str, list[pa.RecordBatch]],
    events: dict,
    expected: pa.Table,
    output_binding: str,
    plan_fingerprint: str,
    rows: int,
    batch_rows: int,
    checkpoint_enabled: bool,
    scenario: str = "asof_join",
) -> dict:
    """Seal exact Arrow message groups, project, oracle, and ordering constraints."""

    if (
        type(rows) is not int
        or rows <= 0
        or type(batch_rows) is not int
        or batch_rows <= 0
        or expected.num_rows != rows
    ):
        raise ValueError("fixture row counts differ from the declared workload")
    if scenario == "asof_join":
        if set(inputs) != {"reference.input", "quotes.input"} or any(
            sum(batch.num_rows for batch in batches) != rows
            for batches in inputs.values()
        ):
            raise ValueError("ASOF fixture row counts or bindings differ")
        _validate_fixture_events(inputs, events, batch_rows)
    elif scenario == "join_sealed":
        _validate_join_fixture(inputs, events, rows, batch_rows)
    else:
        raise ValueError("unsupported matched fixture scenario")
    if root.exists() and any(root.iterdir()):
        raise ValueError("matched fixture export requires an empty destination")
    root.mkdir(parents=True, exist_ok=True)
    project_path, event_path = root / "project.json", root / "events.json"
    write_json(project_path, project)
    write_json(event_path, events)
    input_files = {
        name: _write_ipc(root / f"input-{index}.arrow", list(batches))
        for index, (name, batches) in enumerate(sorted(inputs.items()))
    }
    output_file = _write_ipc(root / "expected.arrow", expected.to_batches())
    identity = {
        name: {key: value for key, value in record.items() if key != "path"}
        for name, record in input_files.items()
    }
    manifest = {
        "schema": "calc-flow.matched-native-fixture.v1",
        "scenario": scenario,
        **(
            {"rows_per_side": rows}
            if scenario == "asof_join"
            else {
                "rows": rows,
                "dimension_rows": sum(batch.num_rows for batch in inputs["right"]),
            }
        ),
        "batch_rows": batch_rows,
        "checkpoint_enabled": checkpoint_enabled,
        "project": {"path": project_path.name, "sha256": sha256_file(project_path)},
        "inputs": input_files,
        "inputs_sha256": fingerprint_json(identity),
        "events": {"path": event_path.name, "sha256": sha256_file(event_path)},
        "shared_partial_order_sha256": fingerprint_json(_partial_order(events)),
        "canonical_native_events_sha256": fingerprint_json(events),
        "expected_output": output_file,
        "output_binding": output_binding,
        "compiled_plan_fingerprint": plan_fingerprint,
    }
    write_json(root / "fixture.json", manifest)
    return manifest


def export_runtime_fixture(
    root: Path, program, streams: dict, expected: pa.Table, case: dict
) -> dict:
    """Export the exact maintained ASOF message geometry before warmup."""
    from datetime import timedelta

    from benchmarks import engine_stream as stream
    from benchmarks.engine_lifecycle import interleaved_events

    inputs = {name: [] for name in streams}
    events = {"preload": [], "processing": [], "terminal": [], "barriers": []}
    timed = {name: values[:-1] for name, values in streams.items()}
    rows = 0
    for chunk, start in enumerate(range(0, len(timed["quotes.input"]), 2)):
        group = {name: values[start : start + 2] for name, values in timed.items()}
        ids = []
        for name, event in interleaved_events(group):
            event_id = f"processing/{len(events['processing'])}"
            record = {"input": name, "event_id": event_id, "chunk_id": chunk}
            if isinstance(event, stream.Data):
                batches = event.batch.to_pyarrow().to_batches()
                first = len(inputs[name])
                inputs[name].extend(batches)
                record.update(
                    kind="data", batches=list(range(first, first + len(batches)))
                )
            elif isinstance(event, stream.Watermark):
                record.update(
                    kind="watermark",
                    micros=int(
                        stream.BASE_MICROS
                        + (event.at - stream.BASE) // timedelta(microseconds=1)
                    ),
                )
            else:
                raise ValueError("unexpected ASOF fixture event")
            events["processing"].append(record)
            ids.append(event_id)
        rows += group["quotes.input"][0].batch.num_rows
        events["barriers"].append(
            {"chunk_id": chunk, "after_events": ids, "output_rows": rows}
        )
    events["terminal"] = [
        {"kind": "end", "input": name, "event_id": f"terminal/{index}"}
        for index, name in enumerate(streams)
    ]
    return export_matched_fixture(
        root,
        project=program.to_project().model_dump(mode="json"),
        inputs=inputs,
        events=events,
        expected=expected,
        output_binding="output",
        plan_fingerprint=program.compile_stream(stream.Runtime()).fingerprint,
        rows=case["rows"],
        batch_rows=case["batch_rows"],
        checkpoint_enabled=case["mode"] != "off",
    )


def export_join_runtime_fixture(
    root: Path, program, streams: dict, expected: pa.Table, case: dict
) -> dict:
    """Seal native-only callback chunks with row 1's physical Join graph."""
    from datetime import timedelta

    from benchmarks import engine_stream as stream

    inputs = {"right": [], "left": []}
    events = {"preload": [], "processing": [], "terminal": [], "barriers": []}
    rows = 0
    for phase, name in (("preload", "right"), ("processing", "left")):
        for index, event in enumerate(streams[name][:-1]):
            record = {
                "input": name,
                "event_id": f"{phase}/{index}",
                "chunk_id": index // 2,
            }
            if isinstance(event, stream.Data):
                batches = event.batch.to_pyarrow().to_batches()
                first = len(inputs[name])
                inputs[name].extend(batches)
                record.update(
                    kind="data", batches=list(range(first, first + len(batches)))
                )
                if phase == "processing":
                    rows += event.batch.num_rows
            elif isinstance(event, stream.Watermark):
                record.update(
                    kind="watermark",
                    micros=int(
                        stream.BASE_MICROS
                        + (event.at - stream.BASE) // timedelta(microseconds=1)
                    ),
                )
            else:
                raise ValueError("unexpected sealed Join fixture event")
            events[phase].append(record)
            if phase == "processing" and index % 2:
                events["barriers"].append(
                    {
                        "chunk_id": index // 2,
                        "after_events": [
                            f"processing/{index - 1}",
                            f"processing/{index}",
                        ],
                        "output_rows": rows,
                    }
                )
    events["terminal"] = [
        {"kind": "end", "input": name, "event_id": f"terminal/{index}"}
        for index, name in enumerate(("right", "left"))
    ]
    return export_matched_fixture(
        root,
        project=program.to_project().model_dump(mode="json"),
        inputs=inputs,
        events=events,
        expected=expected,
        output_binding="output",
        plan_fingerprint=program.compile_stream(stream.Runtime()).fingerprint,
        rows=case["rows"],
        batch_rows=case["batch_rows"],
        checkpoint_enabled=case["mode"] != "off",
        scenario="join_sealed",
    )


def prepare_acceptance_fixture(case: dict, root: Path) -> dict:
    """Prepare sealed inputs and compiler identity without a job or warmup."""
    destination = Path(case.get("fixture_root", root / "matched-fixture"))
    if case.get("ordering_proof"):
        if (
            case["workload"] not in ("asof", "asof_operator")
            or case["rows"] != 2049
            or case["batch_rows"] != 1024
            or case["mode"] not in ("off", "low")
        ):
            raise ValueError("ordering proof requires its fixed tiny ASOF fixture")
        manifest = export_ordering_fixture(
            destination, checkpoint_enabled=case["mode"] != "off"
        )
        path = destination / "fixture.json"
        return {
            "fixture_path": str(path),
            "fixture_sha256": sha256_file(path),
            "manifest": manifest,
        }

    from benchmarks import engine_stream as stream
    from benchmarks.engine_comparison import expected_output, workload

    scenario = (
        "join"
        if case["workload"] in ("join_sealed", "join_small_batch")
        else "asof_join"
    )
    if case["workload"] not in (
        "join_sealed",
        "join_small_batch",
        "asof",
        "asof_operator",
    ):
        raise ValueError("unsupported acceptance fixture workload")
    data = workload(case["rows"], scenario=scenario)
    expected = expected_output(data, scenario)
    quotes = stream.stream_events(
        data.table, data.entities, batch_rows=case["batch_rows"]
    )
    streams = (
        {
            "right": stream.dimension_events(
                stream.stream_dimension(data.dimension), data.table
            ),
            "left": quotes,
        }
        if scenario == "join"
        else {"reference.input": quotes, "quotes.input": quotes}
    )
    program = stream.stream_program(
        scenario, data.table, data.dimension, batch_rows=case["batch_rows"]
    )
    exporter = (
        export_join_runtime_fixture if scenario == "join" else export_runtime_fixture
    )
    manifest = exporter(destination, program, streams, expected, case)
    path = destination / "fixture.json"
    return {
        "fixture_path": str(path),
        "fixture_sha256": sha256_file(path),
        "manifest": manifest,
    }


def export_ordering_fixture(root: Path, *, checkpoint_enabled: bool = True) -> dict:
    """Export three tiny recurring-key chunks solely for callback-order proof."""
    from dataclasses import replace

    from benchmarks import engine_stream as stream
    from benchmarks.engine_comparison import expected_output, workload

    data = workload(2560, scenario="asof_join")
    data = replace(data, table=data.table.slice(0, 2049))
    events = stream.stream_events(data.table, data.entities, batch_rows=1024)
    program = stream.stream_program(
        "asof_join", data.table, data.dimension, batch_rows=1024
    )
    return export_runtime_fixture(
        root,
        program,
        {"reference.input": events, "quotes.input": events},
        expected_output(data, "asof_join"),
        {
            "rows": 2049,
            "batch_rows": 1024,
            "mode": "low" if checkpoint_enabled else "off",
        },
    )


# The worker imports this class without loading calc_flow until construction.
from benchmarks.join_asof_acceptance_runtime import (  # noqa: E402
    AcceptanceEngineCase as AcceptanceEngineCase,
)
