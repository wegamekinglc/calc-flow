"""Exact interval-Join fixtures and a full two-identity output oracle."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from scripts.toolkit import fingerprint_json, sha256_file, write_json


def validate_interval_output(
    actual: pa.Table, expected: pa.Table, input_rows: int
) -> dict:
    """Check every pair, multiplicity, value, type and schema outside timing."""
    columns = ["sequence", "right_sequence", "value"]
    if (
        actual.column_names != columns
        or not actual.schema.equals(expected.schema, check_metadata=True)
        or actual.num_rows != expected.num_rows
        or any(actual[name].null_count for name in columns)
    ):
        raise ValueError("interval output schema, nulls or row count differ")
    order = [(name, "ascending") for name in columns[:2]]
    if not actual.sort_by(order).equals(expected.sort_by(order), check_metadata=True):
        raise ValueError("interval output pair identity, multiplicity or value differs")
    return {
        "passed": True,
        "rows": expected.num_rows,
        "input_rows": input_rows,
        "max_abs_error": 0.0,
    }


def export_interval_fixture(
    root: Path, program, streams: dict, expected: pa.Table, case: dict
) -> dict:
    """Seal both input streams and their real interleaving without ASOF barriers."""
    from benchmarks import engine_stream as stream
    from benchmarks.engine_lifecycle import interleaved_events
    from benchmarks.join_asof_acceptance import _runtime_event_fields, _write_ipc

    if root.exists() and any(root.iterdir()):
        raise ValueError("interval fixture requires an empty destination")
    if set(streams) != {"reference.input", "quotes.input"}:
        raise ValueError("interval fixture requires both maintained source bindings")
    inputs = {name: [] for name in streams}
    events = []
    for name, event in interleaved_events(
        {name: values[:-1] for name, values in streams.items()}
    ):
        fields, batches = _runtime_event_fields(
            event, len(inputs[name]), "unexpected interval event"
        )
        if batches is not None:
            inputs[name].extend(batches)
        events.append({"input": name, **fields})
    if any(
        sum(batch.num_rows for batch in batches) != case["rows"]
        for batches in inputs.values()
    ):
        raise ValueError(
            "interval fixture does not contain the declared rows on both sides"
        )
    if any(values[-1] is not None for values in streams.values()):
        raise ValueError("interval fixture must end each source")
    root.mkdir(parents=True, exist_ok=True)
    project = root / "project.json"
    write_json(project, program.to_project().model_dump(mode="json"))
    write_json(root / "events.json", events)
    manifest = {
        "schema": "calc-flow.interval-acceptance-fixture.v1",
        "scenario": "interval_join",
        "rows_per_side": case["rows"],
        "batch_rows": case["batch_rows"],
        "checkpoint_enabled": case["mode"] != "off",
        "project": {"path": project.name, "sha256": sha256_file(project)},
        "compiled_plan_fingerprint": program.compile_stream(
            stream.Runtime()
        ).fingerprint,
        "inputs": {
            name: _write_ipc(root / f"input-{index}.arrow", batches)
            for index, (name, batches) in enumerate(sorted(inputs.items()))
        },
        "expected_output": _write_ipc(root / "expected.arrow", expected.to_batches()),
        "events_sha256": sha256_file(root / "events.json"),
        "submission_policy": "interleaved-source-events-without-output-count-barriers",
    }
    write_json(root / "fixture.json", manifest)
    return {**manifest, "manifest_sha256": fingerprint_json(manifest)}


def prepare_interval_fixture(case: dict, root: Path) -> dict:
    """Use the suite's existing workload, graph and oracle at the declared size."""
    from benchmarks import engine_stream as stream
    from benchmarks.engine_comparison import expected_output, workload

    data = workload(case["rows"], scenario="interval_join")
    events = stream.stream_events(
        data.table, data.entities, batch_rows=case["batch_rows"]
    )
    program = stream.stream_program(
        "interval_join", data.table, data.dimension, batch_rows=case["batch_rows"]
    )
    manifest = export_interval_fixture(
        root,
        program,
        {"reference.input": events, "quotes.input": events},
        expected_output(data, "interval_join"),
        case,
    )
    path = root / "fixture.json"
    return {
        "fixture_path": str(path),
        "fixture_sha256": sha256_file(path),
        "manifest": manifest,
    }
