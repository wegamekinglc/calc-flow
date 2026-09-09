from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from calc_flow_studio import run_manager
from calc_flow_studio.models import RunEvent, RunStatus


def _status() -> dict[str, object]:
    side = {
        "accepted_rows": 2**64 - 1,
        "late_rows": 0,
        "duplicate_rows": 0,
        "watermark_micros": -(2**63),
        "idle": False,
        "ended": True,
    }
    return {
        "left": side,
        "right": {**side, "watermark_micros": 2**63 - 1},
        "pending_left_rows": 0,
        "retained_right_rows": 0,
        "identity_only_rows": 0,
        "state_rows": 0,
        "state_bytes": 0,
        "emitted_left_rows": 2**64 - 1,
        "matched_rows": 2**64 - 1,
        "unmatched_rows": 0,
        "evicted_right_rows": 0,
        "state_limit_failures": 0,
        "workspace_limit_failures": 0,
        "output_limit_failures": 0,
        "output_watermark_micros": None,
    }


def _event(metrics: object) -> RunEvent:
    return RunEvent.model_validate(
        {
            "sequence": 1,
            "timestamp": "2026-09-09T00:00:00Z",
            "type": "progress",
            "message": "Job progress",
            "stream_asof_joins": metrics,
        }
    )


def test_asof_native_progress_preserves_full_integer_precision_without_mutation():
    native = {"quotes": _status()}
    original = deepcopy(native)
    progress = run_manager._continuous_progress({"stream_asof_joins": native})
    metrics = progress["stream_asof_joins"]
    wire = _event(metrics).model_dump(mode="json")["stream_asof_joins"][0]
    assert wire["left"]["accepted_rows"] == "18446744073709551615"
    assert wire["left"]["watermark_micros"] == "-9223372036854775808"
    assert wire["right"]["watermark_micros"] == "9223372036854775807"
    assert wire["output_watermark_micros"] is None
    assert wire["left"]["idle"] is False
    assert wire["left"]["ended"] is True
    assert native == original


@pytest.mark.parametrize("bad", [True, 1.5, "1", -1, 2**64])
def test_native_asof_counters_require_exact_in_range_integers(bad):
    native = _status()
    native["state_rows"] = bad
    with pytest.raises((TypeError, ValueError)):
        run_manager._continuous_progress({"stream_asof_joins": {"quotes": native}})


@pytest.mark.parametrize("bad", ["+1", "01", "-0", "1.0", "18446744073709551616", 1])
def test_asof_event_rejects_noncanonical_or_out_of_range_counter_strings(bad):
    progress = run_manager._continuous_progress(
        {"stream_asof_joins": {"quotes": _status()}}
    )
    metrics = deepcopy(progress["stream_asof_joins"])
    metrics[0]["state_rows"] = bad
    with pytest.raises(ValidationError):
        _event(metrics)


def test_openapi_describes_asof_integer_strings_in_event_stream(tmp_path):
    from calc_flow_studio.app import create_app

    app = create_app(
        project_directory=tmp_path / "projects",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    document = app.openapi()
    schemas = document["components"]["schemas"]
    metrics = schemas["StreamAsofJoinMetrics"]["properties"]
    assert metrics["state_rows"]["$ref"].endswith("/UnsignedDecimal")
    assert schemas["UnsignedDecimal"]["type"] == "string"
    response = document["paths"]["/api/v3/jobs/{job_id}/events"]["get"]["responses"][
        "200"
    ]
    assert response["content"]["text/event-stream"]["schema"]["$ref"].endswith(
        "/RunEvent"
    )


def test_sse_preserves_explicit_null_asof_watermarks(tmp_path):
    from calc_flow_studio.app import create_app

    progress = run_manager._continuous_progress(
        {"stream_asof_joins": {"quotes": _status()}}
    )
    event = _event(progress["stream_asof_joins"])
    manager = SimpleNamespace(
        get_job=lambda _job_id: None,
        wait_for_events=lambda *_args, **_kwargs: ((event,), RunStatus.COMPLETED),
    )
    app = create_app(project_directory=tmp_path / "projects", run_manager=manager)
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", "").endswith("/events")
    )

    async def read():
        response = await endpoint("asof", None)
        return [frame async for frame in response.body_iterator]

    frame = "".join(asyncio.run(read()))
    payload = json.loads(
        next(line[6:] for line in frame.splitlines() if line.startswith("data: "))
    )
    assert payload["stream_asof_joins"][0]["output_watermark_micros"] is None
