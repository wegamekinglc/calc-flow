from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from calc_flow_studio import run_manager
from calc_flow_studio.models import RunEvent


def _status():
    side = {
        "retained_rows": 0,
        "retained_bytes": 0,
        "evicted_rows": 0,
        "late_rows": 0,
        "late_affected_batches": 0,
        "max_lateness_micros": None,
        "null_event_time_rows": 0,
        "null_key_rows": 0,
        "watermark_micros": -(2**63),
        "idle": False,
        "ended": True,
    }
    return {
        "left": side,
        "right": {**side, "watermark_micros": 2**63 - 1, "idle": True},
        "emitted_match_rows": 1,
        "state_limit_failures": 0,
        "match_limit_failures": 0,
    }


def _event(metrics):
    return RunEvent.model_validate(
        {
            "sequence": 1,
            "timestamp": "2026-10-06T00:00:00Z",
            "type": "progress",
            "message": "Job progress",
            "stream_joins": metrics,
        }
    )


def test_join_progress_preserves_watermark_precision_and_existing_numeric_fields():
    native = {"match": _status()}
    original = deepcopy(native)
    progress = run_manager._continuous_progress({"stream_joins": native})
    wire = _event(progress["stream_joins"]).model_dump(mode="json")["stream_joins"][0]
    assert wire["left"]["watermark_micros"] == "-9223372036854775808"
    assert wire["right"]["watermark_micros"] == "9223372036854775807"
    assert wire["left"]["idle"] is False
    assert wire["left"]["ended"] is True
    assert wire["emitted_match_rows"] == 1
    assert native == original
    assert run_manager._stream_join_progress([wire]) == (wire,)


@pytest.mark.parametrize("bad", [True, 1.5, "1", -(2**63) - 1, 2**63])
def test_native_join_watermark_requires_an_exact_in_range_integer(bad):
    native = _status()
    native["left"]["watermark_micros"] = bad
    with pytest.raises((TypeError, ValueError)):
        run_manager._continuous_progress({"stream_joins": {"match": native}})


@pytest.mark.parametrize("bad", ["+1", "01", "-0", "1.0", "9223372036854775808", 1])
def test_join_event_rejects_noncanonical_or_out_of_range_watermarks(bad):
    metrics = run_manager._stream_join_progress({"match": _status()})
    metrics[0]["left"]["watermark_micros"] = bad
    with pytest.raises(ValidationError):
        _event(metrics)


@pytest.mark.parametrize("bad", [str(-(2**63) - 1), str(2**63)])
def test_join_watermark_range_error_uses_the_shared_integer_domain(bad):
    metrics = run_manager._stream_join_progress({"match": _status()})
    metrics[0]["left"]["watermark_micros"] = bad
    with pytest.raises(ValidationError) as failure:
        _event(metrics)
    assert failure.value.errors()[0]["msg"] == "Value error, watermark exceeds i64"


def test_join_openapi_describes_precise_progress_fields(tmp_path):
    from calc_flow_studio.app import create_app

    schemas = create_app(project_directory=tmp_path / "projects").openapi()[
        "components"
    ]["schemas"]
    side = schemas["StreamJoinSideMetrics"]["properties"]
    assert side["watermark_micros"]["anyOf"][0]["$ref"].endswith("/SignedDecimal")
    assert side["idle"]["type"] == "boolean"
    assert side["ended"]["type"] == "boolean"


def test_join_sse_preserves_null_watermark_and_exact_integer_strings(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace

    from calc_flow_studio.app import create_app
    from calc_flow_studio.models import RunStatus

    native = _status()
    native["left"]["watermark_micros"] = None
    event = _event(run_manager._stream_join_progress({"match": native}))
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
        response = await endpoint("match", None)
        return [frame async for frame in response.body_iterator]

    frame = "".join(asyncio.run(read()))
    payload = json.loads(
        next(line[6:] for line in frame.splitlines() if line.startswith("data: "))
    )
    assert payload["stream_joins"][0]["left"]["watermark_micros"] is None
    assert (
        payload["stream_joins"][0]["right"]["watermark_micros"] == "9223372036854775807"
    )
