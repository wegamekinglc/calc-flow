from __future__ import annotations

from datetime import datetime, timezone
from itertools import count
from runpy import run_path

import pyarrow as pa
import pytest


def test_public_session_reports_64_clocks_after_one_history_push(tmp_path):
    from benchmarks.steady_session import NATIVE_STUDY, observe

    run_path(str(NATIVE_STUDY / "fixture.py"))["generate"](tmp_path)
    ticks = count(0, 100)
    observation = observe(tmp_path, clock=ticks.__next__)
    assert observation["append_ns"] == [100] * 64
    assert next(ticks) == 12_800
    assert observation["validated_rows"] == 4160
    assert observation["finish_rows"] == 0


def test_prepared_watermark_preserves_utc_instants_across_the_epoch():
    from benchmarks.steady_session import _prepared

    instants = (
        datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc),
    )
    table = pa.table({"event_time": pa.array(instants, pa.timestamp("us", tz="UTC"))})
    for i, instant in enumerate(instants):
        _, mark = _prepared(table.slice(0, i + 1))
        assert mark.at == instant
        assert mark.at.tzinfo is timezone.utc


def test_session_oracle_rejects_a_changed_null_key_value_or_metadata():
    from benchmarks.steady_session import verify_output

    reference = pa.table(
        {"a": [1], "b": [2], "c": [3], "d": [4], "spread": pa.array([0.0])}
    )
    wrong_null = reference.set_column(4, "spread", pa.array([None], type=pa.float64()))
    wrong_key = reference.set_column(2, "c", pa.array([4]))
    wrong_spread = reference.set_column(4, "spread", pa.array([1.0]))
    wrong_metadata = reference.replace_schema_metadata({b"changed": b"yes"})
    for wrong in (wrong_null, wrong_key, wrong_spread, wrong_metadata):
        with pytest.raises(ValueError):
            verify_output(wrong, reference)
