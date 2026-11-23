from __future__ import annotations

from runpy import run_path

import pytest

from benchmarks import steady_session as steady


@pytest.mark.parametrize("mode", ["plain", "probed"])
def test_one_real_public_push_has_contiguous_phases_only_when_monitored(tmp_path, mode):
    from benchmarks.steady_phases import observe

    run_path(str(steady.NATIVE_STUDY / "fixture.py"))["generate"](tmp_path)
    result = observe(tmp_path, mode)
    assert result["validated_rows"] == 4160 and result["finish_rows"] == 0
    assert len(result["append_ns"]) == 64
    if mode == "plain":
        assert result["phases_ns"] == [] and result["calibration_ns"] is None
    else:
        assert len(result["phases_ns"]) == 64
        assert all(
            sum(parts) == total
            for total, parts in zip(
                result["append_ns"], result["phases_ns"], strict=True
            )
        )
        assert all(min(parts) > 0 for parts in result["phases_ns"])
        assert result["calibration_ns"] >= 0
