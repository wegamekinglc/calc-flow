from __future__ import annotations

from benchmarks.support import SCALES


def test_overhead_table_and_array_scales_match() -> None:
    scale = SCALES["overhead"]

    assert scale.table_rows == scale.array_elements == 1_000
