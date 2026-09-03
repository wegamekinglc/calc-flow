"""Correctness tests for the rolling-indicator comparison harness."""

from __future__ import annotations

import numpy as np

from benchmarks.rolling_indicator_comparison import (
    DEFAULT_ROW_SCALES,
    INDICATOR_DUAL_SMA_SPREAD,
    TaLibMethod,
    build_calc_flow_methods,
    expected_dual_sma_spread,
    expected_rolling_mean,
    iterations_per_sample,
    rolling_workload,
    ta_lib_expected_dual_sma_spread,
    ta_lib_expected_rolling_mean,
    ta_lib_iterations_per_sample,
    ta_lib_valid_rows,
)


def test_default_matrix_includes_small_row_scales() -> None:
    assert DEFAULT_ROW_SCALES[:3] == (10, 100, 1_000)


def test_small_scales_are_amortized_but_large_scales_run_once() -> None:
    assert iterations_per_sample(10) == 50
    assert iterations_per_sample(1_000) == 50
    assert iterations_per_sample(10_000) == 5
    assert iterations_per_sample(100_000) == 1
    assert ta_lib_iterations_per_sample(10) == 200
    assert ta_lib_iterations_per_sample(10_000) == 200
    assert ta_lib_iterations_per_sample(100_000) == 50
    assert ta_lib_iterations_per_sample(1_000_000) == 5


def test_ta_lib_sma_matches_oracle_after_its_warmup() -> None:
    workload = rolling_workload(rows=257, entities=8)
    actual = TaLibMethod(workload.prices, entities=8, window=20).run()
    expected = ta_lib_expected_rolling_mean(workload.prices, entities=8, window=20)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert ta_lib_valid_rows(rows=257, entities=8, window=20) == 105
    assert ta_lib_valid_rows(rows=1_000, entities=64, window=20) == 0


def test_dual_sma_spread_matches_partial_and_full_window_oracles() -> None:
    workload = rolling_workload(rows=257, entities=8)
    expected = expected_dual_sma_spread(
        workload.prices,
        entities=8,
        fast_window=5,
        slow_window=20,
    )
    methods = build_calc_flow_methods(
        workload,
        window=20,
        indicator=INDICATOR_DUAL_SMA_SPREAD,
        fast_window=5,
    )

    for method in methods.values():
        np.testing.assert_allclose(method.execute(), expected, rtol=1e-12, atol=1e-12)

    ta_lib_actual = TaLibMethod(
        workload.prices,
        entities=8,
        window=20,
        fast_window=5,
    ).run()
    ta_lib_expected = ta_lib_expected_dual_sma_spread(
        workload.prices,
        entities=8,
        fast_window=5,
        slow_window=20,
    )
    np.testing.assert_allclose(
        ta_lib_actual,
        ta_lib_expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_calc_flow_rolling_methods_match_incremental_oracle() -> None:
    workload = rolling_workload(rows=257, entities=8)
    methods = build_calc_flow_methods(workload, window=20)
    expected = expected_rolling_mean(workload.prices, entities=8, window=20)

    for method in methods.values():
        actual = method.execute()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
