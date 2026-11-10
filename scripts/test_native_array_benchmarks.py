from __future__ import annotations

import unittest

import numpy as np
import pyarrow as pa

import calc_flow.array as cfa
from benchmarks.native_array_cases import EAGER_CASES, eager_case
from benchmarks.native_array_support import measure_native
from benchmarks.support import BenchmarkScale
from scripts.benchmark_suite.identity import validate_identity


class NativeArrayBenchmarkTests(unittest.TestCase):
    def test_eager_workloads_validate_real_native_outputs(self) -> None:
        scale = BenchmarkScale("overhead", 64, 64, 4)
        for name in EAGER_CASES:
            with self.subTest(name=name):
                case = eager_case(name, scale)
                case.validate(case.calculate())

    def test_oracle_rejects_wrong_values_with_the_correct_shape(self) -> None:
        case = eager_case("sum", BenchmarkScale("overhead", 64, 64, 4))
        wrong = cfa.from_arrow(pa.array([0.0]), shape=())
        with self.assertRaises(AssertionError):
            case.validate(wrong)

    def test_oracle_distinguishes_null_warmup_from_valid_nan(self) -> None:
        case = eager_case("rsi", BenchmarkScale("overhead", 64, 64, 4))
        wrong = cfa.from_arrow(pa.array(np.full(64, np.nan)))
        with self.assertRaises(AssertionError):
            case.validate(wrong)

    def test_identity_is_recorded_before_fixed_sample_collection(self) -> None:
        case = eager_case("macd", BenchmarkScale("overhead", 64, 64, 4))
        benchmark = _Recorder()
        measure_native(benchmark, case)
        self.assertEqual(benchmark.sampling, (10, 1, 1))
        metadata = benchmark.snapshot
        validate_identity(metadata)
        self.assertEqual(metadata["backend"], "native")
        self.assertEqual(metadata["input_rows"], 64)
        self.assertEqual(metadata["output_rows"], 64)
        workload = metadata["workload_identity"]
        self.assertEqual(workload["scope"], "native-eager-api-v1")
        self.assertEqual(workload["output_shapes"], [[64], [64], [64]])
        self.assertEqual(workload["parameters"]["signal_span"], 9)
        self.assertRegex(workload["input_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(metadata["native_binary_sha256"], r"^[0-9a-f]{64}$")


class _Recorder:
    def __init__(self) -> None:
        self.extra_info = {}
        self.snapshot = {}
        self.sampling = ()

    def pedantic(self, function, *, rounds, iterations, warmup_rounds):
        self.snapshot = dict(self.extra_info)
        self.sampling = (rounds, iterations, warmup_rounds)
        return function()


if __name__ == "__main__":
    unittest.main()
