import { bench, describe } from 'vitest';

import { compareBenchmarkReports, parseBenchmarkReport } from './benchmarkComparisonModel';

const fingerprint = (index: number): string => index.toString(16).padStart(64, '0');

const entry = (index: number, mean: number) => {
  const scenario = `large_report_case_${index}`;
  const backendConfiguration = {};
  const workloadIdentity = {
    benchmark_contract_version: 2,
    scenario,
    scope: 'plan_end_to_end',
    workload_version: 1,
    backend: 'numpy',
    scale: 'standard',
    table_rows: 100_000,
    array_elements: 100_000,
    matrix_dimension: 256,
    input_rows: 100_000,
    output_rows: 1,
    expression: `mean(x_${index})`,
    input_dtype: 'float64',
    output_dtype: 'float64',
    backend_configuration: backendConfiguration,
  };
  return {
    name: scenario,
    fullname: `benchmarks/test_large.py::test_${scenario}`,
    stats: { mean, stddev: 0.001, rounds: 20 },
    extra_info: {
      ...workloadIdentity,
      machine_identity: {
        operating_system: 'linux',
        architecture: 'x86_64',
        cpu_brand: 'benchmark cpu',
        logical_cpu_count: 8,
        python_implementation: 'cpython',
      },
      dependency_identity: {
        python_version: '3.13.9',
        numpy_version: '2.5.1',
      },
      workload_identity: workloadIdentity,
      machine_fingerprint: 'a'.repeat(64),
      dependency_fingerprint: 'b'.repeat(64),
      workload_fingerprint: fingerprint(index + 1),
      python_version: '3.13.9',
      numpy_version: '2.5.1',
      process_rss_bytes: 100_000,
    },
  };
};

const reports = (cases: number) => {
  const baseline = parseBenchmarkReport({
    benchmarks: Array.from({ length: cases }, (_, index) => entry(index, 1)),
  });
  const current = parseBenchmarkReport({
    benchmarks: Array.from({ length: cases }, (_, index) => entry(index, 1.01)),
  });
  return { baseline, current };
};

describe('benchmark report matching', () => {
  for (const cases of [100, 1_000]) {
    const { baseline, current } = reports(cases);
    bench(`compare/${cases}_cases`, () => {
      compareBenchmarkReports(baseline, current);
    });
  }
});
