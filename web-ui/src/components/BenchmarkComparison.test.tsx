import { describe, expect, it } from 'vitest';

import {
  compareBenchmarkReports,
  parseBenchmarkReport,
  type BenchmarkReport,
} from './BenchmarkComparison';

const report = (
  mean: number,
  stddev: number,
  scenario = 'datafusion_projection',
): BenchmarkReport => ({
  benchmarks: [{
    name: 'test_projection',
    fullname: 'benchmarks/test_datafusion.py::test_projection',
    stats: { mean, stddev, rounds: 10 },
    extra_info: { scenario, scale: 'overhead' },
  }],
});

describe('benchmark report comparison', () => {
  it('classifies stable regressions and improvements', () => {
    const regression = compareBenchmarkReports(report(1, 0.02), report(1.2, 0.02));
    const improvement = compareBenchmarkReports(report(1, 0.02), report(0.8, 0.02));

    expect(regression[0].status).toBe('regression');
    expect(regression[0].deltaPercent).toBeCloseTo(20);
    expect(improvement[0].status).toBe('improvement');
    expect(improvement[0].deltaPercent).toBeCloseTo(-20);
  });

  it('keeps noisy cases informational', () => {
    const rows = compareBenchmarkReports(report(1, 0.2), report(1.3, 0.3));

    expect(rows[0].status).toBe('noisy');
  });

  it('matches array cases by scenario and backend', () => {
    const baseline = report(1, 0.01);
    baseline.benchmarks[0].extra_info = { scenario: 'array_mean', backend: 'numpy' };
    const current = report(1, 0.01);
    current.benchmarks[0].extra_info = { scenario: 'array_mean', backend: 'jax' };

    expect(compareBenchmarkReports(baseline, current)).toEqual([]);
  });

  it('does not compare different dataset scales', () => {
    const baseline = report(1, 0.01);
    const current = report(1, 0.01);
    current.benchmarks[0].extra_info = {
      scenario: 'datafusion_projection',
      scale: 'standard',
    };

    expect(compareBenchmarkReports(baseline, current)).toEqual([]);
  });

  it('rejects malformed pytest-benchmark documents', () => {
    expect(() => parseBenchmarkReport({})).toThrow(/pytest-benchmark/);
    expect(() => parseBenchmarkReport({ benchmarks: [] })).toThrow(/no cases/);
    expect(() => parseBenchmarkReport({ benchmarks: [{ name: 'bad' }] })).toThrow(/fullname/);
  });
});
