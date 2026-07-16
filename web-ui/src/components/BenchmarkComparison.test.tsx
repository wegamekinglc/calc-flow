import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { BenchmarkComparison } from './BenchmarkComparison';

const fingerprint = (digit: string): string => digit.repeat(64);

const entry = (mean = 1, machineFingerprint = fingerprint('a')) => ({
  name: 'test_array_mean_numpy_plan_end_to_end',
  fullname: 'benchmarks/test_array.py::test_array_mean_numpy_plan_end_to_end',
  stats: { mean, stddev: 0.01, rounds: 10 },
  extra_info: {
    benchmark_contract_version: 2,
    workload_version: 1,
    scenario: 'array_mean',
    scope: 'plan_end_to_end',
    backend: 'numpy',
    scale: 'overhead',
    table_rows: 1_000,
    array_elements: 1_000,
    matrix_dimension: 16,
    input_rows: 1_000,
    output_rows: 1,
    expression: 'mean(x)',
    input_dtype: 'float64',
    output_dtype: 'float64',
    machine_identity: {
      operating_system: 'linux',
      architecture: 'x86_64',
      cpu_brand: 'example cpu',
      logical_cpu_count: 8,
      python_implementation: 'cpython',
    },
    dependency_identity: { python_version: '3.13.9', numpy_version: '2.5.1' },
    backend_configuration: {},
    workload_identity: {
      benchmark_contract_version: 2,
      scenario: 'array_mean',
      scope: 'plan_end_to_end',
      workload_version: 1,
      backend: 'numpy',
      scale: 'overhead',
      table_rows: 1_000,
      array_elements: 1_000,
      matrix_dimension: 16,
      input_rows: 1_000,
      output_rows: 1,
      expression: 'mean(x)',
      input_dtype: 'float64',
      output_dtype: 'float64',
      backend_configuration: {},
    },
    machine_fingerprint: machineFingerprint,
    dependency_fingerprint: fingerprint('b'),
    workload_fingerprint: fingerprint('c'),
    python_version: '3.13.9',
    numpy_version: '2.5.1',
    process_rss_bytes: 100_000,
  },
});

const report = (benchmark: unknown) => ({ benchmarks: [benchmark] });

const legacyReport = () => report({
  name: 'test_projection',
  fullname: 'benchmarks/test_datafusion.py::test_projection',
  stats: { mean: 1, stddev: 0.01, rounds: 10 },
  extra_info: { scenario: 'datafusion_projection', scale: 'overhead' },
});

const jsonFile = (value: unknown, name: string): File => {
  const contents = typeof value === 'string' ? value : JSON.stringify(value);
  const file = new File([contents], name, { type: 'application/json' });
  Object.defineProperty(file, 'text', { value: async () => contents });
  return file;
};

const upload = (label: string, value: unknown) => {
  fireEvent.change(screen.getByLabelText(label), {
    target: { files: [jsonFile(value, `${label}.json`)] },
  });
};

describe('BenchmarkComparison', () => {
  it('shows legacy reports as unverified without a classification', async () => {
    render(<BenchmarkComparison />);

    upload('Baseline benchmark report', legacyReport());
    upload('Current benchmark report', legacyReport());

    expect(await screen.findByText('Unverified')).toBeInTheDocument();
    expect(screen.getByText(/No performance classification was made/)).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('shows machine incompatibility details without a timing table', async () => {
    render(<BenchmarkComparison />);

    upload('Baseline benchmark report', report(entry()));
    upload('Current benchmark report', report(entry(1.2, fingerprint('d'))));

    expect(await screen.findByText('machine_mismatch')).toBeInTheDocument();
    expect(screen.getByText('machine_fingerprint')).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('renders classifications only for compatible reports', async () => {
    render(<BenchmarkComparison />);

    upload('Baseline benchmark report', report(entry()));
    upload('Current benchmark report', report(entry(1.2)));

    expect(await screen.findByRole('table')).toBeInTheDocument();
    expect(screen.getByText('regression')).toBeInTheDocument();
    expect(screen.queryByText('Unverified')).not.toBeInTheDocument();
  });

  it('shows malformed JSON parser errors', async () => {
    render(<BenchmarkComparison />);

    upload('Baseline benchmark report', '{ definitely not JSON');

    expect(await screen.findByText(/Unexpected|JSON/)).toBeInTheDocument();
  });

  it('clears a stale comparison when a loaded file is replaced', async () => {
    render(<BenchmarkComparison />);

    upload('Baseline benchmark report', report(entry()));
    upload('Current benchmark report', report(entry(1.2)));
    expect(await screen.findByRole('table')).toBeInTheDocument();

    upload('Current benchmark report', '{ definitely not JSON');

    await screen.findByText(/Unexpected|JSON/);
    await waitFor(() => expect(screen.queryByRole('table')).not.toBeInTheDocument());
  });
});
