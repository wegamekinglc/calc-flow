import { describe, expect, it } from 'vitest';

import { at } from '../types';
import {
  compareBenchmarkReports,
  parseBenchmarkReport,
  type BenchmarkCompatibilityIssue,
  type BenchmarkReport,
} from './benchmarkComparisonModel';

const fingerprint = (digit: string): string => digit.repeat(64);

const contractEntry = ({
  backend = 'numpy',
  mean = 1,
  scenario = 'array_mean',
  scope = 'plan_end_to_end',
  stddev = 0.01,
  workloadFingerprint = fingerprint('c'),
}: {
  backend?: 'numpy' | 'jax';
  mean?: number;
  scenario?: string;
  scope?: string;
  stddev?: number;
  workloadFingerprint?: string;
} = {}) => {
  const backendConfiguration = backend === 'jax'
    ? { jax_platform: 'cpu', jax_enable_x64: false }
    : {};
  const dependencyIdentity = {
    python_version: '3.13.9',
    numpy_version: '2.5.1',
    ...(backend === 'jax'
      ? { jax_version: '0.10.2', jaxlib_version: '0.10.2' }
      : {}),
  };
  const workloadIdentity = {
    benchmark_contract_version: 2,
    scenario,
    scope,
    workload_version: 1,
    backend,
    scale: 'overhead',
    table_rows: 1_000,
    array_elements: 1_000,
    matrix_dimension: 16,
    input_rows: 1_000,
    output_rows: 1,
    expression: 'mean(x)',
    input_dtype: 'float64',
    output_dtype: 'float64',
    backend_configuration: backendConfiguration,
  };

  return {
    name: `test_${scenario}_${backend}_${scope}`,
    fullname: `benchmarks/test_array.py::test_${scenario}_${backend}_${scope}`,
    stats: { mean, stddev, rounds: 10 },
    extra_info: {
      benchmark_contract_version: 2,
      workload_version: 1,
      scenario,
      scope,
      backend,
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
      dependency_identity: dependencyIdentity,
      backend_configuration: backendConfiguration,
      workload_identity: workloadIdentity,
      machine_fingerprint: fingerprint('a'),
      dependency_fingerprint: fingerprint('b'),
      workload_fingerprint: workloadFingerprint,
      python_version: '3.13.9',
      numpy_version: '2.5.1',
      ...(backend === 'jax'
        ? {
            jax_version: '0.10.2',
            jaxlib_version: '0.10.2',
            jax_platform: 'cpu',
            jax_enable_x64: false,
          }
        : {}),
      process_rss_bytes: 100_000,
    },
  };
};

const legacyEntry = (scenario = 'legacy_projection') => ({
  name: `test_${scenario}`,
  fullname: `benchmarks/test_legacy.py::test_${scenario}`,
  stats: { mean: 1, stddev: 0.01, rounds: 10 },
  extra_info: { scenario, scale: 'overhead' },
});

const parsedReport = (...benchmarks: unknown[]): BenchmarkReport =>
  parseBenchmarkReport({ benchmarks });

const cloneEntry = (entry: ReturnType<typeof contractEntry>) => structuredClone(entry);

describe('benchmark report compatibility', () => {
  it.each([
    ['stable', 1.05, 0.01],
    ['noisy', 1.3, 0.2],
    ['regression', 1.2, 0.01],
    ['improvement', 0.8, 0.01],
  ] as const)('classifies equal v2 identities as %s', (status, mean, stddev) => {
    const result = compareBenchmarkReports(
      parsedReport(contractEntry()),
      parsedReport(contractEntry({ mean, stddev })),
    );

    expect(result.status).toBe('compatible');
    expect(result.issues).toEqual([]);
    expect(result.rows).toHaveLength(1);
    expect(at(result.rows).status).toBe(status);
  });

  it('rejects different machine identities with the same claimed fingerprint', () => {
    const baselineEntry = contractEntry();
    const currentEntry = cloneEntry(baselineEntry);
    (currentEntry.extra_info.machine_identity as Record<string, unknown>).cpu_brand =
      'different cpu';

    const result = compareBenchmarkReports(
      parsedReport(baselineEntry),
      parsedReport(currentEntry),
    );

    expect(result.status).toBe('incompatible');
    expect(result.rows).toEqual([]);
    expect(result.issues.map((issue) => issue.code)).toEqual(['machine_mismatch']);
  });

  it('rejects different dependency identities with the same claimed fingerprint', () => {
    const baselineEntry = contractEntry();
    const currentEntry = cloneEntry(baselineEntry);
    currentEntry.extra_info.numpy_version = '2.6.0';
    (currentEntry.extra_info.dependency_identity as Record<string, unknown>).numpy_version =
      '2.6.0';

    const result = compareBenchmarkReports(
      parsedReport(baselineEntry),
      parsedReport(currentEntry),
    );

    expect(result.status).toBe('incompatible');
    expect(result.rows).toEqual([]);
    expect(result.issues.map((issue) => issue.code)).toEqual(['dependency_mismatch']);
  });

  it('rejects different workload identities with the same claimed fingerprint', () => {
    const baselineEntry = contractEntry();
    const currentEntry = cloneEntry(baselineEntry);
    currentEntry.extra_info.expression = 'sum(x)';
    (currentEntry.extra_info.workload_identity as Record<string, unknown>).expression =
      'sum(x)';

    const result = compareBenchmarkReports(
      parsedReport(baselineEntry),
      parsedReport(currentEntry),
    );

    expect(result.status).toBe('incompatible');
    expect(result.rows).toEqual([]);
    expect(result.issues.map((issue) => issue.code)).toEqual(['workload_mismatch']);
  });

  it('treats identity documents with different key order as compatible', () => {
    const baselineEntry = contractEntry({ backend: 'jax' });
    const currentEntry = cloneEntry(baselineEntry);
    const reversed = <T extends object>(value: T): T =>
      Object.fromEntries(Object.entries(value).reverse()) as T;
    currentEntry.extra_info.machine_identity = reversed(
      currentEntry.extra_info.machine_identity,
    );
    currentEntry.extra_info.dependency_identity = reversed(
      currentEntry.extra_info.dependency_identity,
    );
    currentEntry.extra_info.backend_configuration = reversed(
      currentEntry.extra_info.backend_configuration,
    );
    const workload = reversed(
      currentEntry.extra_info.workload_identity,
    );
    workload.backend_configuration = reversed(
      workload.backend_configuration as Record<string, unknown>,
    );
    currentEntry.extra_info.workload_identity = workload;

    const result = compareBenchmarkReports(
      parsedReport(baselineEntry),
      parsedReport(currentEntry),
    );

    expect(result.status).toBe('compatible');
    expect(result.issues).toEqual([]);
    expect(result.rows).toHaveLength(1);
  });

  it.each<{
    code: BenchmarkCompatibilityIssue['code'];
    expectedStatus?: 'incompatible' | 'unverified';
    change: (entry: ReturnType<typeof contractEntry>) => void;
  }>([
    {
      code: 'missing_contract_metadata',
      expectedStatus: 'unverified',
      change: (entry) => {
        delete (entry.extra_info as Record<string, unknown>).benchmark_contract_version;
      },
    },
    {
      code: 'contract_version_mismatch',
      change: (entry) => { entry.extra_info.benchmark_contract_version = 3; },
    },
    {
      code: 'machine_mismatch',
      change: (entry) => { entry.extra_info.machine_fingerprint = fingerprint('d'); },
    },
    {
      code: 'dependency_mismatch',
      change: (entry) => { entry.extra_info.dependency_fingerprint = fingerprint('d'); },
    },
    {
      code: 'scale_mismatch',
      change: (entry) => {
        entry.extra_info.array_elements = 2_000;
        (entry.extra_info.workload_identity as Record<string, unknown>).array_elements = 2_000;
      },
    },
    {
      code: 'scope_mismatch',
      change: (entry) => {
        entry.extra_info.scope = 'provider_boundary';
        (entry.extra_info.workload_identity as Record<string, unknown>).scope =
          'provider_boundary';
      },
    },
    {
      code: 'workload_mismatch',
      change: (entry) => { entry.extra_info.workload_fingerprint = fingerprint('d'); },
    },
    {
      code: 'dtype_mismatch',
      change: (entry) => {
        entry.extra_info.output_dtype = 'float32';
        (entry.extra_info.workload_identity as Record<string, unknown>).output_dtype = 'float32';
      },
    },
    {
      code: 'backend_configuration_mismatch',
      change: (entry) => {
        entry.extra_info.jax_enable_x64 = true;
        (entry.extra_info.backend_configuration as Record<string, unknown>).jax_enable_x64 = true;
        const workload = entry.extra_info.workload_identity as Record<string, unknown>;
        (workload.backend_configuration as Record<string, unknown>).jax_enable_x64 = true;
      },
    },
  ])('returns no rows for $code', ({ code, expectedStatus = 'incompatible', change }) => {
    const baselineEntry = contractEntry({ backend: code === 'backend_configuration_mismatch' ? 'jax' : 'numpy' });
    const currentEntry = cloneEntry(baselineEntry);
    change(currentEntry);

    const result = compareBenchmarkReports(
      parsedReport(baselineEntry),
      parsedReport(currentEntry),
    );

    expect(result.status).toBe(expectedStatus);
    expect(result.rows).toEqual([]);
    expect(result.issues.map((issue) => issue.code)).toEqual([code]);
  });

  it('keeps legacy-only reports unverified', () => {
    const result = compareBenchmarkReports(
      parsedReport(legacyEntry()),
      parsedReport(legacyEntry()),
    );

    expect(result.status).toBe('unverified');
    expect(result.rows).toEqual([]);
    expect(result.issues.map((issue) => issue.code)).toEqual(['missing_contract_metadata']);
  });

  it('compares matched v2 entries while ignoring unrelated legacy entries', () => {
    const result = compareBenchmarkReports(
      parsedReport(contractEntry(), legacyEntry('old_case')),
      parsedReport(contractEntry({ mean: 1.2 }), legacyEntry('other_old_case')),
    );

    expect(result.status).toBe('compatible');
    expect(result.rows).toHaveLength(1);
    expect(at(result.rows).status).toBe('regression');
    expect(result.issues).toEqual([]);
  });

  it.each(['baseline', 'current'] as const)(
    'rejects an unsupported version mixed into the %s report',
    (target) => {
      const compatibleBaseline = contractEntry();
      const compatibleCurrent = contractEntry({ mean: 1.2 });
      const unsupported = contractEntry({
        scenario: 'future_contract_case',
        workloadFingerprint: fingerprint('d'),
      });
      unsupported.extra_info.benchmark_contract_version = 3;
      const baselineEntries = target === 'baseline'
        ? [compatibleBaseline, unsupported]
        : [compatibleBaseline];
      const currentEntries = target === 'current'
        ? [compatibleCurrent, unsupported]
        : [compatibleCurrent];

      const result = compareBenchmarkReports(
        parsedReport(...baselineEntries),
        parsedReport(...currentEntries),
      );

      expect(result.status).toBe('incompatible');
      expect(result.rows).toEqual([]);
      expect(result.issues.map((issue) => issue.code)).toContain('contract_version_mismatch');
    },
  );

  it('does not fall back from a v2 case to a legacy case with the same scenario', () => {
    const result = compareBenchmarkReports(
      parsedReport(contractEntry()),
      parsedReport(legacyEntry('array_mean')),
    );

    expect(result.status).toBe('unverified');
    expect(result.rows).toEqual([]);
    expect(result.issues.map((issue) => issue.code)).toEqual(['missing_contract_metadata']);
  });

  it('reports no matching v2 work without classifying it', () => {
    const result = compareBenchmarkReports(
      parsedReport(contractEntry({ scenario: 'array_mean' })),
      parsedReport(contractEntry({ scenario: 'array_sum', workloadFingerprint: fingerprint('d') })),
    );

    expect(result).toEqual({ status: 'unverified', rows: [], issues: [] });
  });

  it('orders issues by the compatibility issue union declaration', () => {
    const baselineEntry = contractEntry({ backend: 'jax' });
    const currentEntry = cloneEntry(baselineEntry);
    Object.assign(currentEntry.extra_info, {
      machine_fingerprint: fingerprint('d'),
      dependency_fingerprint: fingerprint('e'),
      scale: 'standard',
      scope: 'provider_boundary',
      workload_fingerprint: fingerprint('f'),
      input_dtype: 'float32',
      jax_platform: 'gpu',
    });
    const workload = currentEntry.extra_info.workload_identity as Record<string, unknown>;
    Object.assign(workload, {
      scale: 'standard',
      scope: 'provider_boundary',
      input_dtype: 'float32',
    });
    Object.assign(currentEntry.extra_info.backend_configuration as Record<string, unknown>, {
      jax_platform: 'gpu',
    });
    Object.assign(workload.backend_configuration as Record<string, unknown>, {
      jax_platform: 'gpu',
    });

    const result = compareBenchmarkReports(
      parsedReport(baselineEntry),
      parsedReport(currentEntry),
    );

    expect(result.rows).toEqual([]);
    expect(result.issues.map((issue) => issue.code)).toEqual([
      'machine_mismatch',
      'dependency_mismatch',
      'scale_mismatch',
      'scope_mismatch',
      'workload_mismatch',
      'dtype_mismatch',
      'backend_configuration_mismatch',
    ]);
  });

  it('keeps union declaration order across multiple matched cases', () => {
    const workloadBaseline = contractEntry({
      scenario: 'workload_case',
      workloadFingerprint: fingerprint('1'),
    });
    const workloadCurrent = cloneEntry(workloadBaseline);
    workloadCurrent.extra_info.workload_fingerprint = fingerprint('3');
    const machineBaseline = contractEntry({
      scenario: 'machine_case',
      workloadFingerprint: fingerprint('2'),
    });
    const machineCurrent = cloneEntry(machineBaseline);
    machineCurrent.extra_info.machine_fingerprint = fingerprint('d');

    const result = compareBenchmarkReports(
      parsedReport(workloadBaseline, machineBaseline),
      parsedReport(workloadCurrent, machineCurrent),
    );

    expect(result.issues.map((issue) => issue.code)).toEqual([
      'machine_mismatch',
      'workload_mismatch',
    ]);
  });

  it('sorts compatible rows by descending delta', () => {
    const baseline = parsedReport(
      contractEntry({ scenario: 'improved', workloadFingerprint: fingerprint('1') }),
      contractEntry({ scenario: 'regressed', workloadFingerprint: fingerprint('2') }),
      contractEntry({ scenario: 'steady', workloadFingerprint: fingerprint('3') }),
    );
    const current = parsedReport(
      contractEntry({ scenario: 'improved', mean: 0.8, workloadFingerprint: fingerprint('1') }),
      contractEntry({ scenario: 'regressed', mean: 1.3, workloadFingerprint: fingerprint('2') }),
      contractEntry({ scenario: 'steady', mean: 1.05, workloadFingerprint: fingerprint('3') }),
    );

    expect(compareBenchmarkReports(baseline, current).rows.map((row) => row.scenario)).toEqual([
      'regressed',
      'steady',
      'improved',
    ]);
  });

  it('matches 1,000 compatible cases without changing report order', () => {
    const cases = Array.from({ length: 1_000 }, (_, index) => {
      const scenario = `large_case_${index}`;
      const workloadFingerprint = index.toString(16).padStart(64, '0');
      const baseline = contractEntry({ scenario, workloadFingerprint });
      return [baseline, cloneEntry(baseline)] as const;
    });

    const result = compareBenchmarkReports(
      parsedReport(...cases.map(([baseline]) => baseline)),
      parsedReport(...cases.map(([, current]) => current)),
    );

    expect(result.status).toBe('compatible');
    expect(result.rows).toHaveLength(1_000);
    expect(result.rows.map((row) => row.scenario)).toEqual(
      cases.map(([baseline]) => baseline.extra_info.scenario),
    );
  });
});

describe('benchmark report parsing', () => {
  it.each([
    ['non-integer workload version', (entry: ReturnType<typeof contractEntry>) => { entry.extra_info.workload_version = 1.5; }],
    ['empty expression', (entry: ReturnType<typeof contractEntry>) => { entry.extra_info.expression = ''; }],
    ['unsupported scope', (entry: ReturnType<typeof contractEntry>) => { entry.extra_info.scope = 'unknown'; }],
    ['uppercase fingerprint', (entry: ReturnType<typeof contractEntry>) => { entry.extra_info.machine_fingerprint = 'A'.repeat(64); }],
    ['missing JAX platform', (entry: ReturnType<typeof contractEntry>) => { delete entry.extra_info.jax_platform; }],
  ])('rejects malformed contract-v2 metadata: %s', (_name, change) => {
    const entry = contractEntry({ backend: _name === 'missing JAX platform' ? 'jax' : 'numpy' });
    change(entry);

    expect(() => parsedReport(entry)).toThrow(/contract-v2/);
  });

  it('rejects JAX-only fields on NumPy entries', () => {
    const entry = contractEntry();
    Object.assign(entry.extra_info, { jax_platform: 'cpu', jax_enable_x64: false });

    expect(() => parsedReport(entry)).toThrow(/NumPy.*JAX/);
  });

  it.each<{
    name: string;
    field: string;
    change: (entry: ReturnType<typeof contractEntry>) => void;
  }>([
    {
      name: 'contract version',
      field: 'workload_identity.benchmark_contract_version',
      change: (entry) => {
        (entry.extra_info.workload_identity as Record<string, unknown>)
          .benchmark_contract_version = 3;
      },
    },
    {
      name: 'scenario',
      field: 'workload_identity.scenario',
      change: (entry) => {
        (entry.extra_info.workload_identity as Record<string, unknown>).scenario =
          'array_sum';
      },
    },
    {
      name: 'backend',
      field: 'workload_identity.backend',
      change: (entry) => {
        const workload = entry.extra_info.workload_identity as Record<string, unknown>;
        workload.backend = 'jax';
        workload.backend_configuration = {
          jax_platform: 'cpu',
          jax_enable_x64: false,
        };
      },
    },
    {
      name: 'scope',
      field: 'workload_identity.scope',
      change: (entry) => {
        (entry.extra_info.workload_identity as Record<string, unknown>).scope =
          'provider_boundary';
      },
    },
    {
      name: 'workload version',
      field: 'workload_identity.workload_version',
      change: (entry) => {
        (entry.extra_info.workload_identity as Record<string, unknown>).workload_version = 2;
      },
    },
    {
      name: 'scale',
      field: 'workload_identity.scale',
      change: (entry) => {
        (entry.extra_info.workload_identity as Record<string, unknown>).scale = 'small';
      },
    },
    ...[
      'table_rows',
      'array_elements',
      'matrix_dimension',
      'input_rows',
      'output_rows',
    ].map((field) => ({
      name: field,
      field: `workload_identity.${field}`,
      change: (entry: ReturnType<typeof contractEntry>) => {
        (entry.extra_info.workload_identity as Record<string, unknown>)[field] = 2_000;
      },
    })),
    {
      name: 'expression',
      field: 'workload_identity.expression',
      change: (entry) => {
        (entry.extra_info.workload_identity as Record<string, unknown>).expression = 'sum(x)';
      },
    },
    {
      name: 'input dtype',
      field: 'workload_identity.input_dtype',
      change: (entry) => {
        (entry.extra_info.workload_identity as Record<string, unknown>).input_dtype = 'float32';
      },
    },
    {
      name: 'output dtype',
      field: 'workload_identity.output_dtype',
      change: (entry) => {
        (entry.extra_info.workload_identity as Record<string, unknown>).output_dtype = 'float32';
      },
    },
    {
      name: 'backend configuration',
      field: 'workload_identity.backend_configuration.jax_enable_x64',
      change: (entry) => {
        const workload = entry.extra_info.workload_identity as Record<string, unknown>;
        workload.backend_configuration = {
          jax_platform: 'cpu',
          jax_enable_x64: true,
        };
      },
    },
  ])('rejects a flat/nested workload mismatch: $name', ({ field, change }) => {
    const entry = contractEntry({ backend: field.includes('backend_configuration') ? 'jax' : 'numpy' });
    change(entry);

    expect(() => parsedReport(entry)).toThrow(field);
  });

  it.each<{
    name: string;
    field: string;
    change: (entry: ReturnType<typeof contractEntry>) => void;
  }>([
    {
      name: 'Python version',
      field: 'python_version',
      change: (entry) => { entry.extra_info.python_version = '3.14.0'; },
    },
    {
      name: 'NumPy version',
      field: 'numpy_version',
      change: (entry) => { entry.extra_info.numpy_version = '2.6.0'; },
    },
    {
      name: 'JAX version',
      field: 'jax_version',
      change: (entry) => { entry.extra_info.jax_version = '0.11.0'; },
    },
    {
      name: 'JAXlib version',
      field: 'jaxlib_version',
      change: (entry) => { entry.extra_info.jaxlib_version = '0.11.0'; },
    },
    {
      name: 'JAX platform',
      field: 'jax_platform',
      change: (entry) => { entry.extra_info.jax_platform = 'gpu'; },
    },
    {
      name: 'JAX x64 mode',
      field: 'jax_enable_x64',
      change: (entry) => { entry.extra_info.jax_enable_x64 = true; },
    },
  ])('rejects a flat/nested dependency or backend mismatch: $name', ({ name, field, change }) => {
    const entry = contractEntry({ backend: name.startsWith('JAX') ? 'jax' : 'numpy' });
    change(entry);

    expect(() => parsedReport(entry)).toThrow(field);
  });

  it.each<{
    name: string;
    backend?: 'numpy' | 'jax';
    field: string;
    change: (entry: ReturnType<typeof contractEntry>) => void;
  }>([
    {
      name: 'incomplete machine identity',
      field: 'machine_identity.operating_system',
      change: (entry) => {
        delete (entry.extra_info.machine_identity as Record<string, unknown>).operating_system;
      },
    },
    {
      name: 'wrong-typed machine identity',
      field: 'machine_identity.logical_cpu_count',
      change: (entry) => {
        (entry.extra_info.machine_identity as Record<string, unknown>).logical_cpu_count = 'eight';
      },
    },
    {
      name: 'incomplete dependency identity',
      field: 'dependency_identity.numpy_version',
      change: (entry) => {
        delete (entry.extra_info.dependency_identity as Record<string, unknown>).numpy_version;
      },
    },
    {
      name: 'wrong-typed JAX dependency identity',
      backend: 'jax',
      field: 'dependency_identity.jax_version',
      change: (entry) => {
        (entry.extra_info.dependency_identity as Record<string, unknown>).jax_version = 10;
      },
    },
    {
      name: 'JAX dependency fields on a NumPy identity',
      field: 'dependency_identity',
      change: (entry) => {
        (entry.extra_info.dependency_identity as Record<string, unknown>).jax_version = '0.10.2';
      },
    },
    {
      name: 'non-empty NumPy backend configuration',
      field: 'backend_configuration',
      change: (entry) => {
        (entry.extra_info as Record<string, unknown>).backend_configuration = {
          unexpected: true,
        };
      },
    },
    {
      name: 'incomplete workload identity',
      field: 'workload_identity.expression',
      change: (entry) => {
        delete (entry.extra_info.workload_identity as Record<string, unknown>).expression;
      },
    },
    {
      name: 'wrong-typed workload identity',
      field: 'workload_identity.array_elements',
      change: (entry) => {
        (entry.extra_info.workload_identity as Record<string, unknown>).array_elements = '1000';
      },
    },
    {
      name: 'incomplete workload JAX configuration',
      backend: 'jax',
      field: 'workload_identity.backend_configuration.jax_enable_x64',
      change: (entry) => {
        const workload = entry.extra_info.workload_identity as Record<string, unknown>;
        const configuration = structuredClone(
          workload.backend_configuration as Record<string, unknown>,
        );
        delete configuration.jax_enable_x64;
        workload.backend_configuration = configuration;
      },
    },
  ])('rejects $name', ({ backend = 'numpy', field, change }) => {
    const entry = contractEntry({ backend });
    change(entry);

    expect(() => parsedReport(entry)).toThrow(field);
  });

  it('rejects duplicate scenario, backend, and scope identities', () => {
    expect(() => parsedReport(
      contractEntry({ workloadFingerprint: fingerprint('1') }),
      contractEntry({ workloadFingerprint: fingerprint('2') }),
    )).toThrow(/duplicate.*scenario.*backend.*scope/i);
  });

  it.each(['before', 'after'] as const)(
    'rejects a v2 and unsupported-version duplicate with the unsupported entry %s',
    (position) => {
      const supported = contractEntry();
      const unsupported = contractEntry({ workloadFingerprint: fingerprint('d') });
      unsupported.extra_info.benchmark_contract_version = 3;
      const entries = position === 'before'
        ? [unsupported, supported]
        : [supported, unsupported];

      expect(() => parsedReport(...entries)).toThrow(
        /duplicate.*scenario.*backend.*scope/i,
      );
    },
  );

  it('rejects duplicate workload fingerprints', () => {
    expect(() => parsedReport(
      contractEntry({ scenario: 'array_mean' }),
      contractEntry({ scenario: 'array_sum' }),
    )).toThrow(/duplicate workload fingerprint/i);
  });

  it.each([
    ['non-finite mean', { mean: Number.POSITIVE_INFINITY, stddev: 0.01, rounds: 10 }],
    ['non-positive mean', { mean: 0, stddev: 0.01, rounds: 10 }],
    ['negative standard deviation', { mean: 1, stddev: -0.01, rounds: 10 }],
    ['non-integer rounds', { mean: 1, stddev: 0.01, rounds: 1.5 }],
    ['non-positive rounds', { mean: 1, stddev: 0.01, rounds: 0 }],
  ])('rejects invalid statistics: %s', (_name, stats) => {
    const entry = contractEntry();
    entry.stats = stats;

    expect(() => parsedReport(entry)).toThrow(/invalid statistics/);
  });

  it('rejects malformed pytest-benchmark documents', () => {
    expect(() => parseBenchmarkReport({})).toThrow(/pytest-benchmark/);
    expect(() => parseBenchmarkReport({ benchmarks: [] })).toThrow(/no cases/);
    expect(() => parseBenchmarkReport({ benchmarks: [{ name: 'bad' }] })).toThrow(/fullname/);
  });
});
