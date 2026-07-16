export type CompatibilityStatus = 'compatible' | 'incompatible' | 'unverified';

export interface BenchmarkCompatibilityIssue {
  code:
    | 'missing_contract_metadata'
    | 'contract_version_mismatch'
    | 'machine_mismatch'
    | 'dependency_mismatch'
    | 'scale_mismatch'
    | 'scope_mismatch'
    | 'workload_mismatch'
    | 'dtype_mismatch'
    | 'backend_configuration_mismatch';
  field: string;
  baseline: unknown;
  current: unknown;
}

interface BenchmarkStats {
  mean: number;
  stddev: number;
  rounds: number;
}

type BenchmarkBackend = 'numpy' | 'jax';
type BenchmarkScope =
  | 'backend_kernel'
  | 'provider_boundary'
  | 'plan_end_to_end'
  | 'batch_ownership';

interface ContractIdentity {
  version: number;
  scenario: string | null;
  backend: string | null;
  scope: string | null;
}

interface BenchmarkContractV2 extends ContractIdentity {
  version: 2;
  scenario: string;
  backend: BenchmarkBackend;
  scope: BenchmarkScope;
  workloadVersion: number;
  scale: string;
  tableRows: number;
  arrayElements: number;
  matrixDimension: number;
  inputRows: number;
  outputRows: number;
  expression: string;
  inputDtype: string;
  outputDtype: string;
  machineFingerprint: string;
  dependencyFingerprint: string;
  workloadFingerprint: string;
  jaxPlatform: string | null;
  jaxEnableX64: boolean | null;
}

interface BenchmarkEntry {
  name: string;
  fullname: string;
  group?: string;
  stats: BenchmarkStats;
  extraInfo?: Record<string, unknown>;
  contract: ContractIdentity | BenchmarkContractV2 | null;
}

export interface BenchmarkReport {
  benchmarks: BenchmarkEntry[];
  machine_info?: Record<string, unknown>;
  commit_info?: Record<string, unknown>;
}

export interface BenchmarkComparisonRow {
  key: string;
  scenario: string;
  backend: string | null;
  scale: string | null;
  baselineMean: number;
  currentMean: number;
  deltaPercent: number;
  baselineCovPercent: number;
  currentCovPercent: number;
  status: 'regression' | 'improvement' | 'stable' | 'noisy';
}

export interface BenchmarkComparisonResult {
  status: CompatibilityStatus;
  rows: BenchmarkComparisonRow[];
  issues: BenchmarkCompatibilityIssue[];
}

const SUPPORTED_SCOPES = new Set<BenchmarkScope>([
  'backend_kernel',
  'provider_boundary',
  'plan_end_to_end',
  'batch_ownership',
]);
const SUPPORTED_BACKENDS = new Set<BenchmarkBackend>(['numpy', 'jax']);
const JAX_ONLY_FIELDS = ['jax_version', 'jaxlib_version', 'jax_platform', 'jax_enable_x64'] as const;
const FINGERPRINT = /^[0-9a-f]{64}$/;
const ISSUE_ORDER: BenchmarkCompatibilityIssue['code'][] = [
  'missing_contract_metadata',
  'contract_version_mismatch',
  'machine_mismatch',
  'dependency_mismatch',
  'scale_mismatch',
  'scope_mismatch',
  'workload_mismatch',
  'dtype_mismatch',
  'backend_configuration_mismatch',
];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const finiteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const contractError = (name: string, field: string): never => {
  throw new Error(`Benchmark ${name} has invalid contract-v2 metadata: ${field}`);
};

const nonEmptyText = (
  value: unknown,
  name: string,
  field: string,
): string => {
  if (typeof value !== 'string' || value.trim().length === 0) contractError(name, field);
  return value as string;
};

const integer = (
  value: unknown,
  name: string,
  field: string,
  minimum: number,
): number => {
  if (!finiteNumber(value) || !Number.isInteger(value) || value < minimum) {
    contractError(name, field);
  }
  return value as number;
};

const identityObject = (
  value: unknown,
  name: string,
  field: string,
): Record<string, unknown> => {
  if (!isRecord(value)) contractError(name, field);
  return value as Record<string, unknown>;
};

const fingerprint = (value: unknown, name: string, field: string): string => {
  if (typeof value !== 'string' || !FINGERPRINT.test(value)) contractError(name, field);
  return value as string;
};

const optionalIdentityText = (value: unknown): string | null =>
  typeof value === 'string' && value.trim().length > 0 ? value : null;

const parseContractV2 = (
  name: string,
  info: Record<string, unknown>,
): BenchmarkContractV2 => {
  const scenario = nonEmptyText(info.scenario, name, 'scenario');
  const backendText = nonEmptyText(info.backend, name, 'backend');
  if (!SUPPORTED_BACKENDS.has(backendText as BenchmarkBackend)) contractError(name, 'backend');
  const backend = backendText as BenchmarkBackend;
  const scopeText = nonEmptyText(info.scope, name, 'scope');
  if (!SUPPORTED_SCOPES.has(scopeText as BenchmarkScope)) contractError(name, 'scope');
  const scope = scopeText as BenchmarkScope;
  const backendConfiguration = identityObject(
    info.backend_configuration,
    name,
    'backend_configuration',
  );
  identityObject(info.machine_identity, name, 'machine_identity');
  identityObject(info.dependency_identity, name, 'dependency_identity');
  identityObject(info.workload_identity, name, 'workload_identity');
  nonEmptyText(info.python_version, name, 'python_version');
  nonEmptyText(info.numpy_version, name, 'numpy_version');
  integer(info.process_rss_bytes, name, 'process_rss_bytes', 0);

  let jaxPlatform: string | null = null;
  let jaxEnableX64: boolean | null = null;
  if (backend === 'jax') {
    nonEmptyText(info.jax_version, name, 'jax_version');
    nonEmptyText(info.jaxlib_version, name, 'jaxlib_version');
    jaxPlatform = nonEmptyText(info.jax_platform, name, 'jax_platform');
    if (typeof info.jax_enable_x64 !== 'boolean') contractError(name, 'jax_enable_x64');
    jaxEnableX64 = info.jax_enable_x64 as boolean;
    nonEmptyText(backendConfiguration.jax_platform, name, 'backend_configuration.jax_platform');
    if (typeof backendConfiguration.jax_enable_x64 !== 'boolean') {
      contractError(name, 'backend_configuration.jax_enable_x64');
    }
  } else {
    const hasJaxOnlyField = JAX_ONLY_FIELDS.some((field) => field in info || field in backendConfiguration);
    if (hasJaxOnlyField) {
      throw new Error(`Benchmark ${name} has invalid contract-v2 metadata: NumPy entries cannot contain JAX fields`);
    }
  }

  return {
    version: 2,
    scenario,
    backend,
    scope,
    workloadVersion: integer(info.workload_version, name, 'workload_version', 1),
    scale: nonEmptyText(info.scale, name, 'scale'),
    tableRows: integer(info.table_rows, name, 'table_rows', 1),
    arrayElements: integer(info.array_elements, name, 'array_elements', 1),
    matrixDimension: integer(info.matrix_dimension, name, 'matrix_dimension', 1),
    inputRows: integer(info.input_rows, name, 'input_rows', 0),
    outputRows: integer(info.output_rows, name, 'output_rows', 0),
    expression: nonEmptyText(info.expression, name, 'expression'),
    inputDtype: nonEmptyText(info.input_dtype, name, 'input_dtype'),
    outputDtype: nonEmptyText(info.output_dtype, name, 'output_dtype'),
    machineFingerprint: fingerprint(info.machine_fingerprint, name, 'machine_fingerprint'),
    dependencyFingerprint: fingerprint(info.dependency_fingerprint, name, 'dependency_fingerprint'),
    workloadFingerprint: fingerprint(info.workload_fingerprint, name, 'workload_fingerprint'),
    jaxPlatform,
    jaxEnableX64,
  };
};

const parseContract = (
  name: string,
  info: Record<string, unknown> | undefined,
): ContractIdentity | BenchmarkContractV2 | null => {
  if (!info || info.benchmark_contract_version === undefined) return null;
  const version = integer(
    info.benchmark_contract_version,
    name,
    'benchmark_contract_version',
    1,
  );
  if (version === 2) return parseContractV2(name, info);
  return {
    version,
    scenario: optionalIdentityText(info.scenario),
    backend: optionalIdentityText(info.backend),
    scope: optionalIdentityText(info.scope),
  };
};

const parseEntry = (value: unknown): BenchmarkEntry => {
  if (!isRecord(value) || typeof value.name !== 'string' || typeof value.fullname !== 'string') {
    throw new Error('Benchmark entries require name and fullname strings');
  }
  if (!isRecord(value.stats)) throw new Error(`Benchmark ${value.name} has no stats object`);
  const { mean, stddev, rounds } = value.stats;
  if (
    !finiteNumber(mean)
    || mean <= 0
    || !finiteNumber(stddev)
    || stddev < 0
    || !finiteNumber(rounds)
    || !Number.isInteger(rounds)
    || rounds <= 0
  ) {
    throw new Error(`Benchmark ${value.name} contains invalid statistics`);
  }
  const extraInfo = isRecord(value.extra_info) ? value.extra_info : undefined;
  return {
    name: value.name,
    fullname: value.fullname,
    group: typeof value.group === 'string' ? value.group : undefined,
    stats: { mean, stddev, rounds },
    extraInfo,
    contract: parseContract(value.name, extraInfo),
  };
};

const isContractV2 = (
  contract: ContractIdentity | BenchmarkContractV2 | null,
): contract is BenchmarkContractV2 => contract?.version === 2;

const rejectDuplicateContracts = (entries: BenchmarkEntry[]): void => {
  const identities = new Set<string>();
  const workloadFingerprints = new Set<string>();
  for (const entry of entries) {
    if (!isContractV2(entry.contract)) continue;
    const identity = `${entry.contract.scenario}\u0000${entry.contract.backend}\u0000${entry.contract.scope}`;
    if (identities.has(identity)) {
      throw new Error('Benchmark report contains duplicate (scenario, backend, scope) identity');
    }
    identities.add(identity);
    if (workloadFingerprints.has(entry.contract.workloadFingerprint)) {
      throw new Error('Benchmark report contains duplicate workload fingerprint');
    }
    workloadFingerprints.add(entry.contract.workloadFingerprint);
  }
};

export const parseBenchmarkReport = (value: unknown): BenchmarkReport => {
  if (!isRecord(value) || !Array.isArray(value.benchmarks)) {
    throw new Error('Expected a pytest-benchmark JSON report');
  }
  if (!value.benchmarks.length) throw new Error('Benchmark report contains no cases');
  const benchmarks = value.benchmarks.map(parseEntry);
  rejectDuplicateContracts(benchmarks);
  return {
    benchmarks,
    machine_info: isRecord(value.machine_info) ? value.machine_info : undefined,
    commit_info: isRecord(value.commit_info) ? value.commit_info : undefined,
  };
};

const coefficientOfVariation = (entry: BenchmarkEntry): number =>
  (entry.stats.stddev / entry.stats.mean) * 100;

const comparisonRow = (
  baseline: BenchmarkEntry,
  current: BenchmarkEntry,
  contract: BenchmarkContractV2,
): BenchmarkComparisonRow => {
  const baselineCovPercent = coefficientOfVariation(baseline);
  const currentCovPercent = coefficientOfVariation(current);
  const deltaPercent = ((current.stats.mean / baseline.stats.mean) - 1) * 100;
  let status: BenchmarkComparisonRow['status'] = 'stable';
  if (baselineCovPercent > 5 || currentCovPercent > 5) status = 'noisy';
  else if (deltaPercent > 10) status = 'regression';
  else if (deltaPercent < -10) status = 'improvement';
  return {
    key: `${contract.scenario}\u0000${contract.backend}\u0000${contract.scope}`,
    scenario: contract.scenario,
    backend: contract.backend,
    scale: contract.scale,
    baselineMean: baseline.stats.mean,
    currentMean: current.stats.mean,
    deltaPercent,
    baselineCovPercent,
    currentCovPercent,
    status,
  };
};

const compatibilityIssue = (
  code: BenchmarkCompatibilityIssue['code'],
  field: string,
  baseline: unknown,
  current: unknown,
): BenchmarkCompatibilityIssue => ({ code, field, baseline, current });

const compareField = (
  issues: BenchmarkCompatibilityIssue[],
  code: BenchmarkCompatibilityIssue['code'],
  field: string,
  baseline: unknown,
  current: unknown,
): void => {
  if (baseline !== current) issues.push(compatibilityIssue(code, field, baseline, current));
};

const compareContracts = (
  baseline: BenchmarkEntry,
  current: BenchmarkEntry,
): { issues: BenchmarkCompatibilityIssue[]; row: BenchmarkComparisonRow | null } => {
  const baselineContract = baseline.contract;
  const currentContract = current.contract;
  if (!baselineContract || !currentContract) {
    return {
      issues: [compatibilityIssue(
        'missing_contract_metadata',
        'benchmark_contract_version',
        baselineContract?.version,
        currentContract?.version,
      )],
      row: null,
    };
  }
  if (!isContractV2(baselineContract) || !isContractV2(currentContract)) {
    return {
      issues: [compatibilityIssue(
        'contract_version_mismatch',
        'benchmark_contract_version',
        baselineContract.version,
        currentContract.version,
      )],
      row: null,
    };
  }

  const issues: BenchmarkCompatibilityIssue[] = [];
  compareField(issues, 'machine_mismatch', 'machine_fingerprint', baselineContract.machineFingerprint, currentContract.machineFingerprint);
  compareField(issues, 'dependency_mismatch', 'dependency_fingerprint', baselineContract.dependencyFingerprint, currentContract.dependencyFingerprint);
  compareField(issues, 'scale_mismatch', 'scale', baselineContract.scale, currentContract.scale);
  compareField(issues, 'scale_mismatch', 'table_rows', baselineContract.tableRows, currentContract.tableRows);
  compareField(issues, 'scale_mismatch', 'array_elements', baselineContract.arrayElements, currentContract.arrayElements);
  compareField(issues, 'scale_mismatch', 'matrix_dimension', baselineContract.matrixDimension, currentContract.matrixDimension);
  compareField(issues, 'scale_mismatch', 'input_rows', baselineContract.inputRows, currentContract.inputRows);
  compareField(issues, 'scale_mismatch', 'output_rows', baselineContract.outputRows, currentContract.outputRows);
  compareField(issues, 'scope_mismatch', 'scope', baselineContract.scope, currentContract.scope);
  compareField(issues, 'workload_mismatch', 'workload_fingerprint', baselineContract.workloadFingerprint, currentContract.workloadFingerprint);
  compareField(issues, 'dtype_mismatch', 'input_dtype', baselineContract.inputDtype, currentContract.inputDtype);
  compareField(issues, 'dtype_mismatch', 'output_dtype', baselineContract.outputDtype, currentContract.outputDtype);
  compareField(issues, 'backend_configuration_mismatch', 'jax_platform', baselineContract.jaxPlatform, currentContract.jaxPlatform);
  compareField(issues, 'backend_configuration_mismatch', 'jax_enable_x64', baselineContract.jaxEnableX64, currentContract.jaxEnableX64);

  return {
    issues,
    row: issues.length === 0 ? comparisonRow(baseline, current, currentContract) : null,
  };
};

const sameIdentity = (left: ContractIdentity, right: ContractIdentity): boolean =>
  left.scenario !== null
  && left.backend !== null
  && left.scope !== null
  && left.scenario === right.scenario
  && left.backend === right.backend
  && left.scope === right.scope;

const sameScenarioAndBackend = (left: ContractIdentity, right: ContractIdentity): boolean =>
  left.scenario !== null
  && left.backend !== null
  && left.scenario === right.scenario
  && left.backend === right.backend;

const legacyScenario = (entry: BenchmarkEntry): string | null => {
  const scenario = entry.extraInfo?.scenario;
  return typeof scenario === 'string' && scenario.length > 0 ? scenario : null;
};

const hasMatchingLegacyCase = (
  versionedEntries: BenchmarkEntry[],
  legacyEntries: BenchmarkEntry[],
): boolean => versionedEntries.some((versioned) =>
  legacyEntries.some((legacy) => legacyScenario(legacy) === versioned.contract?.scenario));

export const compareBenchmarkReports = (
  baseline: BenchmarkReport,
  current: BenchmarkReport,
): BenchmarkComparisonResult => {
  const baselineVersioned = baseline.benchmarks.filter((entry) => entry.contract !== null);
  const currentVersioned = current.benchmarks.filter((entry) => entry.contract !== null);
  const baselineLegacy = baseline.benchmarks.filter((entry) => entry.contract === null);
  const currentLegacy = current.benchmarks.filter((entry) => entry.contract === null);
  const usedCurrent = new Set<BenchmarkEntry>();
  const issues: BenchmarkCompatibilityIssue[] = [];
  const rows: BenchmarkComparisonRow[] = [];
  let matchedPairs = 0;

  for (const baselineEntry of baselineVersioned) {
    const baselineContract = baselineEntry.contract;
    if (!baselineContract) continue;
    const exact = currentVersioned.find((entry) =>
      !usedCurrent.has(entry)
      && entry.contract !== null
      && sameIdentity(baselineContract, entry.contract));
    const sameWork = currentVersioned.filter((entry) =>
      !usedCurrent.has(entry)
      && entry.contract !== null
      && sameScenarioAndBackend(baselineContract, entry.contract));
    const currentEntry = exact ?? (sameWork.length === 1 ? sameWork[0] : undefined);
    if (!currentEntry) continue;
    usedCurrent.add(currentEntry);
    matchedPairs += 1;
    const comparison = compareContracts(baselineEntry, currentEntry);
    issues.push(...comparison.issues);
    if (comparison.row) rows.push(comparison.row);
  }

  if (issues.length > 0) {
    issues.sort((left, right) => ISSUE_ORDER.indexOf(left.code) - ISSUE_ORDER.indexOf(right.code));
    return { status: 'incompatible', rows: [], issues };
  }
  if (matchedPairs > 0) {
    return {
      status: 'compatible',
      rows: rows.sort((left, right) => right.deltaPercent - left.deltaPercent),
      issues: [],
    };
  }

  const unsupportedBaseline = baselineVersioned.find((entry) => entry.contract?.version !== 2);
  const unsupportedCurrent = currentVersioned.find((entry) => entry.contract?.version !== 2);
  if (unsupportedBaseline || unsupportedCurrent) {
    return {
      status: 'incompatible',
      rows: [],
      issues: [compatibilityIssue(
        'contract_version_mismatch',
        'benchmark_contract_version',
        unsupportedBaseline?.contract?.version,
        unsupportedCurrent?.contract?.version,
      )],
    };
  }

  const onlyLegacy = baselineVersioned.length === 0 && currentVersioned.length === 0;
  const versionedMatchedLegacy = hasMatchingLegacyCase(baselineVersioned, currentLegacy)
    || hasMatchingLegacyCase(currentVersioned, baselineLegacy);
  if (onlyLegacy || versionedMatchedLegacy) {
    return {
      status: 'unverified',
      rows: [],
      issues: [compatibilityIssue(
        'missing_contract_metadata',
        'benchmark_contract_version',
        baselineVersioned[0]?.contract?.version,
        currentVersioned[0]?.contract?.version,
      )],
    };
  }
  return { status: 'unverified', rows: [], issues: [] };
};
