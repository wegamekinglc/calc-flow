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

interface MachineIdentity {
  operatingSystem: string;
  architecture: string;
  cpuBrand: string;
  logicalCpuCount: number;
  pythonImplementation: string;
}

interface DependencyIdentity {
  pythonVersion: string;
  numpyVersion: string;
  jaxVersion: string | null;
  jaxlibVersion: string | null;
}

interface BackendConfiguration {
  jaxPlatform: string | null;
  jaxEnableX64: boolean | null;
}

interface WorkloadIdentity {
  benchmarkContractVersion: 2;
  scenario: string;
  scope: BenchmarkScope;
  workloadVersion: number;
  backend: BenchmarkBackend;
  scale: string;
  tableRows: number;
  arrayElements: number;
  matrixDimension: number;
  inputRows: number;
  outputRows: number;
  expression: string;
  inputDtype: string;
  outputDtype: string;
  backendConfiguration: BackendConfiguration;
}

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
  machineIdentity: MachineIdentity;
  dependencyIdentity: DependencyIdentity;
  workloadIdentity: WorkloadIdentity;
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

const supportedBackend = (
  value: unknown,
  name: string,
  field: string,
): BenchmarkBackend => {
  const backend = nonEmptyText(value, name, field);
  if (!SUPPORTED_BACKENDS.has(backend as BenchmarkBackend)) contractError(name, field);
  return backend as BenchmarkBackend;
};

const supportedScope = (
  value: unknown,
  name: string,
  field: string,
): BenchmarkScope => {
  const scope = nonEmptyText(value, name, field);
  if (!SUPPORTED_SCOPES.has(scope as BenchmarkScope)) contractError(name, field);
  return scope as BenchmarkScope;
};

const rejectUnknownFields = (
  identity: Record<string, unknown>,
  allowedFields: readonly string[],
  name: string,
  field: string,
): void => {
  const allowed = new Set(allowedFields);
  const unknown = Object.keys(identity).find((key) => !allowed.has(key));
  if (unknown !== undefined) contractError(name, `${field}.${unknown}`);
};

const normalizeMachineIdentity = (value: unknown, name: string): MachineIdentity => {
  const identity = identityObject(value, name, 'machine_identity');
  rejectUnknownFields(
    identity,
    [
      'operating_system',
      'architecture',
      'cpu_brand',
      'logical_cpu_count',
      'python_implementation',
    ],
    name,
    'machine_identity',
  );
  return {
    operatingSystem: nonEmptyText(
      identity.operating_system,
      name,
      'machine_identity.operating_system',
    ),
    architecture: nonEmptyText(identity.architecture, name, 'machine_identity.architecture'),
    cpuBrand: nonEmptyText(identity.cpu_brand, name, 'machine_identity.cpu_brand'),
    logicalCpuCount: integer(
      identity.logical_cpu_count,
      name,
      'machine_identity.logical_cpu_count',
      1,
    ),
    pythonImplementation: nonEmptyText(
      identity.python_implementation,
      name,
      'machine_identity.python_implementation',
    ),
  };
};

const normalizeDependencyIdentity = (
  value: unknown,
  name: string,
  backend: BenchmarkBackend,
): DependencyIdentity => {
  const identity = identityObject(value, name, 'dependency_identity');
  rejectUnknownFields(
    identity,
    backend === 'jax'
      ? ['python_version', 'numpy_version', 'jax_version', 'jaxlib_version']
      : ['python_version', 'numpy_version'],
    name,
    'dependency_identity',
  );
  const pythonVersion = nonEmptyText(
    identity.python_version,
    name,
    'dependency_identity.python_version',
  );
  const numpyVersion = nonEmptyText(
    identity.numpy_version,
    name,
    'dependency_identity.numpy_version',
  );
  if (backend === 'jax') {
    return {
      pythonVersion,
      numpyVersion,
      jaxVersion: nonEmptyText(
        identity.jax_version,
        name,
        'dependency_identity.jax_version',
      ),
      jaxlibVersion: nonEmptyText(
        identity.jaxlib_version,
        name,
        'dependency_identity.jaxlib_version',
      ),
    };
  }
  return { pythonVersion, numpyVersion, jaxVersion: null, jaxlibVersion: null };
};

const normalizeBackendConfiguration = (
  value: unknown,
  name: string,
  backend: BenchmarkBackend,
  field: string,
): BackendConfiguration => {
  const configuration = identityObject(value, name, field);
  if (backend === 'jax') {
    rejectUnknownFields(
      configuration,
      ['jax_platform', 'jax_enable_x64'],
      name,
      field,
    );
    const jaxPlatform = nonEmptyText(
      configuration.jax_platform,
      name,
      `${field}.jax_platform`,
    );
    if (typeof configuration.jax_enable_x64 !== 'boolean') {
      contractError(name, `${field}.jax_enable_x64`);
    }
    return {
      jaxPlatform,
      jaxEnableX64: configuration.jax_enable_x64 as boolean,
    };
  }
  if (Object.keys(configuration).length > 0) {
    contractError(name, `${field} must be empty for NumPy`);
  }
  return { jaxPlatform: null, jaxEnableX64: null };
};

const normalizeWorkloadIdentity = (value: unknown, name: string): WorkloadIdentity => {
  const field = 'workload_identity';
  const identity = identityObject(value, name, field);
  rejectUnknownFields(
    identity,
    [
      'benchmark_contract_version',
      'scenario',
      'scope',
      'workload_version',
      'backend',
      'scale',
      'table_rows',
      'array_elements',
      'matrix_dimension',
      'input_rows',
      'output_rows',
      'expression',
      'input_dtype',
      'output_dtype',
      'backend_configuration',
    ],
    name,
    field,
  );
  const version = integer(
    identity.benchmark_contract_version,
    name,
    `${field}.benchmark_contract_version`,
    1,
  );
  if (version !== 2) contractError(name, `${field}.benchmark_contract_version`);
  const scenario = nonEmptyText(identity.scenario, name, `${field}.scenario`);
  const scope = supportedScope(identity.scope, name, `${field}.scope`);
  const workloadVersion = integer(
    identity.workload_version,
    name,
    `${field}.workload_version`,
    1,
  );
  const backend = supportedBackend(identity.backend, name, `${field}.backend`);
  return {
    benchmarkContractVersion: 2,
    scenario,
    scope,
    workloadVersion,
    backend,
    scale: nonEmptyText(identity.scale, name, `${field}.scale`),
    tableRows: integer(identity.table_rows, name, `${field}.table_rows`, 1),
    arrayElements: integer(identity.array_elements, name, `${field}.array_elements`, 1),
    matrixDimension: integer(
      identity.matrix_dimension,
      name,
      `${field}.matrix_dimension`,
      1,
    ),
    inputRows: integer(identity.input_rows, name, `${field}.input_rows`, 0),
    outputRows: integer(identity.output_rows, name, `${field}.output_rows`, 0),
    expression: nonEmptyText(identity.expression, name, `${field}.expression`),
    inputDtype: nonEmptyText(identity.input_dtype, name, `${field}.input_dtype`),
    outputDtype: nonEmptyText(identity.output_dtype, name, `${field}.output_dtype`),
    backendConfiguration: normalizeBackendConfiguration(
      identity.backend_configuration,
      name,
      backend,
      `${field}.backend_configuration`,
    ),
  };
};

const requireCoherentField = (
  name: string,
  field: string,
  flat: unknown,
  nested: unknown,
): void => {
  if (flat !== nested) contractError(name, field);
};

const parseContractV2 = (
  name: string,
  info: Record<string, unknown>,
): BenchmarkContractV2 => {
  const scenario = nonEmptyText(info.scenario, name, 'scenario');
  const backend = supportedBackend(info.backend, name, 'backend');
  const scope = supportedScope(info.scope, name, 'scope');
  const workloadVersion = integer(info.workload_version, name, 'workload_version', 1);
  const scale = nonEmptyText(info.scale, name, 'scale');
  const tableRows = integer(info.table_rows, name, 'table_rows', 1);
  const arrayElements = integer(info.array_elements, name, 'array_elements', 1);
  const matrixDimension = integer(info.matrix_dimension, name, 'matrix_dimension', 1);
  const inputRows = integer(info.input_rows, name, 'input_rows', 0);
  const outputRows = integer(info.output_rows, name, 'output_rows', 0);
  const expression = nonEmptyText(info.expression, name, 'expression');
  const inputDtype = nonEmptyText(info.input_dtype, name, 'input_dtype');
  const outputDtype = nonEmptyText(info.output_dtype, name, 'output_dtype');
  const backendConfiguration = normalizeBackendConfiguration(
    info.backend_configuration,
    name,
    backend,
    'backend_configuration',
  );
  const machineIdentity = normalizeMachineIdentity(info.machine_identity, name);
  const dependencyIdentity = normalizeDependencyIdentity(
    info.dependency_identity,
    name,
    backend,
  );
  const workloadIdentity = normalizeWorkloadIdentity(info.workload_identity, name);
  const pythonVersion = nonEmptyText(info.python_version, name, 'python_version');
  const numpyVersion = nonEmptyText(info.numpy_version, name, 'numpy_version');
  integer(info.process_rss_bytes, name, 'process_rss_bytes', 0);

  let jaxPlatform: string | null = null;
  let jaxEnableX64: boolean | null = null;
  let jaxVersion: string | null = null;
  let jaxlibVersion: string | null = null;
  if (backend === 'jax') {
    jaxVersion = nonEmptyText(info.jax_version, name, 'jax_version');
    jaxlibVersion = nonEmptyText(info.jaxlib_version, name, 'jaxlib_version');
    jaxPlatform = nonEmptyText(info.jax_platform, name, 'jax_platform');
    if (typeof info.jax_enable_x64 !== 'boolean') contractError(name, 'jax_enable_x64');
    jaxEnableX64 = info.jax_enable_x64 as boolean;
  } else {
    const hasJaxOnlyField = JAX_ONLY_FIELDS.some((field) => field in info);
    if (hasJaxOnlyField) {
      throw new Error(`Benchmark ${name} has invalid contract-v2 metadata: NumPy entries cannot contain JAX fields`);
    }
  }

  requireCoherentField(name, 'dependency_identity.python_version', pythonVersion, dependencyIdentity.pythonVersion);
  requireCoherentField(name, 'dependency_identity.numpy_version', numpyVersion, dependencyIdentity.numpyVersion);
  requireCoherentField(name, 'dependency_identity.jax_version', jaxVersion, dependencyIdentity.jaxVersion);
  requireCoherentField(name, 'dependency_identity.jaxlib_version', jaxlibVersion, dependencyIdentity.jaxlibVersion);
  requireCoherentField(name, 'backend_configuration.jax_platform', jaxPlatform, backendConfiguration.jaxPlatform);
  requireCoherentField(name, 'backend_configuration.jax_enable_x64', jaxEnableX64, backendConfiguration.jaxEnableX64);
  requireCoherentField(name, 'workload_identity.benchmark_contract_version', 2, workloadIdentity.benchmarkContractVersion);
  requireCoherentField(name, 'workload_identity.scenario', scenario, workloadIdentity.scenario);
  requireCoherentField(name, 'workload_identity.backend', backend, workloadIdentity.backend);
  requireCoherentField(name, 'workload_identity.scope', scope, workloadIdentity.scope);
  requireCoherentField(name, 'workload_identity.workload_version', workloadVersion, workloadIdentity.workloadVersion);
  requireCoherentField(name, 'workload_identity.scale', scale, workloadIdentity.scale);
  requireCoherentField(name, 'workload_identity.table_rows', tableRows, workloadIdentity.tableRows);
  requireCoherentField(name, 'workload_identity.array_elements', arrayElements, workloadIdentity.arrayElements);
  requireCoherentField(name, 'workload_identity.matrix_dimension', matrixDimension, workloadIdentity.matrixDimension);
  requireCoherentField(name, 'workload_identity.input_rows', inputRows, workloadIdentity.inputRows);
  requireCoherentField(name, 'workload_identity.output_rows', outputRows, workloadIdentity.outputRows);
  requireCoherentField(name, 'workload_identity.expression', expression, workloadIdentity.expression);
  requireCoherentField(name, 'workload_identity.input_dtype', inputDtype, workloadIdentity.inputDtype);
  requireCoherentField(name, 'workload_identity.output_dtype', outputDtype, workloadIdentity.outputDtype);
  requireCoherentField(
    name,
    'workload_identity.backend_configuration.jax_platform',
    backendConfiguration.jaxPlatform,
    workloadIdentity.backendConfiguration.jaxPlatform,
  );
  requireCoherentField(
    name,
    'workload_identity.backend_configuration.jax_enable_x64',
    backendConfiguration.jaxEnableX64,
    workloadIdentity.backendConfiguration.jaxEnableX64,
  );

  return {
    version: 2,
    scenario,
    backend,
    scope,
    workloadVersion,
    scale,
    tableRows,
    arrayElements,
    matrixDimension,
    inputRows,
    outputRows,
    expression,
    inputDtype,
    outputDtype,
    machineFingerprint: fingerprint(info.machine_fingerprint, name, 'machine_fingerprint'),
    dependencyFingerprint: fingerprint(info.dependency_fingerprint, name, 'dependency_fingerprint'),
    workloadFingerprint: fingerprint(info.workload_fingerprint, name, 'workload_fingerprint'),
    jaxPlatform,
    jaxEnableX64,
    machineIdentity,
    dependencyIdentity,
    workloadIdentity,
  };
};

const parseContract = (
  name: string,
  info: Record<string, unknown> | undefined,
): ContractIdentity | BenchmarkContractV2 | null => {
  if (info?.benchmark_contract_version === undefined) return null;
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
    const contract = entry.contract;
    if (!contract) continue;
    if (contract.scenario !== null && contract.backend !== null && contract.scope !== null) {
      const identity = `${contract.scenario}\u0000${contract.backend}\u0000${contract.scope}`;
      if (identities.has(identity)) {
        throw new Error('Benchmark report contains duplicate (scenario, backend, scope) identity');
      }
      identities.add(identity);
    }
    if (!isContractV2(contract)) continue;
    if (workloadFingerprints.has(contract.workloadFingerprint)) {
      throw new Error('Benchmark report contains duplicate workload fingerprint');
    }
    workloadFingerprints.add(contract.workloadFingerprint);
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

const sameNormalizedIdentity = (baseline: object, current: object): boolean =>
  JSON.stringify(baseline) === JSON.stringify(current);

const compareFingerprintAndIdentity = (
  issues: BenchmarkCompatibilityIssue[],
  code: BenchmarkCompatibilityIssue['code'],
  fingerprintField: string,
  baselineFingerprint: string,
  currentFingerprint: string,
  identityField: string,
  baselineIdentity: object,
  currentIdentity: object,
): void => {
  const fingerprintsMatch = baselineFingerprint === currentFingerprint;
  const identitiesMatch = sameNormalizedIdentity(baselineIdentity, currentIdentity);
  if (fingerprintsMatch && identitiesMatch) return;
  issues.push(compatibilityIssue(
    code,
    fingerprintsMatch ? identityField : fingerprintField,
    fingerprintsMatch ? baselineIdentity : baselineFingerprint,
    fingerprintsMatch ? currentIdentity : currentFingerprint,
  ));
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
  compareFingerprintAndIdentity(
    issues,
    'machine_mismatch',
    'machine_fingerprint',
    baselineContract.machineFingerprint,
    currentContract.machineFingerprint,
    'machine_identity',
    baselineContract.machineIdentity,
    currentContract.machineIdentity,
  );
  compareFingerprintAndIdentity(
    issues,
    'dependency_mismatch',
    'dependency_fingerprint',
    baselineContract.dependencyFingerprint,
    currentContract.dependencyFingerprint,
    'dependency_identity',
    baselineContract.dependencyIdentity,
    currentContract.dependencyIdentity,
  );
  const baselineWorkload = baselineContract.workloadIdentity;
  const currentWorkload = currentContract.workloadIdentity;
  compareField(issues, 'scale_mismatch', 'scale', baselineWorkload.scale, currentWorkload.scale);
  compareField(issues, 'scale_mismatch', 'table_rows', baselineWorkload.tableRows, currentWorkload.tableRows);
  compareField(issues, 'scale_mismatch', 'array_elements', baselineWorkload.arrayElements, currentWorkload.arrayElements);
  compareField(issues, 'scale_mismatch', 'matrix_dimension', baselineWorkload.matrixDimension, currentWorkload.matrixDimension);
  compareField(issues, 'scale_mismatch', 'input_rows', baselineWorkload.inputRows, currentWorkload.inputRows);
  compareField(issues, 'scale_mismatch', 'output_rows', baselineWorkload.outputRows, currentWorkload.outputRows);
  compareField(issues, 'scope_mismatch', 'scope', baselineWorkload.scope, currentWorkload.scope);
  const baselineWorkloadCore = {
    benchmarkContractVersion: baselineWorkload.benchmarkContractVersion,
    scenario: baselineWorkload.scenario,
    backend: baselineWorkload.backend,
    workloadVersion: baselineWorkload.workloadVersion,
    expression: baselineWorkload.expression,
  };
  const currentWorkloadCore = {
    benchmarkContractVersion: currentWorkload.benchmarkContractVersion,
    scenario: currentWorkload.scenario,
    backend: currentWorkload.backend,
    workloadVersion: currentWorkload.workloadVersion,
    expression: currentWorkload.expression,
  };
  const workloadFingerprintsMatch =
    baselineContract.workloadFingerprint === currentContract.workloadFingerprint;
  const workloadDocumentsMatch = sameNormalizedIdentity(baselineWorkload, currentWorkload);
  const workloadCoreMatches = sameNormalizedIdentity(
    baselineWorkloadCore,
    currentWorkloadCore,
  );
  const categorizedWorkloadDifference =
    baselineWorkload.scale !== currentWorkload.scale
    || baselineWorkload.tableRows !== currentWorkload.tableRows
    || baselineWorkload.arrayElements !== currentWorkload.arrayElements
    || baselineWorkload.matrixDimension !== currentWorkload.matrixDimension
    || baselineWorkload.inputRows !== currentWorkload.inputRows
    || baselineWorkload.outputRows !== currentWorkload.outputRows
    || baselineWorkload.scope !== currentWorkload.scope
    || baselineWorkload.inputDtype !== currentWorkload.inputDtype
    || baselineWorkload.outputDtype !== currentWorkload.outputDtype
    || !sameNormalizedIdentity(
      baselineWorkload.backendConfiguration,
      currentWorkload.backendConfiguration,
    );
  if (
    !workloadFingerprintsMatch
    || !workloadCoreMatches
    || (!workloadDocumentsMatch && !categorizedWorkloadDifference)
  ) {
    issues.push(compatibilityIssue(
      'workload_mismatch',
      workloadFingerprintsMatch ? 'workload_identity' : 'workload_fingerprint',
      workloadFingerprintsMatch ? baselineWorkload : baselineContract.workloadFingerprint,
      workloadFingerprintsMatch ? currentWorkload : currentContract.workloadFingerprint,
    ));
  }
  compareField(issues, 'dtype_mismatch', 'input_dtype', baselineWorkload.inputDtype, currentWorkload.inputDtype);
  compareField(issues, 'dtype_mismatch', 'output_dtype', baselineWorkload.outputDtype, currentWorkload.outputDtype);
  compareField(
    issues,
    'backend_configuration_mismatch',
    'jax_platform',
    baselineWorkload.backendConfiguration.jaxPlatform,
    currentWorkload.backendConfiguration.jaxPlatform,
  );
  compareField(
    issues,
    'backend_configuration_mismatch',
    'jax_enable_x64',
    baselineWorkload.backendConfiguration.jaxEnableX64,
    currentWorkload.backendConfiguration.jaxEnableX64,
  );

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
  if (matchedPairs > 0) {
    return {
      status: 'compatible',
      rows: rows.sort((left, right) => right.deltaPercent - left.deltaPercent),
      issues: [],
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
