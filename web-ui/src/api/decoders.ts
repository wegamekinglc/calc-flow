import type { components } from './schema';

type CapabilitiesResponse = components['schemas']['CapabilitiesResponse'];
type RunResponse = components['schemas']['RunResponse'];
type ValidationReport = components['schemas']['ValidationReport'];

export class ApiContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApiContractError';
  }
}

const fail = (path: string, message: string): never => {
  throw new ApiContractError(`${path}: ${message}`);
};

const objectAt = (value: unknown, path: string): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return fail(path, 'expected an object');
  }
  return value as Record<string, unknown>;
};

const exactKeys = (
  value: Record<string, unknown>,
  keys: readonly string[],
  path: string,
): void => {
  const expected = new Set(keys);
  const extra = Object.keys(value).find((key) => !expected.has(key));
  if (extra) fail(`${path}.${extra}`, 'extra fields are forbidden');
  const missing = keys.find((key) => !(key in value));
  if (missing) fail(`${path}.${missing}`, 'field is required');
};

const stringAt = (value: unknown, path: string): string => {
  if (typeof value !== 'string') return fail(path, 'expected a string');
  return value;
};

const booleanAt = (value: unknown, path: string): boolean => {
  if (typeof value !== 'boolean') return fail(path, 'expected a boolean');
  return value;
};

const integerAt = (value: unknown, path: string, minimum = 0): number => {
  if (!Number.isInteger(value) || (value as number) < minimum) {
    return fail(path, `expected an integer greater than or equal to ${minimum}`);
  }
  return value as number;
};

const arrayAt = (value: unknown, path: string): unknown[] => {
  if (!Array.isArray(value)) return fail(path, 'expected an array');
  return value;
};

const literalAt = <T extends string>(
  value: unknown,
  allowed: readonly T[],
  path: string,
): T => {
  if (typeof value !== 'string' || !allowed.includes(value as T)) {
    return fail(path, `expected ${allowed.map((item) => `'${item}'`).join(' or ')}`);
  }
  return value as T;
};

const stringArrayAt = (
  value: unknown,
  allowed: readonly string[] | null,
  path: string,
): void => {
  arrayAt(value, path).forEach((item, index) => {
    const itemPath = `${path}[${index}]`;
    const decoded = stringAt(item, itemPath);
    if (allowed && !allowed.includes(decoded)) {
      fail(itemPath, `unsupported value '${decoded}'`);
    }
  });
};

const jsonAt = (value: unknown, path: string, depth = 0): void => {
  if (depth > 32) fail(path, 'JSON value exceeds maximum depth 32');
  if (
    value === null
    || typeof value === 'string'
    || typeof value === 'boolean'
  ) return;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) fail(path, 'JSON numbers must be finite');
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => jsonAt(item, `${path}[${index}]`, depth + 1));
    return;
  }
  const record = objectAt(value, path);
  Object.entries(record).forEach(([key, item]) => {
    jsonAt(item, `${path}.${key}`, depth + 1);
  });
};

const providerPortAt = (value: unknown, path: string): void => {
  const port = objectAt(value, path);
  exactKeys(port, ['name', 'kind', 'required'], path);
  stringAt(port.name, `${path}.name`);
  literalAt(port.kind, ['table', 'array'], `${path}.kind`);
  booleanAt(port.required, `${path}.required`);
};

const providerOptionsAt = (value: unknown, path: string): void => {
  if (value === null) return;
  const schema = objectAt(value, path);
  exactKeys(schema, ['fields', 'additionalProperties'], path);
  if (schema.additionalProperties !== false) {
    fail(`${path}.additionalProperties`, 'expected false');
  }
  arrayAt(schema.fields, `${path}.fields`).forEach((item, index) => {
    const fieldPath = `${path}.fields[${index}]`;
    const field = objectAt(item, fieldPath);
    exactKeys(field, ['name', 'valueType', 'required'], fieldPath);
    stringAt(field.name, `${fieldPath}.name`);
    literalAt(
      field.valueType,
      ['string', 'integer', 'number', 'boolean'],
      `${fieldPath}.valueType`,
    );
    booleanAt(field.required, `${fieldPath}.required`);
  });
};

const runtimeCapabilitiesAt = (value: unknown, path: string): void => {
  const runtime = objectAt(value, path);
  exactKeys(runtime, [
    'scope',
    'packageVersion',
    'projectFormatVersions',
    'batchKinds',
    'portableArrowTypes',
    'operators',
    'udfs',
    'providers',
  ], path);
  const scope = objectAt(runtime.scope, `${path}.scope`);
  exactKeys(scope, ['kind', 'sessionId', 'revision'], `${path}.scope`);
  if (scope.kind !== 'runtimeSession') fail(`${path}.scope.kind`, "expected 'runtimeSession'");
  stringAt(scope.sessionId, `${path}.scope.sessionId`);
  integerAt(scope.revision, `${path}.scope.revision`);
  stringAt(runtime.packageVersion, `${path}.packageVersion`);
  arrayAt(runtime.projectFormatVersions, `${path}.projectFormatVersions`)
    .forEach((item, index) => integerAt(item, `${path}.projectFormatVersions[${index}]`));
  stringArrayAt(runtime.batchKinds, ['table', 'array'], `${path}.batchKinds`);
  stringArrayAt(runtime.portableArrowTypes, null, `${path}.portableArrowTypes`);
  arrayAt(runtime.operators, `${path}.operators`).forEach((item, index) => {
    const itemPath = `${path}.operators[${index}]`;
    const operator = objectAt(item, itemPath);
    exactKeys(
      operator,
      ['kind', 'inputKinds', 'outputKinds', 'requiresDataFusion'],
      itemPath,
    );
    stringAt(operator.kind, `${itemPath}.kind`);
    stringArrayAt(operator.inputKinds, ['table', 'array'], `${itemPath}.inputKinds`);
    stringArrayAt(operator.outputKinds, ['table', 'array'], `${itemPath}.outputKinds`);
    booleanAt(operator.requiresDataFusion, `${itemPath}.requiresDataFusion`);
  });
  arrayAt(runtime.udfs, `${path}.udfs`).forEach((item, index) => {
    const itemPath = `${path}.udfs[${index}]`;
    const udf = objectAt(item, itemPath);
    exactKeys(
      udf,
      ['provider', 'name', 'version', 'kind', 'inputTypes', 'returnType', 'volatility'],
      itemPath,
    );
    stringAt(udf.provider, `${itemPath}.provider`);
    stringAt(udf.name, `${itemPath}.name`);
    stringAt(udf.version, `${itemPath}.version`);
    if (udf.kind !== 'data_fusion_scalar') {
      fail(`${itemPath}.kind`, "expected 'data_fusion_scalar'");
    }
    stringArrayAt(udf.inputTypes, null, `${itemPath}.inputTypes`);
    stringAt(udf.returnType, `${itemPath}.returnType`);
    stringAt(udf.volatility, `${itemPath}.volatility`);
  });
  arrayAt(runtime.providers, `${path}.providers`).forEach((item, index) => {
    const itemPath = `${path}.providers[${index}]`;
    const provider = objectAt(item, itemPath);
    exactKeys(
      provider,
      ['provider', 'name', 'version', 'inputPorts', 'outputPorts', 'optionsSchema'],
      itemPath,
    );
    stringAt(provider.provider, `${itemPath}.provider`);
    stringAt(provider.name, `${itemPath}.name`);
    stringAt(provider.version, `${itemPath}.version`);
    arrayAt(provider.inputPorts, `${itemPath}.inputPorts`)
      .forEach((port, portIndex) => providerPortAt(port, `${itemPath}.inputPorts[${portIndex}]`));
    arrayAt(provider.outputPorts, `${itemPath}.outputPorts`)
      .forEach((port, portIndex) => providerPortAt(port, `${itemPath}.outputPorts[${portIndex}]`));
    providerOptionsAt(provider.optionsSchema, `${itemPath}.optionsSchema`);
  });
};

const previewCapabilitiesAt = (value: unknown, path: string): void => {
  const preview = objectAt(value, path);
  exactKeys(preview, [
    'inputBatchKinds',
    'requestInputFormats',
    'projectInputFormats',
    'workerRegistrations',
    'limits',
  ], path);
  stringArrayAt(preview.inputBatchKinds, ['table', 'array'], `${path}.inputBatchKinds`);
  stringArrayAt(
    preview.requestInputFormats,
    ['arrow_ipc', 'columns', 'records'],
    `${path}.requestInputFormats`,
  );
  stringArrayAt(
    preview.projectInputFormats,
    ['arrow_ipc', 'csv', 'inline_json', 'json'],
    `${path}.projectInputFormats`,
  );
  arrayAt(preview.workerRegistrations, `${path}.workerRegistrations`)
    .forEach((item, index) => {
      const itemPath = `${path}.workerRegistrations[${index}]`;
      const registration = objectAt(item, itemPath);
      const reconstruction = literalAt(
        registration.reconstruction,
        ['serialized', 'lazyBuiltin', 'unavailable'],
        `${itemPath}.reconstruction`,
      );
      const keys = ['reconstruction', 'registrationKind', 'provider', 'name', 'version'];
      if (reconstruction === 'unavailable') keys.push('reasonCode');
      exactKeys(registration, keys, itemPath);
      literalAt(
        registration.registrationKind,
        ['provider', 'dataFusionScalar'],
        `${itemPath}.registrationKind`,
      );
      stringAt(registration.provider, `${itemPath}.provider`);
      stringAt(registration.name, `${itemPath}.name`);
      stringAt(registration.version, `${itemPath}.version`);
      if (reconstruction === 'unavailable' && registration.reasonCode !== 'serializationFailed') {
        fail(`${itemPath}.reasonCode`, "expected 'serializationFailed'");
      }
    });
  const limits = objectAt(preview.limits, `${path}.limits`);
  const limitNames = [
    'maxInputBytes',
    'maxRows',
    'timeoutSeconds',
    'memoryLimitMb',
    'outputRows',
  ];
  exactKeys(limits, limitNames, `${path}.limits`);
  limitNames.forEach((name) => {
    const itemPath = `${path}.limits.${name}`;
    const limit = objectAt(limits[name], itemPath);
    exactKeys(limit, ['default', 'minimum', 'maximum'], itemPath);
    integerAt(limit.default, `${itemPath}.default`);
    integerAt(limit.minimum, `${itemPath}.minimum`);
    integerAt(limit.maximum, `${itemPath}.maximum`);
  });
};

export const decodeCapabilitiesResponse = (value: unknown): CapabilitiesResponse => {
  const root = objectAt(value, 'capabilities');
  if (root.schemaVersion !== 1) {
    throw new ApiContractError(
      `capabilities schema version ${String(root.schemaVersion)} is unsupported; expected 1`,
    );
  }
  exactKeys(root, ['schemaVersion', 'runtime', 'preview'], 'capabilities');
  runtimeCapabilitiesAt(root.runtime, 'capabilities.runtime');
  previewCapabilitiesAt(root.preview, 'capabilities.preview');
  return value as CapabilitiesResponse;
};

const validationIssueAt = (value: unknown, path: string): void => {
  const issue = objectAt(value, path);
  exactKeys(issue, ['path', 'code', 'message'], path);
  stringAt(issue.path, `${path}.path`);
  stringAt(issue.code, `${path}.code`);
  stringAt(issue.message, `${path}.message`);
};

export const decodeValidationReport = (value: unknown): ValidationReport => {
  const report = objectAt(value, 'validation');
  exactKeys(report, ['kind', 'valid', 'issues', 'fingerprint'], 'validation');
  const kind = literalAt(report.kind, ['valid', 'invalid'], 'validation.kind');
  const issues = arrayAt(report.issues, 'validation.issues');
  issues.forEach((issue, index) => validationIssueAt(issue, `validation.issues[${index}]`));
  if (kind === 'valid') {
    if (report.valid !== true) fail('validation.valid', 'expected true');
    if (issues.length !== 0) fail('validation.issues', 'valid reports require no issues');
    if (!stringAt(report.fingerprint, 'validation.fingerprint')) {
      fail('validation.fingerprint', 'valid reports require a fingerprint');
    }
  } else {
    if (report.valid !== false) fail('validation.valid', 'expected false');
    if (issues.length === 0) {
      fail('validation.issues', 'invalid reports require at least one issue');
    }
    if (report.fingerprint !== null) fail('validation.fingerprint', 'expected null');
  }
  return value as ValidationReport;
};

const rowCountsAt = (value: unknown, path: string): void => {
  const counts = objectAt(value, path);
  Object.entries(counts).forEach(([key, count]) => integerAt(count, `${path}.${key}`));
};

const resultAt = (value: unknown, path: string): void => {
  const result = objectAt(value, path);
  exactKeys(result, ['outputs', 'node_timings', 'datafusion_metrics', 'metadata'], path);
  const outputs = objectAt(result.outputs, `${path}.outputs`);
  Object.entries(outputs).forEach(([name, rawOutput]) => {
    const itemPath = `${path}.outputs.${name}`;
    const output = objectAt(rawOutput, itemPath);
    const kind = output.kind;
    if (kind !== 'table' && kind !== 'array') {
      throw new ApiContractError(
        `run result output '${name}' has unsupported kind '${String(kind)}'; `
        + "expected 'table' or 'array'",
      );
    }
    if (kind === 'table') {
      exactKeys(
        output,
        ['kind', 'total_rows', 'truncated', 'schema', 'rows', 'metadata'],
        itemPath,
      );
      arrayAt(output.schema, `${itemPath}.schema`).forEach((field, index) => {
        const fieldPath = `${itemPath}.schema[${index}]`;
        const decoded = objectAt(field, fieldPath);
        exactKeys(decoded, ['name', 'type', 'nullable'], fieldPath);
        stringAt(decoded.name, `${fieldPath}.name`);
        stringAt(decoded.type, `${fieldPath}.type`);
        booleanAt(decoded.nullable, `${fieldPath}.nullable`);
      });
      arrayAt(output.rows, `${itemPath}.rows`).forEach((row, index) => {
        const decoded = objectAt(row, `${itemPath}.rows[${index}]`);
        Object.entries(decoded).forEach(([key, item]) => {
          jsonAt(item, `${itemPath}.rows[${index}].${key}`);
        });
      });
    } else {
      exactKeys(
        output,
        ['kind', 'backend', 'total_rows', 'truncated', 'data', 'metadata'],
        itemPath,
      );
      stringAt(output.backend, `${itemPath}.backend`);
      jsonAt(output.data, `${itemPath}.data`);
    }
    integerAt(output.total_rows, `${itemPath}.total_rows`);
    booleanAt(output.truncated, `${itemPath}.truncated`);
    const metadata = objectAt(output.metadata, `${itemPath}.metadata`);
    Object.entries(metadata).forEach(([key, item]) => {
      jsonAt(item, `${itemPath}.metadata.${key}`);
    });
  });
  const timings = objectAt(result.node_timings, `${path}.node_timings`);
  Object.entries(timings).forEach(([name, rawTiming]) => {
    const itemPath = `${path}.node_timings.${name}`;
    const timing = objectAt(rawTiming, itemPath);
    exactKeys(timing, ['duration_ns', 'input_rows', 'output_rows'], itemPath);
    integerAt(timing.duration_ns, `${itemPath}.duration_ns`);
    rowCountsAt(timing.input_rows, `${itemPath}.input_rows`);
    rowCountsAt(timing.output_rows, `${itemPath}.output_rows`);
  });
  arrayAt(result.datafusion_metrics, `${path}.datafusion_metrics`)
    .forEach((rawMetric, index) => {
      const itemPath = `${path}.datafusion_metrics[${index}]`;
      const metric = objectAt(rawMetric, itemPath);
      exactKeys(metric, [
        'query_id',
        'node_id',
        'planning_ns',
        'execution_ns',
        'output_rows',
        'logical_plan',
        'physical_plan',
      ], itemPath);
      integerAt(metric.query_id, `${itemPath}.query_id`);
      if (metric.node_id !== null) stringAt(metric.node_id, `${itemPath}.node_id`);
      integerAt(metric.planning_ns, `${itemPath}.planning_ns`);
      integerAt(metric.execution_ns, `${itemPath}.execution_ns`);
      integerAt(metric.output_rows, `${itemPath}.output_rows`);
      stringAt(metric.logical_plan, `${itemPath}.logical_plan`);
      stringAt(metric.physical_plan, `${itemPath}.physical_plan`);
    });
  const metadata = objectAt(result.metadata, `${path}.metadata`);
  Object.entries(metadata).forEach(([key, item]) => jsonAt(item, `${path}.metadata.${key}`));
};

const nullableStringAt = (value: unknown, path: string): void => {
  if (value !== null) stringAt(value, path);
};

export const decodeRunResponse = (value: unknown): RunResponse => {
  const run = objectAt(value, 'run');
  exactKeys(run, [
    'id',
    'project_id',
    'status',
    'created_at',
    'started_at',
    'finished_at',
    'error',
    'result',
  ], 'run');
  stringAt(run.id, 'run.id');
  stringAt(run.project_id, 'run.project_id');
  stringAt(run.created_at, 'run.created_at');
  const status = literalAt(
    run.status,
    ['pending', 'running', 'completed', 'failed', 'timed_out', 'cancelled'],
    'run.status',
  );
  nullableStringAt(run.started_at, 'run.started_at');
  nullableStringAt(run.finished_at, 'run.finished_at');
  nullableStringAt(run.error, 'run.error');
  if (status === 'pending') {
    if (run.started_at !== null || run.finished_at !== null) {
      fail('run', 'pending runs cannot have start or finish times');
    }
    if (run.error !== null || run.result !== null) fail('run', 'pending run payload is inconsistent');
  } else if (status === 'running') {
    stringAt(run.started_at, 'run.started_at');
    if (run.finished_at !== null || run.error !== null || run.result !== null) {
      fail('run', 'running run payload is inconsistent');
    }
  } else if (status === 'completed') {
    stringAt(run.started_at, 'run.started_at');
    stringAt(run.finished_at, 'run.finished_at');
    if (run.error !== null) fail('run.error', 'completed runs require null');
    resultAt(run.result, 'run.result');
  } else if (status === 'failed' || status === 'timed_out') {
    stringAt(run.started_at, 'run.started_at');
    stringAt(run.finished_at, 'run.finished_at');
    if (!stringAt(run.error, 'run.error')) fail('run.error', 'must not be empty');
    if (run.result !== null) fail('run.result', 'failed runs require null');
  } else {
    stringAt(run.finished_at, 'run.finished_at');
    if (run.error !== null || run.result !== null) {
      fail('run', 'cancelled run payload is inconsistent');
    }
  }
  return value as RunResponse;
};
