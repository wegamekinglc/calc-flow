import type { components } from './schema';

type CapabilitiesResponse = components['schemas']['CapabilitiesResponse'];
type JobResponse = components['schemas']['JobResponse'];
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

const capabilityLifecycleAt = (
  value: Record<string, unknown>,
  path: string,
  fields: {
    modes: unknown;
    finality: unknown;
    stateful: unknown;
    microbatchInvariant: unknown;
    requiresWatermark: unknown;
    checkpointSupport: unknown;
    stateVersion: unknown;
    deterministic: unknown;
    replaySafe: unknown;
  },
): void => {
  const modes = arrayAt(fields.modes, `${path}.modes`);
  stringArrayAt(modes, ['batch', 'stream'], `${path}.modes`);
  if (modes.length === 0) {
    fail(`${path}.modes`, 'expected at least one execution mode');
  }
  literalAt(
    fields.finality,
    ['per_row_final', 'group_final_append_only', 'unproven'],
    `${path}.finality`,
  );
  booleanAt(fields.stateful, `${path}.stateful`);
  booleanAt(fields.microbatchInvariant, `${path}.microbatchInvariant`);
  booleanAt(fields.requiresWatermark, `${path}.requiresWatermark`);
  literalAt(
    fields.checkpointSupport,
    ['stateless', 'checkpointed_stateful', 'unproven'],
    `${path}.checkpointSupport`,
  );
  if (fields.stateVersion === null) {
    if (fields.checkpointSupport === 'checkpointed_stateful') {
      fail(`${path}.stateVersion`, 'must be a positive integer when checkpointed_stateful');
    }
  } else {
    integerAt(fields.stateVersion, `${path}.stateVersion`);
    if (
      typeof fields.stateVersion === 'number'
      && fields.stateVersion < 1
    ) {
      fail(`${path}.stateVersion`, 'expected a positive state layout version');
    }
    if (fields.checkpointSupport !== 'checkpointed_stateful') {
      fail(
        `${path}.stateVersion`,
        'must be null unless checkpointSupport is checkpointed_stateful',
      );
    }
  }
  if (fields.checkpointSupport === 'stateless' && fields.stateful === true) {
    fail(`${path}.stateful`, 'stateless capability must set stateful=false');
  }
  booleanAt(fields.deterministic, `${path}.deterministic`);
  booleanAt(fields.replaySafe, `${path}.replaySafe`);
};

const operatorStateLayoutsAt = (
  layouts: unknown,
  checkpointSupport: unknown,
  stateVersion: unknown,
  path: string,
): void => {
  const items = arrayAt(layouts, `${path}.stateLayouts`);
  if (checkpointSupport !== 'checkpointed_stateful') {
    if (items.length > 0) {
      fail(`${path}.stateLayouts`, 'must be empty unless checkpointSupport is checkpointed_stateful');
    }
    return;
  }
  if (items.length === 0) {
    fail(`${path}.stateLayouts`, 'checkpointed_stateful requires at least one state layout');
  }
  let previousLayout: unknown = null;
  items.forEach((layout) => {
    integerAt(layout, `${path}.stateLayouts`);
    if (typeof layout === 'number' && layout < 1) {
      fail(`${path}.stateLayouts`, 'expected a positive state layout');
    }
    if (
      typeof previousLayout === 'number'
      && typeof layout === 'number'
      && previousLayout >= layout
    ) {
      fail(`${path}.stateLayouts`, 'must be strictly ascending without duplicates');
    }
    previousLayout = layout;
  });
  if (typeof stateVersion === 'number' && !items.includes(stateVersion)) {
    fail(`${path}.stateLayouts`, 'must contain stateVersion');
  }
};

const CLOSED_CAPABILITY_RULES: ReadonlySet<string> = new Set([
  'array_api_safe_dtype@1',
  'elementwise_broadcast@1',
  'feature_axis_reduction@1',
  'table_matmul_static_rhs@1',
]);

const capabilityRuleAt = (value: unknown, path: string): void => {
  const rule = objectAt(value, path);
  exactKeys(rule, ['name', 'version'], path);
  stringAt(rule.name, `${path}.name`);
  stringAt(rule.version, `${path}.version`);
  if (!CLOSED_CAPABILITY_RULES.has(`${String(rule.name)}@${String(rule.version)}`)) {
    fail(path, `unknown capability rule ${String(rule.name)}@${String(rule.version)}`);
  }
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
    'connectors',
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
      [
        'kind',
        'version',
        'inputPorts',
        'outputPorts',
        'modes',
        'finality',
        'requiresDatafusion',
        'stateful',
        'microbatchInvariant',
        'requiresWatermark',
        'checkpointSupport',
        'stateVersion',
        'deterministic',
        'replaySafe',
        'stateLayouts',
      ],
      itemPath,
    );
    stringAt(operator.kind, `${itemPath}.kind`);
    stringAt(operator.version, `${itemPath}.version`);
    arrayAt(operator.inputPorts, `${itemPath}.inputPorts`)
      .forEach((port, portIndex) => providerPortAt(port, `${itemPath}.inputPorts[${portIndex}]`));
    arrayAt(operator.outputPorts, `${itemPath}.outputPorts`)
      .forEach((port, portIndex) => providerPortAt(port, `${itemPath}.outputPorts[${portIndex}]`));
    booleanAt(operator.requiresDatafusion, `${itemPath}.requiresDatafusion`);
    capabilityLifecycleAt(operator, itemPath, {
      modes: operator.modes,
      finality: operator.finality,
      stateful: operator.stateful,
      microbatchInvariant: operator.microbatchInvariant,
      requiresWatermark: operator.requiresWatermark,
      checkpointSupport: operator.checkpointSupport,
      stateVersion: operator.stateVersion,
      deterministic: operator.deterministic,
      replaySafe: operator.replaySafe,
    });
    operatorStateLayoutsAt(
      operator.stateLayouts,
      operator.checkpointSupport,
      operator.stateVersion,
      itemPath,
    );
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
      [
        'provider',
        'name',
        'version',
        'inputPorts',
        'outputPorts',
        'optionsSchema',
        'modes',
        'finality',
        'stateful',
        'microbatchInvariant',
        'requiresWatermark',
        'checkpointSupport',
        'stateVersion',
        'deterministic',
        'replaySafe',
        'supportsStaticInputs',
        'partitionContract',
        'arrayRules',
      ],
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
    capabilityLifecycleAt(provider, itemPath, {
      modes: provider.modes,
      finality: provider.finality,
      stateful: provider.stateful,
      microbatchInvariant: provider.microbatchInvariant,
      requiresWatermark: provider.requiresWatermark,
      checkpointSupport: provider.checkpointSupport,
      stateVersion: provider.stateVersion,
      deterministic: provider.deterministic,
      replaySafe: provider.replaySafe,
    });
    booleanAt(provider.supportsStaticInputs, `${itemPath}.supportsStaticInputs`);
    literalAt(
      provider.partitionContract,
      ['none', 'row_axis_independent'],
      `${itemPath}.partitionContract`,
    );
    if (provider.arrayRules === null) return;
    const rulesPath = `${itemPath}.arrayRules`;
    const rules = objectAt(provider.arrayRules, rulesPath);
    exactKeys(rules, ['supportedDtypes', 'safeDtypeRule', 'shapeRules'], rulesPath);
    stringArrayAt(rules.supportedDtypes, null, `${rulesPath}.supportedDtypes`);
    capabilityRuleAt(rules.safeDtypeRule, `${rulesPath}.safeDtypeRule`);
    arrayAt(rules.shapeRules, `${rulesPath}.shapeRules`)
      .forEach((rule, ruleIndex) => {
        capabilityRuleAt(rule, `${rulesPath}.shapeRules[${ruleIndex}]`);
      });
  });
  arrayAt(runtime.connectors, `${path}.connectors`).forEach((item, index) => {
    const itemPath = `${path}.connectors[${index}]`;
    const connector = objectAt(item, itemPath);
    exactKeys(
      connector,
      ['provider', 'name', 'version', 'kind', 'capabilities', 'formats', 'optionsSchema'],
      itemPath,
    );
    stringAt(connector.provider, `${itemPath}.provider`);
    stringAt(connector.name, `${itemPath}.name`);
    stringAt(connector.version, `${itemPath}.version`);
    literalAt(connector.kind, ['source', 'sink', 'both'], `${itemPath}.kind`);
    stringArrayAt(connector.formats, null, `${itemPath}.formats`);
    const axes = objectAt(connector.capabilities, `${itemPath}.capabilities`);
    exactKeys(
      axes,
      ['delivery', 'replay', 'watermark', 'transaction', 'snapshot', 'polling', 'cdc', 'lookup'],
      `${itemPath}.capabilities`,
    );
    literalAt(
      axes.delivery,
      ['best_effort', 'at_least_once', 'exactly_once'],
      `${itemPath}.capabilities.delivery`,
    );
    literalAt(
      axes.replay,
      ['replayable_exact', 'unreplayable'],
      `${itemPath}.capabilities.replay`,
    );
    literalAt(
      axes.watermark,
      ['native', 'generated_only'],
      `${itemPath}.capabilities.watermark`,
    );
    literalAt(
      axes.transaction,
      ['none', 'pre_commit_commit', 'ledger_idempotent', 'retry_deduplicated'],
      `${itemPath}.capabilities.transaction`,
    );
    booleanAt(axes.snapshot, `${itemPath}.capabilities.snapshot`);
    booleanAt(axes.polling, `${itemPath}.capabilities.polling`);
    booleanAt(axes.cdc, `${itemPath}.capabilities.cdc`);
    booleanAt(axes.lookup, `${itemPath}.capabilities.lookup`);
    const options = objectAt(connector.optionsSchema, `${itemPath}.optionsSchema`);
    Object.entries(options).forEach(([name, option]) => {
      jsonAt(option, `${itemPath}.optionsSchema.${name}`);
    });
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
  if (root.schemaVersion !== 3) {
    throw new ApiContractError(
      `capabilities schema version ${String(root.schemaVersion)} is unsupported; expected 3`,
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
        'runtime_acquire_ns',
        'session_state_create_ns',
        'input_adapter_ns',
        'table_register_ns',
        'sql_parse_ns',
        'logical_planning_ns',
        'physical_planning_ns',
        'physical_planning_count',
        'planning_ns',
        'stream_open_ns',
        'execution_to_first_batch_ns',
        'execution_remaining_ns',
        'execution_ns',
        'collect_ns',
        'output_arrow_wrap_ns',
        'audit_ns',
        'metrics_traversal_ns',
        'logical_plan_string_ns',
        'physical_plan_string_ns',
        'batch_envelope_ns',
        'run_result_ns',
        'physical_metric_count',
        'output_partition_count',
        'output_partition_rows',
        'window_partition_count',
        'window_partition_rows',
        'spill_bytes',
        'elapsed_compute_ns',
        'window_compute_ns',
        'repartition_sort_compute_ns',
        'window_operator_count',
        'repartition_operator_count',
        'sort_operator_count',
        'coalesce_operator_count',
        'output_rows',
        'configured_batch_size',
        'parallelism_mode',
        'configured_target_partitions',
        'requested_target_partitions',
        'effective_target_partitions',
        'available_parallelism',
        'max_partitions',
        'min_rows_per_partition',
        'small_rows_threshold',
        'parallelism_decision_reused',
        'decision_input_rows',
        'decision_active_entities',
        'decision_active_entities_source',
        'input_rows',
        'active_entities',
        'active_entities_source',
        'partition_limit_reason',
        'rolling_rewrite_enabled',
        'diagnostics_collected',
        'rolling_candidate_windows',
        'rolling_rewritten_windows',
        'rolling_fallback_reasons',
        'logical_plan',
        'physical_plan',
      ], itemPath);
      integerAt(metric.query_id, `${itemPath}.query_id`);
      if (metric.node_id !== null) stringAt(metric.node_id, `${itemPath}.node_id`);
      integerAt(metric.runtime_acquire_ns, `${itemPath}.runtime_acquire_ns`);
      integerAt(metric.session_state_create_ns, `${itemPath}.session_state_create_ns`);
      integerAt(metric.input_adapter_ns, `${itemPath}.input_adapter_ns`);
      integerAt(metric.table_register_ns, `${itemPath}.table_register_ns`);
      integerAt(metric.sql_parse_ns, `${itemPath}.sql_parse_ns`);
      integerAt(metric.logical_planning_ns, `${itemPath}.logical_planning_ns`);
      integerAt(metric.physical_planning_ns, `${itemPath}.physical_planning_ns`);
      integerAt(metric.physical_planning_count, `${itemPath}.physical_planning_count`);
      integerAt(metric.planning_ns, `${itemPath}.planning_ns`);
      integerAt(metric.stream_open_ns, `${itemPath}.stream_open_ns`);
      integerAt(metric.execution_to_first_batch_ns, `${itemPath}.execution_to_first_batch_ns`);
      integerAt(metric.execution_remaining_ns, `${itemPath}.execution_remaining_ns`);
      integerAt(metric.execution_ns, `${itemPath}.execution_ns`);
      integerAt(metric.collect_ns, `${itemPath}.collect_ns`);
      integerAt(metric.output_arrow_wrap_ns, `${itemPath}.output_arrow_wrap_ns`);
      integerAt(metric.audit_ns, `${itemPath}.audit_ns`);
      integerAt(metric.metrics_traversal_ns, `${itemPath}.metrics_traversal_ns`);
      integerAt(metric.logical_plan_string_ns, `${itemPath}.logical_plan_string_ns`);
      integerAt(metric.physical_plan_string_ns, `${itemPath}.physical_plan_string_ns`);
      integerAt(metric.batch_envelope_ns, `${itemPath}.batch_envelope_ns`);
      integerAt(metric.run_result_ns, `${itemPath}.run_result_ns`);
      integerAt(metric.physical_metric_count, `${itemPath}.physical_metric_count`);
      integerAt(metric.output_partition_count, `${itemPath}.output_partition_count`);
      arrayAt(metric.output_partition_rows, `${itemPath}.output_partition_rows`)
        .forEach((rows, partition) => integerAt(
          rows,
          `${itemPath}.output_partition_rows[${partition}]`,
        ));
      integerAt(metric.window_partition_count, `${itemPath}.window_partition_count`);
      arrayAt(metric.window_partition_rows, `${itemPath}.window_partition_rows`)
        .forEach((rows, partition) => integerAt(
          rows,
          `${itemPath}.window_partition_rows[${partition}]`,
        ));
      integerAt(metric.spill_bytes, `${itemPath}.spill_bytes`);
      integerAt(metric.elapsed_compute_ns, `${itemPath}.elapsed_compute_ns`);
      integerAt(metric.window_compute_ns, `${itemPath}.window_compute_ns`);
      integerAt(metric.repartition_sort_compute_ns, `${itemPath}.repartition_sort_compute_ns`);
      integerAt(metric.window_operator_count, `${itemPath}.window_operator_count`);
      integerAt(metric.repartition_operator_count, `${itemPath}.repartition_operator_count`);
      integerAt(metric.sort_operator_count, `${itemPath}.sort_operator_count`);
      integerAt(metric.coalesce_operator_count, `${itemPath}.coalesce_operator_count`);
      integerAt(metric.output_rows, `${itemPath}.output_rows`);
      integerAt(metric.configured_batch_size, `${itemPath}.configured_batch_size`);
      if (metric.parallelism_mode !== 'fixed' && metric.parallelism_mode !== 'auto') {
        throw new Error(`${itemPath}.parallelism_mode must be fixed or auto`);
      }
      integerAt(metric.configured_target_partitions, `${itemPath}.configured_target_partitions`);
      integerAt(metric.requested_target_partitions, `${itemPath}.requested_target_partitions`);
      integerAt(metric.effective_target_partitions, `${itemPath}.effective_target_partitions`);
      integerAt(metric.available_parallelism, `${itemPath}.available_parallelism`);
      integerAt(metric.max_partitions, `${itemPath}.max_partitions`);
      integerAt(metric.min_rows_per_partition, `${itemPath}.min_rows_per_partition`);
      integerAt(metric.small_rows_threshold, `${itemPath}.small_rows_threshold`);
      booleanAt(metric.parallelism_decision_reused, `${itemPath}.parallelism_decision_reused`);
      integerAt(metric.decision_input_rows, `${itemPath}.decision_input_rows`);
      if (metric.decision_active_entities !== null) {
        integerAt(metric.decision_active_entities, `${itemPath}.decision_active_entities`);
      }
      stringAt(
        metric.decision_active_entities_source,
        `${itemPath}.decision_active_entities_source`,
      );
      integerAt(metric.input_rows, `${itemPath}.input_rows`);
      if (metric.active_entities !== null) {
        integerAt(metric.active_entities, `${itemPath}.active_entities`);
      }
      stringAt(metric.active_entities_source, `${itemPath}.active_entities_source`);
      stringAt(metric.partition_limit_reason, `${itemPath}.partition_limit_reason`);
      booleanAt(metric.rolling_rewrite_enabled, `${itemPath}.rolling_rewrite_enabled`);
      booleanAt(metric.diagnostics_collected, `${itemPath}.diagnostics_collected`);
      integerAt(metric.rolling_candidate_windows, `${itemPath}.rolling_candidate_windows`);
      integerAt(metric.rolling_rewritten_windows, `${itemPath}.rolling_rewritten_windows`);
      arrayAt(metric.rolling_fallback_reasons, `${itemPath}.rolling_fallback_reasons`)
        .forEach((reason, reasonIndex) => stringAt(
          reason,
          `${itemPath}.rolling_fallback_reasons[${reasonIndex}]`,
        ));
      stringAt(metric.logical_plan, `${itemPath}.logical_plan`);
      stringAt(metric.physical_plan, `${itemPath}.physical_plan`);
    });
  const metadata = objectAt(result.metadata, `${path}.metadata`);
  Object.entries(metadata).forEach(([key, item]) => jsonAt(item, `${path}.metadata.${key}`));
};

const nullableStringAt = (value: unknown, path: string): void => {
  if (value !== null) stringAt(value, path);
};

export const decodeJobResponse = (value: unknown): JobResponse => {
  const job = objectAt(value, 'job');
  exactKeys(job, [
    'id',
    'project_id',
    'status',
    'created_at',
    'started_at',
    'finished_at',
    'error_code',
    'reason_code',
    'error',
  ], 'job');
  stringAt(job.id, 'job.id');
  stringAt(job.project_id, 'job.project_id');
  stringAt(job.created_at, 'job.created_at');
  const status = literalAt(
    job.status,
    ['pending', 'running', 'completed', 'failed', 'cancelled'],
    'job.status',
  );
  nullableStringAt(job.started_at, 'job.started_at');
  nullableStringAt(job.finished_at, 'job.finished_at');
  nullableStringAt(job.error_code, 'job.error_code');
  nullableStringAt(job.reason_code, 'job.reason_code');
  nullableStringAt(job.error, 'job.error');
  if (status === 'pending') {
    if (job.started_at !== null || job.finished_at !== null) {
      fail('job', 'pending jobs cannot have start or finish times');
    }
    if (job.error_code !== null || job.reason_code !== null || job.error !== null) {
      fail('job', 'pending job payload is inconsistent');
    }
  } else if (status === 'running') {
    stringAt(job.started_at, 'job.started_at');
    if (
      job.finished_at !== null
      || job.error_code !== null
      || job.reason_code !== null
      || job.error !== null
    ) {
      fail('job', 'running job payload is inconsistent');
    }
  } else if (status === 'completed') {
    stringAt(job.started_at, 'job.started_at');
    stringAt(job.finished_at, 'job.finished_at');
    if (job.error_code !== null || job.reason_code !== null || job.error !== null) {
      fail('job.error', 'completed jobs require null errors');
    }
  } else if (status === 'failed') {
    stringAt(job.started_at, 'job.started_at');
    stringAt(job.finished_at, 'job.finished_at');
    literalAt(
      job.error_code,
      ['job_limit_exceeded', 'worker_failed'],
      'job.error_code',
    );
    if (!stringAt(job.error, 'job.error')) fail('job.error', 'must not be empty');
  } else {
    stringAt(job.finished_at, 'job.finished_at');
    if (job.error_code !== null || job.reason_code !== null || job.error !== null) {
      fail('job', 'cancelled job payload is inconsistent');
    }
  }
  return value as JobResponse;
};
