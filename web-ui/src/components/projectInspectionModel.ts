import type { NodeConfig, ProjectDocument } from '../types';

export interface LoweredNodeInspection {
  contract: 'strict ProjectDocument v3';
  nodeId: string;
  nodeKind: NodeConfig['operator']['kind'];
  sourceExpressions: string[];
  state: string;
  watermark: string;
  staticInputs: string[];
  providerIdentity: string;
  copyBoundaries: string[];
}

const FIXED_TYPE_BYTES: Readonly<Record<string, number>> = {
  bool: 1,
  int8: 1,
  uint8: 1,
  int16: 2,
  uint16: 2,
  int32: 4,
  uint32: 4,
  float32: 4,
  'date32': 4,
  'time32[s]': 4,
  int64: 8,
  uint64: 8,
  float64: 8,
  'date64': 8,
  'time64[us]': 8,
  'timestamp[ms]': 8,
  'timestamp[us]': 8,
  'timestamp[us, UTC]': 8,
};

const record = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;

const sortedJsonValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sortedJsonValue);
  const mapping = record(value);
  if (!mapping) return value;
  return Object.fromEntries(
    Object.keys(mapping)
      .sort()
      .map((key) => [key, sortedJsonValue(mapping[key])]),
  );
};

const stableJson = (value: unknown): string => JSON.stringify(sortedJsonValue(value));

const outputName = (output: Record<string, unknown>): string =>
  typeof output.output === 'string' ? output.output : '?';

const rollingSource = (value: unknown): string => {
  const output = record(value) ?? {};
  const kind = typeof output.kind === 'string' ? output.kind : 'rolling';
  const arguments_ = typeof output.input === 'string'
    ? output.input
    : [output.left, output.right].filter((item) => typeof item === 'string').join(', ');
  const details: string[] = [];
  if (typeof output.periods === 'number') details.push(`periods=${output.periods}`);
  const frame = record(output.frame);
  if (frame?.kind === 'rows' && typeof frame.size === 'number') {
    details.push(`rows=${frame.size}`);
  }
  if (frame?.kind === 'duration' && typeof frame.micros === 'number') {
    details.push(`duration=${frame.micros}µs`);
  }
  if (typeof output.min_periods === 'number') {
    details.push(`min_periods=${output.min_periods}`);
  }
  if (typeof output.ddof === 'number') details.push(`ddof=${output.ddof}`);
  const detailText = details.length ? `, ${details.join(', ')}` : '';
  return `${kind}(${arguments_}${detailText}) → ${outputName(output)}`;
};

const crossSectionSource = (value: unknown): string => {
  const output = record(value) ?? {};
  const kind = typeof output.kind === 'string' ? output.kind : 'cross_section';
  const input = typeof output.input === 'string' ? output.input : '?';
  const details = Object.entries(output)
    .filter(([key]) => !['kind', 'primitive_version', 'input', 'output'].includes(key))
    .map(([key, item]) => `${key}=${stableJson(item)}`);
  const detailText = details.length ? `, ${details.join(', ')}` : '';
  return `${kind}(${input}${detailText}) → ${outputName(output)}`;
};

const sourceExpressions = (node: NodeConfig): string[] => {
  const operator = node.operator;
  switch (operator.kind) {
    case 'expression':
      return [
        ...(operator.expression ? [operator.expression] : operator.select),
        ...(operator.filter ? [`filter ${operator.filter}`] : []),
      ];
    case 'sql':
      return [operator.query];
    case 'rolling':
      return operator.spec.outputs.map(rollingSource);
    case 'cross_section':
      return operator.spec.outputs.map(crossSectionSource);
    case 'external':
      return operator.options.expression === undefined
        ? []
        : [stableJson(operator.options.expression)];
    case 'window':
      return [stableJson(operator.spec)];
    case 'stream_join':
      return [stableJson(operator.spec)];
    default:
      return [];
  }
};

const inputSchemaCost = (node: NodeConfig): { fixed: number; variable: number } => {
  const fields = node.input_ports.find((port) => port.kind === 'table')?.schema ?? [];
  return fields.reduce(
    (cost, field) => {
      const width = FIXED_TYPE_BYTES[field.data_type];
      return width === undefined
        ? { ...cost, variable: cost.variable + 1 }
        : { ...cost, fixed: cost.fixed + width };
    },
    { fixed: 0, variable: 0 },
  );
};

const rollingBounds = (outputs: readonly unknown[]): { rows?: number; duration?: number } =>
  outputs.reduce<{ rows?: number; duration?: number }>((bounds, value) => {
    const output = record(value);
    if (!output) return bounds;
    const lagRows = typeof output.periods === 'number' ? output.periods + 1 : undefined;
    const frame = record(output.frame);
    const frameRows = frame?.kind === 'rows' && typeof frame.size === 'number'
      ? frame.size
      : undefined;
    const duration = frame?.kind === 'duration' && typeof frame.micros === 'number'
      ? frame.micros
      : undefined;
    return {
      rows: Math.max(bounds.rows ?? 0, lagRows ?? 0, frameRows ?? 0) || undefined,
      duration: Math.max(bounds.duration ?? 0, duration ?? 0) || undefined,
    };
  }, {});

const stateFact = (node: NodeConfig): string => {
  const operator = node.operator;
  if (operator.kind === 'rolling') {
    const bounds = rollingBounds(operator.spec.outputs);
    const cost = inputSchemaCost(node);
    return [
      'bounded',
      ...(bounds.rows === undefined ? [] : [`rows≤${bounds.rows}`]),
      ...(bounds.duration === undefined ? [] : [`duration≤${bounds.duration}µs`]),
      `fixed≥${cost.fixed} B/row`,
      `variable=${cost.variable}`,
    ].join(' · ');
  }
  if (operator.kind === 'cross_section') {
    const cost = inputSchemaCost(node);
    return [
      'bounded by watermark-final groups',
      'active_groups=runtime',
      `fixed≥${cost.fixed} B/row`,
      `variable=${cost.variable}`,
    ].join(' · ');
  }
  if (operator.kind === 'stream_join') {
    const { max_state_rows_per_side: rows, max_state_bytes_per_side: bytes } =
      operator.spec.limits;
    return `bounded · rows≤${rows} per side · bytes≤${bytes} per side`;
  }
  if (operator.kind === 'window') return 'bounded by declared window';
  return 'stateless';
};

const latePolicyKind = (value: unknown): string => {
  const policy = record(value);
  return typeof policy?.kind === 'string' ? policy.kind : 'unknown';
};

const watermarkFact = (node: NodeConfig): string => {
  const operator = node.operator;
  if (operator.kind === 'rolling' || operator.kind === 'cross_section') {
    return [
      'required',
      `event_time=${operator.spec.event_time}`,
      `lateness=${operator.spec.allowed_lateness_micros}µs`,
      `policy=${latePolicyKind(operator.spec.late_policy)}`,
    ].join(' · ');
  }
  if (operator.kind === 'stream_join') {
    return [
      'required',
      `left=${operator.spec.left_event_time}`,
      `right=${operator.spec.right_event_time}`,
    ].join(' · ');
  }
  if (operator.kind === 'window') return 'required by window finality';
  return 'not required';
};

type StaticInput = NonNullable<ProjectDocument['static_inputs']>[number];

const staticInputDescription = (input: StaticInput): string => input.kind === 'array'
  ? `${input.name} · array · ${input.backend} · ${input.dtype} · [${input.shape.join(', ')}]`
  : `${input.name} · table · fields=${input.schema.length}`;

const nodeStaticInputs = (project: ProjectDocument, node: NodeConfig): StaticInput[] => {
  const declared = project.static_inputs ?? [];
  const requiredNames = new Set(
    node.input_ports.filter((port) => port.required).map((port) => port.name),
  );
  return declared.filter((input) => requiredNames.has(input.name));
};

const safeStaticBytes = (input: StaticInput): number | null => {
  if (input.kind !== 'array') return null;
  const width = FIXED_TYPE_BYTES[input.dtype];
  if (width === undefined) return null;
  let bytes = width;
  for (const dimension of input.shape) {
    bytes *= dimension;
    if (!Number.isSafeInteger(bytes)) return null;
  }
  return bytes;
};

const copyBoundaries = (
  node: NodeConfig,
  staticInputs: readonly StaticInput[],
): string[] => {
  const operator = node.operator;
  if (operator.kind !== 'external') return [];
  const columns = Array.isArray(operator.options.columns)
    ? operator.options.columns.length
    : null;
  const hasTableInput = node.input_ports.some((port) => port.kind === 'table');
  const hasTableOutput = node.output_ports.some((port) => port.kind === 'table');
  const facts: string[] = [];
  if (hasTableInput && (columns !== null || operator.name === 'symbolic_matrix')) {
    facts.push(`table → dense array · columns=${columns ?? 'runtime'} · rows=runtime`);
  }
  if (operator.provider === 'jax') facts.push('host → device · backend=jax');
  for (const input of staticInputs) {
    const bytes = safeStaticBytes(input);
    facts.push(
      `static ${input.name} → provider · bytes=${bytes === null ? 'runtime' : bytes}`,
    );
  }
  if (hasTableOutput && operator.name === 'symbolic_matrix') {
    facts.push('array → table · rows preserved');
  }
  return facts;
};

export const inspectLoweredNode = (
  project: ProjectDocument,
  node: NodeConfig,
): LoweredNodeInspection => {
  const staticInputs = nodeStaticInputs(project, node);
  return {
    contract: 'strict ProjectDocument v3',
    nodeId: node.id,
    nodeKind: node.operator.kind,
    sourceExpressions: sourceExpressions(node),
    state: stateFact(node),
    watermark: watermarkFact(node),
    staticInputs: staticInputs.map(staticInputDescription),
    providerIdentity: node.operator.kind === 'external'
      ? `${node.operator.provider}:${node.operator.name}@${node.operator.version}`
      : 'native calc-flow operator',
    copyBoundaries: copyBoundaries(node, staticInputs),
  };
};
