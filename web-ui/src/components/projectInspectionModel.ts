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

const FIXED_TYPE_BYTES: ReadonlyMap<string, number> = new Map([
  ['bool', 1],
  ['int8', 1],
  ['uint8', 1],
  ['int16', 2],
  ['uint16', 2],
  ['int32', 4],
  ['uint32', 4],
  ['float32', 4],
  ['date32', 4],
  ['time32[s]', 4],
  ['int64', 8],
  ['uint64', 8],
  ['float64', 8],
  ['date64', 8],
  ['time64[us]', 8],
  ['timestamp[ms]', 8],
  ['timestamp[us]', 8],
  ['timestamp[us, UTC]', 8],
]);

const record = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;

const compareCanonicalKeys = (left: string, right: string): number => {
  if (left === right) return 0;
  return left < right ? -1 : 1;
};

const sortedJsonValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sortedJsonValue);
  const mapping = record(value);
  if (!mapping) return value;
  const entries = Object.entries(mapping).sort(([left], [right]) =>
    compareCanonicalKeys(left, right));
  return Object.fromEntries(
    entries.map(([key, item]) => [key, sortedJsonValue(item)]),
  );
};

const stableJson = (value: unknown): string => JSON.stringify(sortedJsonValue(value));

const isString = (value: string | null): value is string => value !== null;

const outputName = (output: Record<string, unknown>): string =>
  typeof output.output === 'string' ? output.output : '?';

const rollingArguments = (output: Record<string, unknown>): string => {
  if (typeof output.input === 'string') return output.input;
  return [output.left, output.right]
    .filter((item): item is string => typeof item === 'string')
    .join(', ');
};

const numberDetail = (
  label: string,
  value: unknown,
  suffix = '',
): string | null => typeof value === 'number'
  ? `${label}=${value}${suffix}`
  : null;

const frameDetails = (value: unknown): string[] => {
  const frame = record(value);
  if (frame?.kind === 'rows') {
    return [numberDetail('rows', frame.size)].filter(isString);
  }
  if (frame?.kind === 'duration') {
    return [numberDetail('duration', frame.micros, 'µs')].filter(isString);
  }
  return [];
};

const detailSuffix = (details: readonly string[]): string =>
  details.length === 0 ? '' : `, ${details.join(', ')}`;

const rollingSource = (value: unknown): string => {
  const output = record(value) ?? {};
  const kind = typeof output.kind === 'string' ? output.kind : 'rolling';
  const details = [
    numberDetail('periods', output.periods),
    ...frameDetails(output.frame),
    numberDetail('min_periods', output.min_periods),
    numberDetail('ddof', output.ddof),
  ].filter(isString);
  return `${kind}(${rollingArguments(output)}${detailSuffix(details)})`
    + ` → ${outputName(output)}`;
};

const crossSectionSource = (value: unknown): string => {
  const output = record(value) ?? {};
  const kind = typeof output.kind === 'string' ? output.kind : 'cross_section';
  const input = typeof output.input === 'string' ? output.input : '?';
  const details = Object.entries(output)
    .filter(([key]) => !['kind', 'primitive_version', 'input', 'output'].includes(key))
    .map(([key, item]) => `${key}=${stableJson(item)}`);
  return `${kind}(${input}${detailSuffix(details)}) → ${outputName(output)}`;
};

const expressionSources = (
  operator: Extract<NodeConfig['operator'], { kind: 'expression' }>,
): string[] => [
  ...(operator.expression ? [operator.expression] : operator.select),
  ...(operator.filter ? [`filter ${operator.filter}`] : []),
];

const directSourceExpressions = (node: NodeConfig): string[] | null => {
  const operator = node.operator;
  if (operator.kind === 'expression') return expressionSources(operator);
  if (operator.kind === 'sql') return [operator.query];
  if (operator.kind !== 'external') return null;
  return operator.options.expression === undefined
    ? []
    : [stableJson(operator.options.expression)];
};

const structuredSourceExpressions = (node: NodeConfig): string[] => {
  const operator = node.operator;
  if (operator.kind === 'rolling') return operator.spec.outputs.map(rollingSource);
  if (operator.kind === 'cross_section') {
    return operator.spec.outputs.map(crossSectionSource);
  }
  if (operator.kind === 'window' || operator.kind === 'stream_join') {
    return [stableJson(operator.spec)];
  }
  return [];
};

const sourceExpressions = (node: NodeConfig): string[] =>
  directSourceExpressions(node) ?? structuredSourceExpressions(node);

const inputSchemaCost = (node: NodeConfig): { fixed: number; variable: number } => {
  const fields = node.input_ports.find((port) => port.kind === 'table')?.schema ?? [];
  return fields.reduce(
    (cost, field) => {
      const width = FIXED_TYPE_BYTES.get(field.data_type);
      return width === undefined
        ? { ...cost, variable: cost.variable + 1 }
        : { ...cost, fixed: cost.fixed + width };
    },
    { fixed: 0, variable: 0 },
  );
};

const rowHistoryBound = (value: unknown): number | null => {
  const output = record(value);
  if (!output) return null;
  const frame = record(output.frame);
  const candidates = [
    typeof output.periods === 'number' ? output.periods : null,
    frame?.kind === 'rows' && typeof frame.size === 'number' ? frame.size : null,
  ].filter((item): item is number => item !== null);
  return candidates.length === 0 ? null : Math.max(...candidates);
};

const durationHistoryBound = (value: unknown): number | null => {
  const output = record(value);
  const frame = record(output?.frame);
  return frame?.kind === 'duration' && typeof frame.micros === 'number'
    ? frame.micros
    : null;
};

const maximum = (values: readonly (number | null)[]): number | null => {
  const present = values.filter((value): value is number => value !== null);
  return present.length === 0 ? null : Math.max(...present);
};

interface RollingBounds {
  rows: number | null;
  duration: number | null;
}

const rollingBounds = (outputs: readonly unknown[]): RollingBounds => ({
  rows: maximum(outputs.map(rowHistoryBound)),
  duration: maximum(outputs.map(durationHistoryBound)),
});

const rollingHistoryFacts = (bounds: RollingBounds): string[] => [
  bounds.rows === null ? null : `row-frame history≤${bounds.rows} rows`,
  bounds.duration === null
    ? null
    : `duration-frame history≤${bounds.duration}µs`,
].filter(isString);

const rollingStateFact = (node: NodeConfig): string => {
  if (node.operator.kind !== 'rolling') return 'unknown';
  const bounds = rollingBounds(node.operator.spec.outputs);
  const cost = inputSchemaCost(node);
  return [
    'bounded',
    ...rollingHistoryFacts(bounds),
    `fixed≥${cost.fixed} B/retained row`,
    `variable=${cost.variable}`,
  ].join(' · ');
};

const stateFact = (node: NodeConfig): string => {
  const operator = node.operator;
  if (operator.kind === 'rolling') return rollingStateFact(node);
  if (operator.kind === 'cross_section') {
    const cost = inputSchemaCost(node);
    return [
      'bounded by watermark-final groups',
      'active_groups=runtime',
      `fixed≥${cost.fixed} B/retained row`,
      `variable=${cost.variable}`,
    ].join(' · ');
  }
  if (operator.kind === 'stream_join') {
    const { max_state_rows_per_side: rows, max_state_bytes_per_side: bytes } =
      operator.spec.limits;
    return `bounded · rows≤${rows} per side · bytes≤${bytes} per side`;
  }
  if (operator.kind === 'window') return 'bounded by declared window';
  if (operator.kind === 'external') return 'unknown · provider lifecycle not encoded';
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
  if (operator.kind === 'external') {
    return 'unknown · provider watermark contract not encoded';
  }
  return 'not required';
};

type StaticInput = NonNullable<ProjectDocument['static_inputs']>[number];
type ExternalOperator = Extract<NodeConfig['operator'], { kind: 'external' }>;
type MatrixBackend = 'numpy' | 'jax';

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
  const width = FIXED_TYPE_BYTES.get(input.dtype);
  if (width === undefined) return null;
  let bytes = width;
  for (const dimension of input.shape) {
    bytes *= dimension;
    if (!Number.isSafeInteger(bytes)) return null;
  }
  return bytes;
};

const matrixBackend = (operator: ExternalOperator): MatrixBackend | null => {
  if (operator.name !== 'symbolic_matrix' || operator.version !== '1') return null;
  if (operator.provider === 'numpy' || operator.provider === 'jax') {
    return operator.provider;
  }
  return null;
};

const isNonEmptyStringArray = (value: unknown): value is string[] =>
  Array.isArray(value)
  && value.length > 0
  && value.every((item) => typeof item === 'string' && item.length > 0);

const hasUniqueItems = (values: readonly string[]): boolean =>
  new Set(values).size === values.length;

const isUniqueStringArray = (value: unknown): value is string[] =>
  isNonEmptyStringArray(value) && hasUniqueItems(value);

const hasExactKeys = (
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean => JSON.stringify(Object.keys(value).sort(compareCanonicalKeys))
  === JSON.stringify(expected);

const isExactPort = (
  port: NodeConfig['input_ports'][number],
  name: string,
  kind: 'table' | 'array',
): boolean => hasExactKeys(port, ['kind', 'name', 'required', 'schema'])
  && port.name === name
  && port.kind === kind
  && port.required
  && port.schema.length === 0;

const hasExactMatrixPorts = (node: NodeConfig): boolean => {
  if (node.input_ports.length !== 2 || node.output_ports.length !== 1) return false;
  const [tableInput, weightsInput] = node.input_ports;
  const [tableOutput] = node.output_ports;
  return isExactPort(tableInput, 'input', 'table')
    && isExactPort(weightsInput, 'weights', 'array')
    && isExactPort(tableOutput, 'output', 'table');
};

const isExpressionLeaf = (value: unknown, operation: string): boolean => {
  const expression = record(value);
  return expression !== null
    && hasExactKeys(expression, ['op'])
    && expression.op === operation;
};

const isRecognizedMatrixExpression = (value: unknown): boolean => {
  const expression = record(value);
  return expression !== null
    && hasExactKeys(expression, ['left', 'op', 'right'])
    && expression.op === 'matmul'
    && isExpressionLeaf(expression.left, 'input')
    && isExpressionLeaf(expression.right, 'weights');
};

const hasStaticWeightIdentity = (
  input: unknown,
  backend: MatrixBackend,
): boolean => {
  const rawInput = record(input);
  if (rawInput === null) return false;
  if (!hasExactKeys(
    rawInput,
    ['backend', 'dtype', 'kind', 'mutability', 'name', 'shape'],
  )) return false;
  if (rawInput.kind !== 'array') return false;
  if (rawInput.name !== 'weights') return false;
  if (rawInput.mutability !== 'static') return false;
  if (rawInput.backend !== backend) return false;
  return rawInput.dtype === 'float32' || rawInput.dtype === 'float64';
};

const hasStaticWeightShape = (
  input: StaticInput,
  columns: readonly string[],
  names: readonly string[],
): boolean => input.kind === 'array'
  && input.shape.length === 2
  && input.shape[0] === columns.length
  && input.shape[1] === names.length;

const hasExactStaticWeights = (
  staticInputs: readonly StaticInput[],
  backend: MatrixBackend,
  columns: readonly string[],
  names: readonly string[],
): boolean => {
  if (staticInputs.length !== 1) return false;
  const [input] = staticInputs;
  return hasStaticWeightIdentity(input, backend)
    && hasStaticWeightShape(input, columns, names);
};

interface MatrixOptions {
  columns: string[];
  names: string[];
}

const recognizedMatrixOptions = (
  options: Record<string, unknown>,
): MatrixOptions | null => {
  if (!hasExactKeys(options, ['columns', 'expression', 'names'])) return null;
  if (!isUniqueStringArray(options.columns)) return null;
  if (!isUniqueStringArray(options.names)) return null;
  return isRecognizedMatrixExpression(options.expression)
    ? { columns: options.columns, names: options.names }
    : null;
};

interface SymbolicMatrixFacts {
  backend: MatrixBackend;
  columnCount: number;
}

const symbolicMatrixFacts = (
  node: NodeConfig,
  staticInputs: readonly StaticInput[],
): SymbolicMatrixFacts | null => {
  if (node.operator.kind !== 'external') return null;
  const backend = matrixBackend(node.operator);
  if (backend === null || !hasExactMatrixPorts(node)) return null;
  if (!hasExactKeys(
    node.operator,
    ['kind', 'name', 'options', 'provider', 'version'],
  )) return null;
  const options = recognizedMatrixOptions(node.operator.options);
  if (options === null) return null;
  return hasExactStaticWeights(staticInputs, backend, options.columns, options.names)
    ? { backend, columnCount: options.columns.length }
    : null;
};

const copyBoundaries = (
  node: NodeConfig,
  staticInputs: readonly StaticInput[],
): string[] => {
  const matrix = symbolicMatrixFacts(node, staticInputs);
  if (matrix === null) return [];
  const staticFacts = staticInputs.map((input) => {
    const bytes = safeStaticBytes(input);
    return `static ${input.name} → provider · bytes=${bytes ?? 'runtime'}`;
  });
  return [
    `table → dense array · columns=${matrix.columnCount} · rows=runtime`,
    ...(matrix.backend === 'jax' ? ['host → device · backend=jax'] : []),
    ...staticFacts,
    'array → table · rows preserved',
  ];
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
