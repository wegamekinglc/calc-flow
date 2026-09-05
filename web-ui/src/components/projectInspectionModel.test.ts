import { describe, expect, it } from 'vitest';

import type { NodeConfig, ProjectDocument } from '../types';
import { inspectLoweredNode } from './projectInspectionModel';

const projectWith = (
  node: NodeConfig,
  change: Partial<ProjectDocument> = {},
): ProjectDocument => ({
  format_version: 3,
  id: 'inspect_project',
  name: 'Inspection project',
  description: '',
  runtime: {
    mode: 'stream',
    options: {
      checkpoint_interval_ms: 1_000,
      max_batch_bytes: 1_048_576,
      max_batch_rows: 1_024,
    },
  },
  graph: {
    name: 'Inspection graph',
    nodes: [node],
    edges: [],
    datafusion: {
      batch_size: 8192,
      target_partitions: 1,
      parallelism_mode: 'fixed',
      max_partitions: 32,
      min_rows_per_partition: 65536,
      small_rows_threshold: 10001,
      enable_rolling_rewrite: true,
      collect_diagnostics: true,
    },
  },
  data_sources: [],
  sources: [],
  sinks: [],
  state: { root: '.calc-flow-state', retention: 3 },
  ...change,
});

const validMatrixOptions = (): Record<string, unknown> => ({
  names: ['score'],
  expression: {
    right: { op: 'weights' },
    op: 'matmul',
    left: { op: 'input' },
  },
  columns: ['return', 'volatility'],
});

interface MatrixNodeChange {
  options?: Record<string, unknown>;
  inputPorts?: NodeConfig['input_ports'];
  outputPorts?: NodeConfig['output_ports'];
}

const symbolicMatrixNode = (change: MatrixNodeChange = {}): NodeConfig => ({
  id: 'signals',
  input_ports: change.inputPorts ?? [
    { name: 'input', kind: 'table', required: true, schema: [] },
    { name: 'weights', kind: 'array', required: true, schema: [] },
  ],
  output_ports: change.outputPorts ?? [
    { name: 'output', kind: 'table', required: true, schema: [] },
  ],
  operator: {
    kind: 'external',
    provider: 'jax',
    name: 'symbolic_matrix',
    version: '1',
    options: change.options ?? validMatrixOptions(),
  },
});

const staticWeights = (
  shape: number[] = [2, 1],
): NonNullable<ProjectDocument['static_inputs']>[number] => ({
  name: 'weights',
  kind: 'array',
  mutability: 'static',
  backend: 'jax',
  dtype: 'float64',
  shape,
});

describe('lowered ProjectDocument inspection', () => {
  it('renders deterministic symbolic matrix provenance and copy facts', () => {
    const node = symbolicMatrixNode();
    const project = projectWith(node, {
      static_inputs: [staticWeights()],
    });

    expect(inspectLoweredNode(project, node)).toEqual({
      contract: 'strict ProjectDocument v3',
      nodeId: 'signals',
      nodeKind: 'external',
      sourceExpressions: [
        '{"left":{"op":"input"},"op":"matmul","right":{"op":"weights"}}',
      ],
      state: 'unknown · provider lifecycle not encoded',
      watermark: 'unknown · provider watermark contract not encoded',
      staticInputs: ['weights · array · jax · float64 · [2, 1]'],
      providerIdentity: 'jax:symbolic_matrix@1',
      copyBoundaries: [
        'table → dense array · columns=2 · rows=runtime',
        'host → device · backend=jax',
        'static weights → provider · bytes=16',
        'array → table · rows preserved',
      ],
    });
  });

  it.each([
    {
      case: 'an extra option',
      node: symbolicMatrixNode({
        options: { ...validMatrixOptions(), extra: true },
      }),
      weights: staticWeights(),
    },
    {
      case: 'an arbitrary expression tree',
      node: symbolicMatrixNode({
        options: {
          ...validMatrixOptions(),
          expression: { op: 'literal', value: true },
        },
      }),
      weights: staticWeights(),
    },
    {
      case: 'duplicate selected columns',
      node: symbolicMatrixNode({
        options: {
          ...validMatrixOptions(),
          columns: ['return', 'return'],
        },
      }),
      weights: staticWeights(),
    },
    {
      case: 'names and static-weight width that disagree',
      node: symbolicMatrixNode({
        options: { ...validMatrixOptions(), names: ['score', 'hedge'] },
      }),
      weights: staticWeights(),
    },
    {
      case: 'the wrong table input name',
      node: symbolicMatrixNode({
        inputPorts: [
          { name: 'table', kind: 'table', required: true, schema: [] },
          { name: 'weights', kind: 'array', required: true, schema: [] },
        ],
      }),
      weights: staticWeights(),
    },
    {
      case: 'an optional output with the wrong name',
      node: symbolicMatrixNode({
        outputPorts: [
          { name: 'result', kind: 'table', required: false, schema: [] },
        ],
      }),
      weights: staticWeights(),
    },
    {
      case: 'an extra input port',
      node: symbolicMatrixNode({
        inputPorts: [
          { name: 'input', kind: 'table', required: true, schema: [] },
          { name: 'weights', kind: 'array', required: true, schema: [] },
          { name: 'side', kind: 'table', required: false, schema: [] },
        ],
      }),
      weights: staticWeights(),
    },
  ])('does not infer copy facts for $case', ({ node, weights }) => {
    const project = projectWith(node, { static_inputs: [weights] });

    expect(inspectLoweredNode(project, node).copyBoundaries).toEqual([]);
  });

  it('does not invent lifecycle or copy facts for an arbitrary external provider', () => {
    const node: NodeConfig = {
      id: 'custom_jax_provider',
      input_ports: [
        { name: 'input', kind: 'table', required: true, schema: [] },
      ],
      output_ports: [
        { name: 'output', kind: 'table', required: true, schema: [] },
      ],
      operator: {
        kind: 'external',
        provider: 'jax',
        name: 'stateful_features',
        version: '9',
        options: { columns: ['price'] },
      },
    };

    const inspection = inspectLoweredNode(projectWith(node), node);

    expect(inspection.state).toBe('unknown · provider lifecycle not encoded');
    expect(inspection.watermark).toBe(
      'unknown · provider watermark contract not encoded',
    );
    expect(inspection.copyBoundaries).toEqual([]);
  });

  it('sorts serialized expression keys by locale-independent code points', () => {
    const node: NodeConfig = {
      id: 'serialized_expression',
      input_ports: [],
      output_ports: [],
      operator: {
        kind: 'external',
        provider: 'custom',
        name: 'serialized',
        version: '1',
        options: {
          expression: { 'é': 4, z: 2, A: 1, 'ä': 3 },
        },
      },
    };

    expect(inspectLoweredNode(projectWith(node), node).sourceExpressions).toEqual([
      '{"A":1,"z":2,"ä":3,"é":4}',
    ]);
  });

  it('explains rolling state and watermark bounds from the lowered node', () => {
    const node: NodeConfig = {
      id: 'signals__cf_rolling',
      input_ports: [
        {
          name: 'input',
          kind: 'table',
          required: true,
          schema: [
            { name: 'ts', data_type: 'timestamp[us, UTC]', nullable: false },
            { name: 'symbol', data_type: 'string', nullable: false },
            { name: 'price', data_type: 'float64', nullable: true },
          ],
        },
      ],
      output_ports: [],
      operator: {
        kind: 'rolling',
        spec: {
          configuration_version: 1,
          state_layout_version: 1,
          partition_by: ['symbol'],
          event_time: 'ts',
          sequence_by: ['ts'],
          outputs: [
            {
              kind: 'lag',
              primitive_version: 1,
              input: 'price',
              output: 'previous',
              periods: 4,
            },
            {
              kind: 'mean',
              primitive_version: 1,
              input: 'price',
              output: 'mean_1m',
              frame: { kind: 'duration', micros: 60_000_000 },
              min_periods: 2,
            },
          ],
          allowed_lateness_micros: 5_000_000,
          late_policy: { kind: 'drop', metrics_version: 1 },
          value_policy: 'stateful_numeric_v1',
        },
      },
    };

    const inspection = inspectLoweredNode(projectWith(node), node);

    expect(inspection.sourceExpressions).toEqual([
      'lag(price, periods=4) → previous',
      'mean(price, duration=60000000µs, min_periods=2) → mean_1m',
    ]);
    expect(inspection.state).toBe(
      'bounded · row-frame history≤4 rows · duration-frame history≤60000000µs'
      + ' · fixed≥16 B/retained row · variable=1',
    );
    expect(inspection.watermark).toBe(
      'required · event_time=ts · lateness=5000000µs · policy=drop',
    );
    expect(inspection.providerIdentity).toBe('native calc-flow operator');
    expect(inspection.copyBoundaries).toEqual([]);
  });
});
