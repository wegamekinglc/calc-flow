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
    datafusion: { batch_size: 8192, target_partitions: 1 },
  },
  data_sources: [],
  sources: [],
  sinks: [],
  state: { root: '.calc-flow-state', retention: 3 },
  ...change,
});

describe('lowered ProjectDocument inspection', () => {
  it('renders deterministic symbolic matrix provenance and copy facts', () => {
    const node: NodeConfig = {
      id: 'signals',
      input_ports: [
        { name: 'input', kind: 'table', required: true, schema: [] },
        { name: 'weights', kind: 'array', required: true, schema: [] },
      ],
      output_ports: [
        { name: 'output', kind: 'table', required: true, schema: [] },
      ],
      operator: {
        kind: 'external',
        provider: 'jax',
        name: 'symbolic_matrix',
        version: '1',
        options: {
          names: ['score'],
          expression: {
            right: { op: 'weights' },
            op: 'matmul',
            left: { op: 'input' },
          },
          columns: ['return', 'volatility'],
        },
      },
    };
    const project = projectWith(node, {
      static_inputs: [
        {
          name: 'weights',
          kind: 'array',
          mutability: 'static',
          backend: 'jax',
          dtype: 'float64',
          shape: [2, 1],
        },
      ],
    });

    expect(inspectLoweredNode(project, node)).toEqual({
      contract: 'strict ProjectDocument v3',
      nodeId: 'signals',
      nodeKind: 'external',
      sourceExpressions: [
        '{"left":{"op":"input"},"op":"matmul","right":{"op":"weights"}}',
      ],
      state: 'stateless',
      watermark: 'not required',
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
      'bounded · rows≤5 · duration≤60000000µs · fixed≥16 B/row · variable=1',
    );
    expect(inspection.watermark).toBe(
      'required · event_time=ts · lateness=5000000µs · policy=drop',
    );
    expect(inspection.providerIdentity).toBe('native calc-flow operator');
    expect(inspection.copyBoundaries).toEqual([]);
  });
});
