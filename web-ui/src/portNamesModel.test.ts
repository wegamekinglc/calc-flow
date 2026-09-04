import { describe, expect, it } from 'vitest';

import { derivedInputNames, derivedOutputNames } from './portNamesModel';
import type { NodeConfig } from './types';

const streamJoinNode = (): NodeConfig => ({
  id: 'join',
  input_ports: [
    { name: 'left', kind: 'table', required: true, schema: [] },
    { name: 'right', kind: 'table', required: true, schema: [] },
  ],
  output_ports: [],
  position: { x: 0, y: 0 },
  operator: {
    kind: 'stream_join',
    spec: {
      join_type: 'inner',
      left_keys: [],
      right_keys: [],
      left_event_time: '',
      right_event_time: '',
      bounds: { before_micros: 0, after_micros: 0 },
      limits: {
        max_state_rows_per_side: 100_000,
        max_state_bytes_per_side: 134_217_728,
        max_matches_per_input_batch: 1_000_000,
      },
      left_prefix: 'left',
      right_prefix: 'right',
    },
  },
});

const expressionNode = (): NodeConfig => ({
  id: 'calc',
  input_ports: [],
  output_ports: [],
  position: { x: 0, y: 0 },
  operator: {
    kind: 'expression',
    expression: 'total = a + b',
    select: [],
    filter: null,
    udfs: [],
  },
});

const sqlNode = (): NodeConfig => ({
  id: 'sql',
  input_ports: [],
  output_ports: [],
  position: { x: 0, y: 0 },
  operator: {
    kind: 'sql',
    query: 'SELECT * FROM input',
    aliases: ['staging', 'budget'],
    udfs: [],
  },
});

const externalNode = (): NodeConfig => ({
  id: 'ext',
  input_ports: [],
  output_ports: [],
  position: { x: 0, y: 0 },
  operator: {
    kind: 'external',
    name: 'numpy',
    provider: 'numpy',
    version: '1',
    options: {},
  },
});

describe('derivedInputNames', () => {
  it('prefers declared input ports', () => {
    const node = expressionNode();
    const declared: NodeConfig = {
      ...node,
      input_ports: [{ name: 'events', kind: 'table', required: true, schema: [] }],
    };
    expect(derivedInputNames(declared)).toEqual(['events']);
  });

  it('falls back to sql aliases for sql nodes without declared ports', () => {
    expect(derivedInputNames(sqlNode())).toEqual(['staging', 'budget']);
  });

  it('falls back to a single input port for expression nodes', () => {
    expect(derivedInputNames(expressionNode())).toEqual(['input']);
  });

  it('keeps declared stream join side ports', () => {
    expect(derivedInputNames(streamJoinNode())).toEqual(['left', 'right']);
  });
});

describe('derivedOutputNames', () => {
  it('prefers declared output ports', () => {
    const node = expressionNode();
    const declared: NodeConfig = {
      ...node,
      output_ports: [{ name: 'events', kind: 'table', required: true, schema: [] }],
    };
    expect(derivedOutputNames(declared)).toEqual(['events']);
  });

  it('falls back to the required output port for stream join nodes', () => {
    expect(derivedOutputNames(streamJoinNode())).toEqual(['output']);
  });

  it('falls back to a single output port for expression and sql nodes', () => {
    expect(derivedOutputNames(expressionNode())).toEqual(['output']);
    expect(derivedOutputNames(sqlNode())).toEqual(['output']);
  });

  it('derives no fallback output port for external nodes', () => {
    expect(derivedOutputNames(externalNode())).toEqual([]);
  });
});
