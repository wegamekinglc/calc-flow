import { fireEvent, render, screen } from '@testing-library/react';
import type { Dispatch } from 'react';
import { expect, it, vi } from 'vitest';

import { at, type ArrowFieldConfig, type NodeConfig } from '../types';
import fields from './lateSchema.fixture.json';
import { NodeInspector } from './NodeInspector';

vi.mock('./SchemaEditor', () => ({
  SchemaEditor: ({ onChange }: { onChange: Dispatch<ArrowFieldConfig[]> }) =>
    <button onClick={() => onChange(fields.output)}>Apply complete schema</button>,
}));

const operator = (kind: 'rolling' | 'cross_section'): NodeConfig['operator'] => {
  const shared = {
    configuration_version: 1, state_layout_version: 1,
    event_time: 'ts', sequence_by: ['seq'], allowed_lateness_micros: 3,
    late_policy: { kind: 'side_output' as const, metrics_version: 1, schema_version: 1 },
  };
  return kind === 'rolling' ? { kind, spec: {
    ...shared, partition_by: ['symbol'], value_policy: 'stateful_numeric_v1',
    outputs: [{ kind: 'lag', primitive_version: 1, input: 'x', output: 'result', periods: 1 }],
  } } : { kind, spec: {
    ...shared, partition_by: [], entity_by: ['symbol'], grouping: { kind: 'exact_time' },
    value_policy: 'nan_exclude_preserve_v1',
    outputs: [{ kind: 'rank', primitive_version: 1, input: 'x', output: 'result',
      tie_method: 'average', direction: 'ascending', null_placement: 'last', min_samples: 1 }],
  } };
};

it.each(['rolling', 'cross_section'] as const)(
  'preserves the complete implicit late schema when editing %s normal output',
  (kind) => {
    const node: NodeConfig = {
      id: 'roll', operator: operator(kind),
      input_ports: [{ name: 'input', kind: 'table', required: true, schema: fields.input }],
      output_ports: [],
    };
    const before = structuredClone(node);
    const onChange = vi.fn();
    render(<NodeInspector node={node} arrowTypes={[]} udfs={[]} onChange={onChange}
      onSqlAliasEdit={vi.fn()} onDelete={vi.fn()} streamMode lateOutputSupported lateOutputInUse />);
    fireEvent.click(at(screen.getAllByRole('button', { name: 'Apply complete schema' }), 1));
    const changed = at(onChange.mock.calls)[0] as NodeConfig;
    expect(changed).toEqual({ ...node, output_ports: [
      { name: 'output', kind: 'table', required: true, schema: fields.output },
      { name: 'late', kind: 'table', required: true, schema: fields.late },
    ] });
    expect(node).toEqual(before);
  },
);
