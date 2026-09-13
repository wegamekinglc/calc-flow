import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { derivedOutputNames, externalOutputs } from '../portNamesModel';
import { at, blankProject, type NodeConfig } from '../types';
import { NodeInspector } from './NodeInspector';
import { inspectLoweredNode } from './projectInspectionModel';
import { StreamConfigEditor } from './StreamConfigEditor';

const roll = (): NodeConfig => ({
  id: 'roll', input_ports: [], output_ports: [],
  operator: {
    kind: 'rolling',
    spec: {
      configuration_version: 1, state_layout_version: 1,
      partition_by: ['symbol'], event_time: 'ts', sequence_by: ['seq'],
      allowed_lateness_micros: 3, value_policy: 'stateful_numeric_v1',
      late_policy: { kind: 'side_output', metrics_version: 1, schema_version: 1 },
      outputs: [{ kind: 'lag', primitive_version: 1, input: 'x', output: 'previous', periods: 1 }],
    },
  },
});

const project = () => ({
  ...blankProject(),
  runtime: { mode: 'stream' as const, options: {
    checkpoint_interval_ms: 5000, max_batch_rows: 100, max_batch_bytes: 100000,
  } },
  graph: { ...blankProject().graph, nodes: [roll()] },
  data_sources: [],
  sinks: ['output', 'late'].map((binding) => ({
    binding, connector: { provider: 'builtin', name: 'file', version: '1' },
    delivery: 'at_least_once' as const, format: null,
    options: { path: binding }, secrets: {},
  })),
});

describe('late output Studio contract', () => {
  it('derives both handles without requiring explicit output ports', () => {
    expect(derivedOutputNames(roll())).toEqual(['output', 'late']);
  });

  it('uses native physical names after a late expression route', () => {
    const value = project();
    const expression = { ...at(blankProject().graph.nodes), id: 'diagnostics' };
    const graph = { ...value.graph, nodes: [roll(), expression], edges: [{
      source_node: 'roll', source_port: 'late', target_node: 'diagnostics', target_port: 'input',
    }] };
    expect(externalOutputs(graph)).toEqual([
      { nodeId: 'roll', port: 'output', binding: 'roll.output' },
      { nodeId: 'diagnostics', port: 'output', binding: 'diagnostics.output' },
    ]);
  });

  it('keeps a bound late policy intact and fails closed without runtime support', () => {
    const node = roll();
    const onChange = vi.fn();
    render(<NodeInspector node={node} arrowTypes={[]} udfs={[]}
      onChange={onChange} onSqlAliasEdit={vi.fn()} onDelete={vi.fn()}
      streamMode lateOutputInUse lateOutputSupported={false} />);
    expect(screen.getByLabelText('Late row policy')).toBeDisabled();
    expect(screen.getByText(/runtime has not confirmed/)).toBeInTheDocument();
    expect(screen.getByText(/Disconnect late/)).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('enables the local policy without changing existing configuration', () => {
    const base = roll();
    if (base.operator.kind !== 'rolling') throw new Error('rolling fixture');
    const node = { ...base, operator: { ...base.operator, spec: {
      ...base.operator.spec, late_policy: { kind: 'drop' as const, metrics_version: 1 },
    } } };
    const onChange = vi.fn();
    render(<NodeInspector node={node} arrowTypes={[]} udfs={[]}
      onChange={onChange} onSqlAliasEdit={vi.fn()} onDelete={vi.fn()}
      streamMode lateOutputSupported lateOutputInUse={false} />);
    fireEvent.change(screen.getByLabelText('Late row policy'), { target: { value: 'side_output' } });
    expect(onChange).toHaveBeenCalledWith({ ...node, operator: {
      ...node.operator, spec: { ...node.operator.spec,
        late_policy: { kind: 'side_output', metrics_version: 1, schema_version: 1 },
      },
    } });
    expect(node.operator.spec.late_policy.kind).toBe('drop');
  });

  it('preserves both bindings when clicking the active mode or attempting batch', () => {
    const onChange = vi.fn();
    const value = project();
    const before = structuredClone(value);
    render(<StreamConfigEditor project={value} connectors={[]} onChange={onChange} />);
    fireEvent.click(screen.getByRole('button', { name: 'Stream' }));
    fireEvent.click(screen.getByRole('button', { name: 'Batch' }));
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByText(/Side output requires stream mode/)).toBeInTheDocument();
    expect(value).toEqual(before);
  });

  it('edits one sink delivery without changing the other route', () => {
    const onChange = vi.fn();
    const value = project();
    render(<StreamConfigEditor project={value} connectors={[]} onChange={onChange} />);
    fireEvent.change(at(screen.getAllByLabelText('Delivery'), 1), { target: { value: 'best_effort' } });
    const changed = at(onChange.mock.calls)[0];
    expect(changed.sinks[0]).toEqual(value.sinks[0]);
    expect(changed.sinks[1]).toEqual({ ...value.sinks[1], delivery: 'best_effort' });
    expect(changed.graph).toEqual(value.graph);
  });

  it('offers each physical output binding and identifies missing consumers', () => {
    const value = { ...project(), sinks: project().sinks.slice(0, 1) };
    render(<StreamConfigEditor project={value} connectors={[]} onChange={vi.fn()} />);
    expect(screen.getByText('Unbound outputs: late')).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'roll.late → late', hidden: true })).toHaveValue('late');
  });

  it('restores the exact Error policy only after consumers are removed', () => {
    const node = roll();
    const onChange = vi.fn();
    render(<NodeInspector node={node} arrowTypes={[]} udfs={[]}
      onChange={onChange} onSqlAliasEdit={vi.fn()} onDelete={vi.fn()}
      streamMode lateOutputSupported lateOutputInUse={false} />);
    fireEvent.change(screen.getByLabelText('Late row policy'), { target: { value: 'error' } });
    expect(at(onChange.mock.calls)[0].operator.spec.late_policy).toEqual({ kind: 'error', scope: 'envelope' });
  });

  it('keeps explicit normal schema and derives non-null 64-bit late diagnostics', () => {
    const base = roll();
    if (base.operator.kind !== 'rolling') throw new Error('rolling fixture');
    const input = { name: 'input', kind: 'table' as const, required: true,
      schema: [{ name: 'ts', data_type: 'timestamp[us, UTC]', nullable: false }],
    };
    const output = { ...input, name: 'output' };
    const node = { ...base, input_ports: [input], output_ports: [output], operator: {
      ...base.operator, spec: { ...base.operator.spec, late_policy: { kind: 'drop' as const, metrics_version: 1 } },
    } };
    const onChange = vi.fn();
    render(<NodeInspector node={node} arrowTypes={[]} udfs={[]} onChange={onChange}
      onSqlAliasEdit={vi.fn()} onDelete={vi.fn()} streamMode lateOutputSupported lateOutputInUse={false} />);
    fireEvent.change(screen.getByLabelText('Late row policy'), { target: { value: 'side_output' } });
    const changed = at(onChange.mock.calls)[0] as NodeConfig;
    expect(changed.output_ports[0]).toEqual(output);
    const late = at(changed.output_ports, 1);
    expect(late.schema[0]).toEqual(input.schema[0]);
    expect(late.schema.slice(3, 6).map((field) => field.data_type)).toEqual(['int64', 'int64', 'int64']);
    expect(late.schema.slice(-2).map((field) => field.data_type)).toEqual(['uint64', 'uint64']);
    expect(late.schema.every((field) => !field.nullable)).toBe(true);
    expect(node.output_ports).toEqual([output]);
  });

  it('labels normal progress separately from late and shows local closing boundary', () => {
    const node = roll();
    const inspection = inspectLoweredNode(project(), node);
    expect(inspection.watermark).toContain('output only');
    render(<NodeInspector node={node} inspection={inspection} arrowTypes={[]} udfs={[]}
      onChange={vi.fn()} onSqlAliasEdit={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.getByText('out · late')).toBeInTheDocument();
    expect(screen.getByText(/no Watermark\/Idle/)).toBeInTheDocument();
    expect(screen.getByText(/t \+ 3µs ≤ W/)).toBeInTheDocument();
    expect(screen.getByText(/64-bit.*Arrow/)).toBeInTheDocument();
  });
});
