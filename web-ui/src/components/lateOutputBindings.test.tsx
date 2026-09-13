import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import { connectProject } from '../App';
import { externalOutputs, lateOutputInUse, withProjectGraph } from '../portNamesModel';
import { at, blankProject, type EditableProject, type NodeConfig } from '../types';
import { NodeInspector } from './NodeInspector';
import { editSqlInputAliases } from './inputAliasEditorModel';
import { StreamConfigEditor } from './StreamConfigEditor';

const roll = (id: string): NodeConfig => ({
  id, input_ports: [], output_ports: [],
  operator: { kind: 'rolling', spec: {
    configuration_version: 1, state_layout_version: 1, partition_by: ['symbol'],
    event_time: 'ts', sequence_by: ['seq'], allowed_lateness_micros: 3,
    value_policy: 'stateful_numeric_v1',
    late_policy: { kind: 'side_output', metrics_version: 1, schema_version: 1 },
    outputs: [{ kind: 'lag', primitive_version: 1, input: 'x', output: 'previous', periods: 1 }],
  } },
});

const project = (): EditableProject => ({
  ...blankProject(), runtime: { mode: 'stream', options: {
    checkpoint_interval_ms: 5000, max_batch_bytes: 100000, max_batch_rows: 100,
  } },
  graph: { ...blankProject().graph, name: 'late-bindings', edges: [], nodes: [
    roll('left'), roll('right'),
    { ...at(blankProject().graph.nodes), id: 'left_route' },
    { ...at(blankProject().graph.nodes), id: 'right_route' },
  ] },
  sinks: ['left.output', 'left.late', 'right.output', 'right.late'].map((binding, i) => ({
    binding, connector: { provider: 'calc-flow-connectors', name: 'file', version: '2.0.0' },
    delivery: i === 3 ? 'best_effort' as const : 'at_least_once' as const,
    options: { path: binding }, format: null,
    secrets: { token: { key: 'reference', resolver: 'environment' as const } },
  })),
});

const connection = (side: string) => ({
  source: side, sourceHandle: 'late', target: side + '_route', targetHandle: 'input',
});

it('keeps late bindings when removing a SQL alias disconnects its incoming edge', () => {
  const base = project();
  const original = { ...base, graph: { ...base.graph, nodes: base.graph.nodes.map((node) =>
    node.id === 'left_route' ? { ...node, operator: {
      kind: 'sql' as const, aliases: ['input'], query: 'SELECT * FROM input', udfs: [],
    } } : node) } };
  const connected = connectProject(connectProject(original, connection('left')), connection('right'));
  const changed = editSqlInputAliases(connected, 'left_route', { type: 'remove', alias: 'input' });
  expect(changed.sinks).toEqual(original.sinks.map((sink) =>
    sink.binding === 'left.late' ? { ...sink, binding: 'late' } : sink));
  expect(lateOutputInUse(changed, at(changed.graph.nodes))).toBe(true);
});

it('restores sink identities and delivery when both late edges are disconnected', () => {
  const original = project();
  const connected = connectProject(connectProject(original, connection('left')), connection('right'));
  const leftDisconnected = withProjectGraph(connected, {
    ...connected.graph,
    edges: connected.graph.edges.filter((edge) => edge.source_node !== 'left'),
  });
  expect(leftDisconnected.sinks).toEqual(original.sinks.map((sink) =>
    sink.binding === 'left.late' ? { ...sink, binding: 'late' } : sink));
  const disconnected = withProjectGraph(leftDisconnected, { ...leftDisconnected.graph, edges: [] });
  expect(disconnected.sinks).toEqual(original.sinks);
  expect(lateOutputInUse(disconnected, at(disconnected.graph.nodes))).toBe(true);
  expect(lateOutputInUse(disconnected, at(disconnected.graph.nodes, 1))).toBe(true);
});

it('preserves an untouched late sink by identity when its physical name changes', () => {
  const original = project();
  const before = structuredClone(original);
  const right = at(original.graph.nodes, 1);
  expect(lateOutputInUse(original, right)).toBe(true);
  const changed = connectProject(original, connection('left'));
  expect(externalOutputs(changed.graph).find((port) => port.nodeId === 'right' && port.port === 'late')?.binding)
    .toBe('late');
  expect(changed.sinks).toEqual(original.sinks.map((sink) =>
    sink.binding === 'right.late' ? { ...sink, binding: 'late' } : sink));
  expect(lateOutputInUse(changed, right)).toBe(true);
  expect(original).toEqual(before);
});

it('keeps connected sink settings and explicitly requests rebinding both late consumers', () => {
  const original = project();
  const leftConnected = connectProject(original, connection('left'));
  const connected = connectProject(leftConnected, connection('right'));
  expect(connected.sinks).toEqual(original.sinks);
  expect(lateOutputInUse(connected, at(connected.graph.nodes))).toBe(true);
  expect(lateOutputInUse(connected, at(connected.graph.nodes, 1))).toBe(true);
  const onChange = vi.fn();
  render(<StreamConfigEditor project={connected} connectors={[]} onChange={onChange} />);
  expect(screen.getByText('Rebind sinks without an external output: left.late, right.late')).toBeInTheDocument();
  fireEvent.change(at(screen.getAllByLabelText('Graph output'), 3), { target: { value: 'right_route.output' } });
  expect(at(onChange.mock.calls)[0].sinks).toEqual(connected.sinks.map((sink, i) =>
    i === 3 ? { ...sink, binding: 'right_route.output' } : sink));
});

it('protects a consumer with a stale qualified late binding until it is rebound', () => {
  const original = project();
  const connected = connectProject(original, connection('left'));
  const stale = { ...connected, sinks: original.sinks };
  const right = at(stale.graph.nodes, 1);
  expect(lateOutputInUse(stale, right)).toBe(true);
  const onChange = vi.fn();
  render(<NodeInspector node={right} arrowTypes={[]} udfs={[]} onChange={onChange}
    onSqlAliasEdit={vi.fn()} onDelete={vi.fn()} streamMode lateOutputSupported
    lateOutputInUse={lateOutputInUse(stale, right)} />);
  expect(screen.getByLabelText('Late row policy')).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Late row policy'), { target: { value: 'drop' } });
  expect(onChange).not.toHaveBeenCalled();
});
