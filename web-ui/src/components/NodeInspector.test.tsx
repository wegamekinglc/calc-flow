import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { blankProject } from '../types';
import { NodeInspector } from './NodeInspector';
import { inspectLoweredNode } from './projectInspectionModel';

describe('NodeInspector', () => {
  it('updates a nested v2 expression operator with a trusted UDF reference', () => {
    const node = blankProject().graph.nodes[0];
    const onChange = vi.fn();

    render(
      <NodeInspector
        node={node}
        arrowTypes={['int64']}
        udfs={[
          {
            provider: 'server',
            name: 'double_value',
            version: '1',
            kind: 'data_fusion_scalar',
            signature: { input_types: ['int64'], return_type: 'int64' },
            volatility: 'immutable',
          },
        ]}
        onChange={onChange}
        onSqlAliasEdit={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    fireEvent.change(screen.getByLabelText('Filter expression'), {
      target: { value: 'a > 0' },
    });
    expect(onChange).toHaveBeenLastCalledWith({
      ...node,
      operator: { ...node.operator, filter: 'a > 0' },
    });

    fireEvent.click(screen.getByRole('checkbox', { name: /double_value/i }));
    expect(onChange).toHaveBeenLastCalledWith({
      ...node,
      operator: {
        ...node.operator,
        udfs: [
          {
            provider: 'server',
            name: 'double_value',
            version: '1',
            kind: 'data_fusion_scalar',
          },
        ],
      },
    });
    expect(node.operator).toMatchObject({ filter: null, udfs: [] });
  });

  it('shows only configured schema ports for an external source', () => {
    const base = blankProject().graph.nodes[0];
    const node = {
      ...base,
      input_ports: [],
      output_ports: [
        {
          name: 'rows',
          kind: 'table' as const,
          required: true,
          schema: [],
        },
      ],
      operator: {
        kind: 'external' as const,
        provider: 'trusted',
        name: 'lookup',
        version: '2',
        options: {},
      },
    };

    render(
      <NodeInspector
        node={node}
        arrowTypes={['int64']}
        udfs={[]}
        onChange={vi.fn()}
        onSqlAliasEdit={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(screen.getByText('External provider')).toBeInTheDocument();
    expect(screen.getByText('trusted · lookup · v2')).toBeInTheDocument();
    expect(screen.queryByText('in · input')).not.toBeInTheDocument();
    expect(screen.getByText('out · rows')).toBeInTheDocument();
  });

  it('shows only configured schema ports for an external sink', () => {
    const base = blankProject().graph.nodes[0];
    const node = {
      ...base,
      input_ports: [
        {
          name: 'rows',
          kind: 'table' as const,
          required: true,
          schema: [],
        },
      ],
      output_ports: [],
      operator: {
        kind: 'external' as const,
        provider: 'trusted',
        name: 'publish',
        version: '2',
        options: {},
      },
    };

    render(
      <NodeInspector
        node={node}
        arrowTypes={['int64']}
        udfs={[]}
        onChange={vi.fn()}
        onSqlAliasEdit={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(screen.getByText('in · rows')).toBeInTheDocument();
    expect(screen.queryByText('out · output')).not.toBeInTheDocument();
  });

  it('emits semantic edits for independent SQL alias rows', () => {
    const base = blankProject().graph.nodes[0];
    const node = {
      ...base,
      operator: {
        kind: 'sql' as const,
        query: 'SELECT * FROM left JOIN right USING (id)',
        aliases: ['left', 'right'],
        udfs: [],
      },
    };
    const onSqlAliasEdit = vi.fn();

    render(
      <NodeInspector
        node={node}
        arrowTypes={['int64']}
        udfs={[]}
        onChange={vi.fn()}
        onSqlAliasEdit={onSqlAliasEdit}
        onDelete={vi.fn()}
      />,
    );

    expect(screen.getByLabelText('Input alias 1')).toHaveValue('left');
    expect(screen.getByLabelText('Input alias 2')).toHaveValue('right');

    fireEvent.click(screen.getByRole('button', { name: 'Add input alias' }));
    expect(onSqlAliasEdit).toHaveBeenLastCalledWith({ type: 'add' });

    fireEvent.change(screen.getByLabelText('Input alias 2'), {
      target: { value: 'rhs' },
    });
    fireEvent.keyDown(screen.getByLabelText('Input alias 2'), { key: 'Enter' });
    expect(onSqlAliasEdit).toHaveBeenLastCalledWith({
      type: 'rename',
      alias: 'right',
      nextAlias: 'rhs',
    });

    fireEvent.click(screen.getByRole('button', { name: 'Remove input alias 1' }));
    expect(onSqlAliasEdit).toHaveBeenLastCalledWith({
      type: 'remove',
      alias: 'left',
    });
  });

  it('serializes bounded integers for stream join numeric fields', () => {
    const base = blankProject().graph.nodes[0];
    const node = {
      ...base,
      operator: {
        kind: 'stream_join' as const,
        spec: {
          join_type: 'inner' as const,
          left_keys: ['account_id'],
          right_keys: ['account_id'],
          left_event_time: 'authorized_at',
          right_event_time: 'paid_at',
          bounds: { before_micros: 300_000_000, after_micros: 60_000_000 },
          limits: {
            max_state_rows_per_side: 100_000,
            max_state_bytes_per_side: 134_217_728,
            max_matches_per_input_batch: 1_000_000,
          },
          left_prefix: 'left',
          right_prefix: 'right',
        },
      },
    };
    const onChange = vi.fn();

    render(
      <NodeInspector
        node={node}
        arrowTypes={['int64']}
        udfs={[]}
        onChange={onChange}
        onSqlAliasEdit={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    fireEvent.change(screen.getByLabelText('before micros'), {
      target: { value: '150.75' },
    });
    expect(onChange).toHaveBeenLastCalledWith({
      ...node,
      operator: {
        ...node.operator,
        spec: {
          ...node.operator.spec,
          bounds: { before_micros: 150, after_micros: 60_000_000 },
        },
      },
    });

    fireEvent.change(screen.getByLabelText('max state rows per side'), {
      target: { value: '' },
    });
    expect(onChange).toHaveBeenLastCalledWith({
      ...node,
      operator: {
        ...node.operator,
        spec: {
          ...node.operator.spec,
          limits: {
            max_state_rows_per_side: 1,
            max_state_bytes_per_side: 134_217_728,
            max_matches_per_input_batch: 1_000_000,
          },
        },
      },
    });
  });

  it('shows read-only facts from the strict lowered project', () => {
    const project = blankProject();
    const node = project.graph.nodes[0];

    render(
      <NodeInspector
        node={node}
        inspection={inspectLoweredNode(project, node)}
        arrowTypes={['int64']}
        udfs={[]}
        onChange={vi.fn()}
        onSqlAliasEdit={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    expect(
      screen.getByRole('region', { name: 'Lowered project inspection' }),
    ).toHaveTextContent('strict ProjectDocument v3');
    expect(screen.getByText('total = a + b', { selector: 'code' })).toBeInTheDocument();
    expect(screen.getByText('stateless')).toBeInTheDocument();
    expect(screen.getByText('not required')).toBeInTheDocument();
    expect(screen.getByText('native calc-flow operator')).toBeInTheDocument();
    expect(screen.getByText('none', { selector: '.inspection-empty' })).toBeInTheDocument();
  });
});
