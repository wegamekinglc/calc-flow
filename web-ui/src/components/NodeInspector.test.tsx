import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { blankProject } from '../types';
import { NodeInspector } from './NodeInspector';

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
});
