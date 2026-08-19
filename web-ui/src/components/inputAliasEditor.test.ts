import { describe, expect, it } from 'vitest';

import { blankProject } from '../types';
import {
  editSqlInputAliases,
  nextInputAlias,
  validateInputAlias,
} from './inputAliasEditor';

const sqlProject = () => {
  const project = blankProject();
  return {
    ...project,
    graph: {
      ...project.graph,
      nodes: [
        {
          id: 'join',
          operator: {
            kind: 'sql' as const,
            query: 'SELECT * FROM left JOIN right USING (id)',
            aliases: ['left', 'right'],
            udfs: [],
          },
          input_ports: [
            { name: 'left', kind: 'table' as const, required: true, schema: [] },
            {
              name: 'right',
              kind: 'table' as const,
              required: true,
              schema: [{ name: 'id', data_type: 'int64', nullable: false }],
            },
          ],
          output_ports: [],
          position: { x: 0, y: 0 },
        },
      ],
      edges: [
        {
          source_node: 'a',
          source_port: 'output',
          target_node: 'join',
          target_port: 'left',
        },
        {
          source_node: 'b',
          source_port: 'output',
          target_node: 'join',
          target_port: 'right',
        },
      ],
    },
  };
};

describe('SQL input aliases', () => {
  it('generates the first available deterministic alias', () => {
    expect(nextInputAlias([])).toBe('input');
    expect(nextInputAlias(['input', 'input_2'])).toBe('input_3');
  });

  it('validates committed alias text without changing the saved value', () => {
    expect(validateInputAlias('', 'left', ['left', 'right']))
      .toBe('Input alias is required');
    expect(validateInputAlias(' right ', 'left', ['left', 'right']))
      .toBe('Input aliases must be unique');
    expect(validateInputAlias(' lhs ', 'left', ['left', 'right'])).toBeNull();
  });

  it('renames the operator alias, schema port, and matching incoming edge', () => {
    const project = sqlProject();
    const updated = editSqlInputAliases(project, 'join', {
      type: 'rename',
      alias: 'right',
      nextAlias: 'rhs',
    });

    expect(updated.graph.nodes[0].operator).toMatchObject({
      aliases: ['left', 'rhs'],
    });
    expect(updated.graph.nodes[0].input_ports[1]).toEqual({
      name: 'rhs',
      kind: 'table',
      required: true,
      schema: [{ name: 'id', data_type: 'int64', nullable: false }],
    });
    expect(updated.graph.edges.map((edge) => edge.target_port))
      .toEqual(['left', 'rhs']);
    expect(project.graph.nodes[0].input_ports[1].name).toBe('right');
    expect(project.graph.edges[1].target_port).toBe('right');
  });

  it('removes only the selected alias, port, and incoming edge', () => {
    const project = sqlProject();
    const updated = editSqlInputAliases(project, 'join', {
      type: 'remove',
      alias: 'right',
    });

    expect(updated.graph.nodes[0].operator).toMatchObject({ aliases: ['left'] });
    expect(updated.graph.nodes[0].input_ports.map((port) => port.name))
      .toEqual(['left']);
    expect(updated.graph.edges.map((edge) => edge.target_port)).toEqual(['left']);
  });

  it('adds a derived alias without materializing explicit ports', () => {
    const project = sqlProject();
    const derived = {
      ...project,
      graph: {
        ...project.graph,
        nodes: project.graph.nodes.map((node) => ({ ...node, input_ports: [] })),
      },
    };
    const updated = editSqlInputAliases(derived, 'join', { type: 'add' });

    expect(updated.graph.nodes[0].operator).toMatchObject({
      aliases: ['left', 'right', 'input'],
    });
    expect(updated.graph.nodes[0].input_ports).toEqual([]);
  });

  it('rejects an invalid rename without cloning the project', () => {
    const project = sqlProject();

    expect(editSqlInputAliases(project, 'join', {
      type: 'rename',
      alias: 'right',
      nextAlias: 'left',
    })).toBe(project);
  });
});
