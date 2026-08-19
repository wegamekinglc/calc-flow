import { describe, expect, it } from 'vitest';

import { blankProject, updateNodeOperator } from './types';

describe('v3 project transforms', () => {
  it('creates a semantically valid portable v3 expression project', () => {
    const project = blankProject();

    expect(project.format_version).toBe(3);
    expect(project).not.toHaveProperty('$defs');
    expect(project.id).toMatch(/^[A-Za-z][A-Za-z0-9_-]*$/);
    expect(project.graph.nodes[0].operator).toEqual({
      kind: 'expression',
      expression: 'total = a + b',
      select: [],
      filter: null,
      udfs: [],
    });
    expect(project.data_sources).toEqual([
      {
        id: 'sample',
        input: 'input',
        format: 'inline_json',
        data: [
          { a: 1, b: 2 },
          { a: 3, b: 4 },
        ],
      },
    ]);
    expect(project).toMatchObject({
      runtime: { mode: 'batch' },
      sources: [],
      sinks: [],
      state: { root: '.calc-flow-state', retention: 3 },
    });
  });

  it('updates a nested operator without mutating the project or node', () => {
    const project = blankProject();
    const originalNode = project.graph.nodes[0];

    const updated = updateNodeOperator(project, originalNode.id, (operator) =>
      operator.kind === 'expression'
        ? { ...operator, expression: 'total = a - b' }
        : operator,
    );

    expect(updated).not.toBe(project);
    expect(updated.graph).not.toBe(project.graph);
    expect(updated.graph.nodes[0]).not.toBe(originalNode);
    expect(updated.graph.nodes[0].operator).toMatchObject({
      kind: 'expression',
      expression: 'total = a - b',
    });
    expect(originalNode.operator).toMatchObject({
      kind: 'expression',
      expression: 'total = a + b',
    });
  });
});
