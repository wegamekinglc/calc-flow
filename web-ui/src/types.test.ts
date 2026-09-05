import { describe, expect, it } from 'vitest';

import { at, blankProject } from './types';

describe('v3 project transforms', () => {
  it('creates a semantically valid portable v3 expression project', () => {
    const project = blankProject();

    expect(project.format_version).toBe(3);
    expect(project).not.toHaveProperty('$defs');
    expect(project.id).toMatch(/^[A-Za-z][A-Za-z0-9_-]*$/);
    expect(at(project.graph.nodes).operator).toEqual({
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
});
