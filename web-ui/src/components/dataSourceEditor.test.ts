import { describe, expect, it } from 'vitest';

import type { DataSourceSpec } from '../types';
import {
  createDataSourceDrafts,
  materializeDataSources,
  nextDataSource,
} from './dataSourceEditor';

const sources: DataSourceSpec[] = [
  { id: 'left', input: 'left_source', format: 'inline_json', data: [{ value: 1 }] },
  { id: 'right', input: 'right_source', format: 'csv', data: 'value\n2\n' },
];

describe('data source editor helpers', () => {
  it('creates stable drafts without mutating source data', () => {
    const keys = ['key-left', 'key-right'][Symbol.iterator]();
    const drafts = createDataSourceDrafts(sources, () => keys.next().value!);

    expect(drafts).toEqual([
      { key: 'key-left', dataText: '[\n  {\n    "value": 1\n  }\n]', error: null },
      { key: 'key-right', dataText: 'value\n2\n', error: null },
    ]);
    expect(sources[0].data).toEqual([{ value: 1 }]);
  });

  it('selects the first source and input names that are both unused', () => {
    expect(nextDataSource([
      ...sources,
      { id: 'source_1', input: 'input_2', format: 'inline_json', data: [] },
    ])).toEqual({ id: 'source_3', input: 'input_3', format: 'inline_json', data: [] });
  });

  it('materializes all supported formats and clears prior errors', () => {
    const allSources: DataSourceSpec[] = [
      sources[0],
      { id: 'json', input: 'json_input', format: 'json', data: '' },
      sources[1],
      { id: 'arrow', input: 'arrow_input', format: 'arrow_ipc', data: '' },
    ];
    const drafts = createDataSourceDrafts(allSources, () => crypto.randomUUID()).map(
      (draft, index) => ({
        ...draft,
        dataText: ['[{"value":3}]', '{"value":4}', 'value\n5\n', 'YXJyb3c='][index],
        error: 'old error',
      }),
    );

    const result = materializeDataSources(allSources, drafts);

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.sources.map(({ data }) => data)).toEqual([
      [{ value: 3 }],
      '{"value":4}',
      'value\n5\n',
      'YXJyb3c=',
    ]);
    expect(result.drafts.every(({ error }) => error === null)).toBe(true);
  });

  it('marks only invalid inline JSON and returns no stale sources', () => {
    const drafts = createDataSourceDrafts(sources, () => crypto.randomUUID());
    const result = materializeDataSources(sources, [
      { ...drafts[0], dataText: '[{' },
      drafts[1],
    ]);

    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.message).toBe('Data source left contains invalid inline JSON');
    expect(result.drafts[0].error).toBe('Invalid inline JSON');
    expect(result.drafts[1].error).toBeNull();
  });
});
