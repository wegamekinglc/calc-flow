import type { DataSourceSpec, JSONValue } from '../types';

export const DATA_SOURCE_FORMATS = ['inline_json', 'json', 'csv', 'arrow_ipc'] as const;
export type DataSourceFormat = typeof DATA_SOURCE_FORMATS[number];

export interface DataSourceDraft {
  readonly key: string;
  readonly dataText: string;
  readonly error: string | null;
}

type Materialization =
  | { readonly ok: true; readonly sources: DataSourceSpec[]; readonly drafts: DataSourceDraft[] }
  | { readonly ok: false; readonly message: string; readonly drafts: DataSourceDraft[] };

const sourceDataText = (source: DataSourceSpec): string => {
  if (source.format === 'inline_json') return JSON.stringify(source.data, null, 2);
  return typeof source.data === 'string'
    ? source.data
    : JSON.stringify(source.data, null, 2);
};

export const createDataSourceDrafts = (
  sources: readonly DataSourceSpec[],
  keyFactory: () => string = () => crypto.randomUUID(),
): DataSourceDraft[] => sources.map((source) => ({
  key: keyFactory(),
  dataText: sourceDataText(source),
  error: null,
}));

export const nextDataSource = (
  sources: readonly DataSourceSpec[],
  index = 1,
): DataSourceSpec => {
  const id = `source_${index}`;
  const input = `input_${index}`;
  return sources.some((source) => source.id === id || source.input === input)
    ? nextDataSource(sources, index + 1)
    : { id, input, format: 'inline_json', data: [] };
};

export const materializeDataSources = (
  sources: readonly DataSourceSpec[],
  drafts: readonly DataSourceDraft[],
): Materialization => {
  const invalid = new Set<number>();
  const materialized = sources.map((source, index) => {
    const text = drafts[index]?.dataText ?? sourceDataText(source);
    if (source.format !== 'inline_json') return { ...source, data: text };
    try {
      return { ...source, data: JSON.parse(text) as JSONValue };
    } catch {
      invalid.add(index);
      return source;
    }
  });
  const nextDrafts = drafts.map((draft, index) => ({
    ...draft,
    error: invalid.has(index) ? 'Invalid inline JSON' : null,
  }));
  const first = invalid.values().next().value as number | undefined;
  if (first !== undefined) {
    const label = sources.at(first)?.id ?? `#${first + 1}`;
    return {
      ok: false,
      message: `Data source ${label} contains invalid inline JSON`,
      drafts: nextDrafts,
    };
  }
  return { ok: true, sources: materialized, drafts: nextDrafts };
};
