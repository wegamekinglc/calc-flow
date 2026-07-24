# Studio Multi-Source Editor Implementation Plan

> **Historical status:** Implemented and merged in PR #17. Unchecked boxes
> preserve the original execution plan; they are not current pending work.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Studio's single preview override with a persistent multi-source editor and prove that a true two-external-input graph saves, validates, and runs.

**Architecture:** Keep project `data_sources` canonical for source metadata and keep only textarea text/error state in parallel stable-key drafts. A focused component renders the cards; pure helpers create, materialize, and validate drafts. Every persistence path materializes one immutable project value, and Run sends `{}` so the backend consumes the complete saved source set.

**Tech Stack:** React 19, TypeScript 5.9, Vitest and Testing Library, Playwright, Vite, existing FastAPI `/api/v2` client.

## Global Constraints

- Make no backend route, model, OpenAPI, generated API type, Rust core, or Python binding changes.
- Support exactly the saved-source formats `inline_json`, `json`, `csv`, and `arrow_ipc`.
- Keep React state immutable and use functional updates whenever new state depends on previous state.
- Preserve temporarily invalid inline JSON in its own editor card, but block every action that would persist stale source data.
- Save, Validate, Run, and checkpoint inspection must pass the same materialized project value directly to persistence; never depend on a pending React state update.
- Run must submit `{}` and let the backend use the saved project sources; do not construct browser preview overrides.
- Runtime project data, `.calc-flow-web`, logs, generated native libraries, build output, and other ignored artifacts must not be committed.
- Every behavior change begins with a focused failing test and recorded expected failure.

---

## File Structure

- Create `web-ui/src/components/dataSourceEditor.ts`: pure format, draft, naming, and materialization helpers.
- Create `web-ui/src/components/dataSourceEditor.test.ts`: unit tests for deterministic names, immutable draft construction, and format materialization.
- Create `web-ui/src/components/DataSourceEditor.tsx`: accessible stateless list/card UI.
- Create `web-ui/src/components/DataSourceEditor.test.tsx`: component interaction and targeted-file tests.
- Modify `web-ui/src/types.ts`: export the generated `DataSourceSpec` alias used by the helper and component boundary.
- Modify `web-ui/src/App.tsx`: own source drafts, synchronize project transitions, materialize persistence actions, load per-source files, and submit an empty run request.
- Modify `web-ui/src/App.test.tsx`: replace the single-preview contract test and add invalid-data/project-transition coverage.
- Modify `web-ui/src/styles.css`: replace the single sample editor styling with a bounded, scrollable card list.
- Modify `web-ui/e2e/studio.spec.ts`: exercise add/remove plus a real two-source saved-data run.
- Runtime-only: update `two_upstream_demo` through `/api/v2`, restart the managed Studio, and verify the live result.

### Task 1: Pure Source Draft and Materialization Model

**Files:**
- Modify: `web-ui/src/types.ts`
- Create: `web-ui/src/components/dataSourceEditor.ts`
- Create: `web-ui/src/components/dataSourceEditor.test.ts`

**Interfaces:**
- Consumes: `DataSourceSpec = GeneratedProjectDocument['$defs']['DataSourceSpec']` and `JSONValue` from `web-ui/src/types.ts`.
- Produces: `DATA_SOURCE_FORMATS`, `DataSourceFormat`, `DataSourceDraft`, `createDataSourceDrafts`, `nextDataSource`, and `materializeDataSources` for Tasks 2 and 3.

- [ ] **Step 1: Write the failing helper tests**

Create `web-ui/src/components/dataSourceEditor.test.ts` exactly as follows:

```ts
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
```

- [ ] **Step 2: Run the helper test and record the expected failure**

Run:

```bash
cd web-ui
npx vitest run src/components/dataSourceEditor.test.ts
```

Expected: FAIL because `./dataSourceEditor` and the exported `DataSourceSpec` do not exist.

- [ ] **Step 3: Export the generated data-source type**

Add beside the other generated aliases in `web-ui/src/types.ts`:

```ts
export type DataSourceSpec = GeneratedProjectDocument['$defs']['DataSourceSpec'];
```

- [ ] **Step 4: Implement the pure helper module**

Create `web-ui/src/components/dataSourceEditor.ts` with these exact contracts and behavior:

```ts
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
  if (source.format === 'inline_json') return JSON.stringify(source.data, null, 2) ?? '';
  return typeof source.data === 'string'
    ? source.data
    : JSON.stringify(source.data, null, 2) ?? '';
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
    const label = sources[first]?.id || `#${first + 1}`;
    return {
      ok: false,
      message: `Data source ${label} contains invalid inline JSON`,
      drafts: nextDrafts,
    };
  }
  return { ok: true, sources: materialized, drafts: nextDrafts };
};
```

- [ ] **Step 5: Run focused tests and type-aware build**

Run:

```bash
cd web-ui
npx vitest run src/components/dataSourceEditor.test.ts
npm run build
```

Expected: four helper tests PASS and the TypeScript/Vite build succeeds.

- [ ] **Step 6: Commit the pure model**

```bash
git add web-ui/src/types.ts web-ui/src/components/dataSourceEditor.ts web-ui/src/components/dataSourceEditor.test.ts
git commit -m "feat: add Studio data source editor model"
```

### Task 2: Accessible Multi-Source Card Component

**Files:**
- Create: `web-ui/src/components/DataSourceEditor.tsx`
- Create: `web-ui/src/components/DataSourceEditor.test.tsx`
- Modify: `web-ui/src/styles.css`

**Interfaces:**
- Consumes: `DataSourceSpec`, `DataSourceDraft`, `DataSourceFormat`, and `DATA_SOURCE_FORMATS` from Task 1.
- Produces: `DataSourceEditor` with indexed immutable callbacks for App integration in Task 3.

- [ ] **Step 1: Write failing component interaction tests**

Create `web-ui/src/components/DataSourceEditor.test.tsx` exactly as follows:

```tsx
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import {
  DataSourceEditor,
  type DataSourceEditorProps,
} from './DataSourceEditor';

describe('DataSourceEditor', () => {
  it('dispatches changes to the addressed source card', () => {
    const props = {
      sources: [
        { id: 'left', input: 'left_source', format: 'inline_json', data: [] },
        { id: 'right', input: 'right_source', format: 'csv', data: '' },
      ],
      drafts: [
        { key: 'left-key', dataText: '[]', error: null },
        { key: 'right-key', dataText: 'value\n2\n', error: 'CSV problem' },
      ],
      busy: false,
      onAdd: vi.fn(),
      onRemove: vi.fn(),
      onFieldChange: vi.fn(),
      onDataChange: vi.fn(),
      onLoadFile: vi.fn(),
    } satisfies DataSourceEditorProps;

    render(<DataSourceEditor {...props} />);
    expect(screen.getByLabelText('Source ID 1')).toHaveValue('left');
    expect(screen.getByLabelText('Graph input 2')).toHaveValue('right_source');
    expect(screen.getByLabelText('Data 2')).toHaveAttribute('aria-invalid', 'true');
    fireEvent.change(screen.getByLabelText('Graph input 2'), {
      target: { value: 'prices' },
    });
    expect(props.onFieldChange).toHaveBeenCalledWith(1, 'input', 'prices');
    fireEvent.click(screen.getByRole('button', { name: 'Remove source 1' }));
    expect(props.onRemove).toHaveBeenCalledWith(0);
    fireEvent.click(screen.getByRole('button', { name: 'Add data source' }));
    expect(props.onAdd).toHaveBeenCalledOnce();
    const file = new File(['value\n3\n'], 'right.csv', { type: 'text/csv' });
    fireEvent.change(screen.getByLabelText('Load file 2'), {
      target: { files: [file] },
    });
    expect(props.onLoadFile).toHaveBeenCalledWith(1, file);
  });

  it('explains how to configure an empty source list', () => {
    render(
      <DataSourceEditor
        sources={[]}
        drafts={[]}
        busy={false}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
        onFieldChange={vi.fn()}
        onDataChange={vi.fn()}
        onLoadFile={vi.fn()}
      />,
    );

    expect(screen.getByText('Add one data source for every external graph input.'))
      .toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add data source' })).toBeEnabled();
  });
});
```

- [ ] **Step 2: Run the component test and record the expected failure**

Run:

```bash
cd web-ui
npx vitest run src/components/DataSourceEditor.test.tsx
```

Expected: FAIL because `DataSourceEditor.tsx` does not exist.

- [ ] **Step 3: Implement the stateless component**

Create `web-ui/src/components/DataSourceEditor.tsx` with this complete stateless
implementation:

```tsx
import type { DataSourceSpec } from '../types';
import {
  DATA_SOURCE_FORMATS,
  type DataSourceDraft,
  type DataSourceFormat,
} from './dataSourceEditor';

export interface DataSourceEditorProps {
  readonly sources: readonly DataSourceSpec[];
  readonly drafts: readonly DataSourceDraft[];
  readonly busy: boolean;
  readonly onAdd: () => void;
  readonly onRemove: (index: number) => void;
  readonly onFieldChange: (
    index: number,
    field: 'id' | 'input' | 'format',
    value: string,
  ) => void;
  readonly onDataChange: (index: number, value: string) => void;
  readonly onLoadFile: (index: number, file: File) => void;
}

const SOURCE_FORMAT_LABELS: Record<DataSourceFormat, string> = {
  inline_json: 'Inline JSON',
  json: 'JSON / JSONL',
  csv: 'CSV',
  arrow_ipc: 'Arrow IPC',
};

const SOURCE_FILE_ACCEPT: Record<DataSourceFormat, string> = {
  inline_json: '.json,application/json',
  json: '.json,.jsonl,.ndjson,application/json,application/x-ndjson',
  csv: '.csv,text/csv',
  arrow_ipc: '.arrow,.ipc,application/vnd.apache.arrow.file,application/vnd.apache.arrow.stream',
};

export function DataSourceEditor({
  sources,
  drafts,
  busy,
  onAdd,
  onRemove,
  onFieldChange,
  onDataChange,
  onLoadFile,
}: DataSourceEditorProps) {
  return (
    <section className="data-source-editor" aria-labelledby="data-source-heading">
      <div className="data-source-heading">
        <span className="eyebrow" id="data-source-heading">Data sources</span>
        <button className="text-button" type="button" disabled={busy} onClick={onAdd}>
          Add data source
        </button>
      </div>
      {!sources.length && (
        <p className="data-source-empty">
          Add one data source for every external graph input.
        </p>
      )}
      <div className="data-source-list">
        {sources.map((source, index) => {
          const number = index + 1;
          const draft = drafts[index];
          const errorId = `data-source-error-${draft?.key ?? number}`;
          const format = source.format as DataSourceFormat;
          return (
            <article className="data-source-card" key={draft?.key ?? `${source.id}-${number}`}>
              <header>
                <strong>{source.id || `Source ${number}`}</strong>
                <button
                  className="icon-button"
                  type="button"
                  disabled={busy}
                  aria-label={`Remove source ${number}`}
                  onClick={() => onRemove(index)}
                >
                  Remove
                </button>
              </header>
              <label>
                Source ID {number}
                <input
                  disabled={busy}
                  value={source.id}
                  onChange={(event) => onFieldChange(index, 'id', event.target.value)}
                />
              </label>
              <label>
                Graph input {number}
                <input
                  disabled={busy}
                  value={source.input}
                  onChange={(event) => onFieldChange(index, 'input', event.target.value)}
                />
              </label>
              <label>
                Format {number}
                <select
                  disabled={busy}
                  value={source.format}
                  onChange={(event) => onFieldChange(index, 'format', event.target.value)}
                >
                  {DATA_SOURCE_FORMATS.map((candidate) => (
                    <option value={candidate} key={candidate}>
                      {SOURCE_FORMAT_LABELS[candidate]}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Data {number}
                <textarea
                  rows={7}
                  disabled={busy}
                  value={draft?.dataText ?? ''}
                  aria-invalid={Boolean(draft?.error)}
                  aria-describedby={draft?.error ? errorId : undefined}
                  onChange={(event) => onDataChange(index, event.target.value)}
                />
              </label>
              {draft?.error && (
                <p className="data-source-error" id={errorId}>{draft.error}</p>
              )}
              <label className="file-button">
                Load file
                <input
                  type="file"
                  disabled={busy}
                  aria-label={`Load file ${number}`}
                  accept={SOURCE_FILE_ACCEPT[format]}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) onLoadFile(index, file);
                    event.target.value = '';
                  }}
                />
              </label>
            </article>
          );
        })}
      </div>
    </section>
  );
}
```

The component only invokes callbacks; it does not mutate sources/drafts or own
duplicate state. The file input resets after dispatch so the same file can be
selected again.

- [ ] **Step 4: Add bounded sidebar card styling**

Replace `.sample-editor` with styles for:

```css
.data-source-editor { display: grid; gap: 9px; margin-top: 22px; padding-top: 17px; border-top: 1px solid var(--line); }
.data-source-heading { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.data-source-list { display: grid; gap: 9px; max-height: 430px; overflow: auto; padding-right: 3px; }
.data-source-card { display: grid; gap: 8px; padding: 9px; border: 1px solid var(--line); border-radius: 9px; background: #091815; }
.data-source-card header { display: flex; align-items: center; justify-content: space-between; gap: 7px; }
.data-source-card header strong { overflow: hidden; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
.data-source-card textarea { min-height: 92px; }
.data-source-error { margin: -2px 0 0; color: var(--danger); font-size: 9px; }
.data-source-empty { margin: 0; color: var(--muted); font-size: 9px; line-height: 1.45; }
```

Reuse `.text-button`, `.icon-button`, and `.file-button`; do not introduce a new
global button system.

- [ ] **Step 5: Run focused component verification**

```bash
cd web-ui
npx vitest run src/components/DataSourceEditor.test.tsx src/components/dataSourceEditor.test.ts
npm run build
```

Expected: all focused tests PASS and the production frontend builds.

- [ ] **Step 6: Commit the component**

```bash
git add web-ui/src/components/DataSourceEditor.tsx web-ui/src/components/DataSourceEditor.test.tsx web-ui/src/styles.css
git commit -m "feat: add Studio multi-source controls"
```

### Task 3: App Persistence and Saved-Source Run Integration

**Files:**
- Modify: `web-ui/src/App.tsx`
- Modify: `web-ui/src/App.test.tsx`

**Interfaces:**
- Consumes: `DataSourceEditor` and all Task 1 helpers.
- Produces: synchronized project/draft lifecycle and saved-source execution for Task 4.

- [ ] **Step 1: Replace the single-preview contract test with a failing two-source test**

In `web-ui/src/App.test.tsx`, create a loaded project containing two data sources
and mock catalog, list, load, PUT, refreshed list, and run requests. Assert:

```ts
expect(await screen.findByLabelText('Source ID 1')).toHaveValue('left');
expect(screen.getByLabelText('Source ID 2')).toHaveValue('right');
fireEvent.change(screen.getByLabelText('Data 1'), {
  target: { value: '[{"id":1,"value":4}]' },
});
fireEvent.click(screen.getByRole('button', { name: /Run preview/ }));

const saveCall = fetchMock.mock.calls.find(
  ([path, init]) => String(path).endsWith('/projects/two_source') && init?.method === 'PUT',
);
expect(JSON.parse(String(saveCall?.[1]?.body)).data_sources).toEqual([
  { id: 'left', input: 'left_source', format: 'inline_json', data: [{ id: 1, value: 4 }] },
  { id: 'right', input: 'right_source', format: 'csv', data: 'id,adjustment\n1,10\n' },
]);
const runCall = fetchMock.mock.calls.find(
  ([path]) => String(path).endsWith('/projects/two_source/runs'),
);
expect(JSON.parse(String(runCall?.[1]?.body))).toEqual({});
```

Keep the `FakeEventSource` cleanup assertion from the existing preview test.

- [ ] **Step 2: Add failing invalid-draft and project-transition tests**

Add one test that fills `Data 1` with `[{`, clicks Save, Validate, Run preview,
and Inspect, then asserts `Data source sample contains invalid inline JSON`,
`aria-invalid="true"`, and no create/save/validate/run/checkpoint request.

Add one test with two project summaries. Load the first project, edit its draft,
select the second project, and assert `Source ID 1` and `Data 1` reflect only the
second document. This proves load/switch replaces drafts instead of matching by
editable IDs.

- [ ] **Step 3: Run the App tests and record the expected failures**

```bash
cd web-ui
npx vitest run src/App.test.tsx
```

Expected: FAIL because the single preview UI remains, Run submits `inputs.input`,
and project transitions do not synchronize drafts.

- [ ] **Step 4: Replace scalar preview state with per-source drafts**

In `web-ui/src/App.tsx`:

- remove `sampleFormat`, `sampleInputName`, `sampleData`, and the now-unused
  `JSONValue` import;
- import `DataSourceEditor`, `createDataSourceDrafts`, `materializeDataSources`,
  `nextDataSource`, `DataSourceFormat`, and `DataSourceDraft`;
- initialize drafts from the initial blank project's `data_sources`;
- add `replaceEditableProject(next, isPersisted)` that updates project, drafts,
  persisted state, selected node, validation, run, and checkpoint together; and
- call that replacement helper from initial load, New, import, explicit project
  load, and post-delete fallback.

Use these immutable callback shapes:

```ts
const addDataSource = () => {
  const source = nextDataSource(project.data_sources);
  updateProject((current) => ({
    ...current,
    data_sources: [...current.data_sources, source],
  }));
  setSourceDrafts((current) => [
    ...current,
    ...createDataSourceDrafts([source]),
  ]);
};

const removeDataSource = (index: number) => {
  updateProject((current) => ({
    ...current,
    data_sources: current.data_sources.filter((_, currentIndex) => currentIndex !== index),
  }));
  setSourceDrafts((current) => current.filter((_, currentIndex) => currentIndex !== index));
};

const updateDataSourceField = (
  index: number,
  field: 'id' | 'input' | 'format',
  value: string,
) => updateProject((current) => ({
  ...current,
  data_sources: current.data_sources.map((source, currentIndex) =>
    currentIndex === index
      ? { ...source, [field]: field === 'format' ? value as DataSourceFormat : value }
      : source,
  ),
}));

const updateDataSourceData = (index: number, dataText: string) => {
  setSourceDrafts((current) => current.map((draft, currentIndex) =>
    currentIndex === index ? { ...draft, dataText, error: null } : draft,
  ));
  setValidation(null);
  setRun(null);
};
```

- [ ] **Step 5: Materialize one explicit project per persistence action**

Change persistence to accept an explicit argument:

```ts
const persistProject = async (nextProject: EditableProject): Promise<ProjectDocument> => {
  const saved = persisted
    ? await api.saveProject(nextProject)
    : await api.createProject(nextProject);
  setProject(saved);
  setPersisted(true);
  await refreshProjects();
  return saved;
};

const prepareProject = (): EditableProject | null => {
  const materialized = materializeDataSources(project.data_sources, sourceDrafts);
  setSourceDrafts(materialized.drafts);
  if (!materialized.ok) {
    setMessage(materialized.message);
    return null;
  }
  const prepared = { ...project, data_sources: materialized.sources };
  setProject(prepared);
  return prepared;
};
```

Save, Validate, Run, and Inspect each call `prepareProject()` first, return early
when it is null, and pass the returned object directly to `persistProject`.
Execute then calls:

```ts
const submitted = await api.runProject(saved.id, {});
```

This explicit parameter is mandatory: do not call `setProject(prepared)` and
then read `project` in the same event turn.

- [ ] **Step 6: Add targeted per-source file loading and render the component**

Replace `loadSampleFile` with:

```ts
const loadDataSourceFile = async (index: number, file: File) => {
  try {
    const format = project.data_sources[index]?.format;
    const dataText = format === 'arrow_ipc'
      ? await fileToBase64(file)
      : await file.text();
    updateDataSourceData(index, dataText);
  } catch (error) {
    setMessage((error as Error).message);
  }
};
```

Replace the old `.sample-editor` JSX with `DataSourceEditor`, passing all source,
draft, busy, add/remove/edit/data/file callbacks. Confirm no reference to
`sampleFormat`, `sampleInputName`, `sampleData`, or `browser-preview` remains.

- [ ] **Step 7: Run App and complete frontend unit verification**

```bash
cd web-ui
npx vitest run src/App.test.tsx
npm test
npm run build
```

Expected: App tests, all frontend unit tests, and the production build PASS.

- [ ] **Step 8: Commit App integration**

```bash
git add web-ui/src/App.tsx web-ui/src/App.test.tsx
git commit -m "feat: run Studio projects with saved data sources"
```

### Task 4: Browser-Level Two-Source Workflow

**Files:**
- Modify: `web-ui/e2e/studio.spec.ts`

**Interfaces:**
- Consumes: the multi-source UI and saved-source execution from Tasks 1-3.
- Produces: browser evidence that two external inputs reach one downstream SQL join.

- [ ] **Step 1: Add a failing two-source browser test**

Define a complete `twoSourceProject` value in `studio.spec.ts` with:

```ts
const twoSourceProject = {
  format_version: 2,
  id: 'two_source_e2e',
  name: 'Two source E2E',
  description: 'Two independent saved sources join downstream.',
  pipeline: {
    name: 'Two source pipeline',
    nodes: [
      {
        id: 'left_branch',
        operator: {
          kind: 'sql',
          query: 'SELECT id, value * 2 AS left_value FROM left_source',
          aliases: ['left_source'],
          udfs: [],
        },
        input_ports: [], output_ports: [], position: { x: 80, y: 80 },
      },
      {
        id: 'right_branch',
        operator: {
          kind: 'sql',
          query: 'SELECT id, adjustment AS right_value FROM right_source',
          aliases: ['right_source'],
          udfs: [],
        },
        input_ports: [], output_ports: [], position: { x: 80, y: 280 },
      },
      {
        id: 'join_result',
        operator: {
          kind: 'sql',
          query: 'SELECT l.id, l.left_value, r.right_value, l.left_value + r.right_value AS total FROM left l JOIN right r ON l.id = r.id ORDER BY l.id',
          aliases: ['left', 'right'],
          udfs: [],
        },
        input_ports: [], output_ports: [], position: { x: 480, y: 180 },
      },
    ],
    edges: [
      { source_node: 'left_branch', source_port: 'output', target_node: 'join_result', target_port: 'left' },
      { source_node: 'right_branch', source_port: 'output', target_node: 'join_result', target_port: 'right' },
    ],
    datafusion: { batch_size: 8192, target_partitions: 1 },
  },
  data_sources: [
    { id: 'left', input: 'left_source', format: 'inline_json', data: [{ id: 1, value: 3 }, { id: 2, value: 5 }] },
    { id: 'right', input: 'right_source', format: 'inline_json', data: [{ id: 1, adjustment: 10 }, { id: 2, adjustment: 20 }] },
  ],
  run_options: { max_input_bytes: 10485760, max_rows: 100000, timeout_seconds: 30, memory_limit_mb: 512, output_rows: 1000 },
};
```

The test must POST this document before navigation, assert two cards, add and
remove a third card, edit `Data 1` to use value `4`, Save, Validate, Run, and
assert the completed result contains a `total` column and value `18`. GET the
saved project and assert both configured input names remain
`['left_source', 'right_source']`. Delete the project at the end.

- [ ] **Step 2: Run the browser acceptance test**

Stop the managed Studio first so Playwright owns isolated ports:

```bash
./web-ui/scripts/stop_web_ui.sh
cd web-ui
npm run test:e2e
```

Expected: both the existing UDF workflow and the new two-source workflow PASS.
The focused App and component tests in Tasks 2 and 3 already recorded the red
state for the production behavior; this test adds browser-level acceptance
evidence without another production change.

- [ ] **Step 3: Keep selectors semantic and the result assertion exact**

Use label- and role-based Playwright locators. Do not add test-only production
attributes unless semantic labels cannot uniquely identify a control. Keep the
expected calculation at `4 * 2 + 10 = 18`; do not reduce the assertion to status
alone.

- [ ] **Step 4: Run isolated browser verification**

```bash
cd web-ui
npm run test:e2e
```

Expected: existing UDF workflow and new two-source workflow PASS.

- [ ] **Step 5: Commit browser coverage**

```bash
git add web-ui/e2e/studio.spec.ts
git commit -m "test: cover Studio multi-source workflow"
```

### Task 5: Full Verification, Live Example, and Managed Studio Restart

**Files:**
- Verify unchanged: `schemas/project-v2.schema.json`
- Verify unchanged: `web-ui/openapi.json`
- Verify unchanged: `web-ui/src/api/schema.d.ts`
- Runtime-only update: persisted Studio project `two_upstream_demo`

**Interfaces:**
- Consumes: all implementation tasks.
- Produces: a clean verified branch and a live user-visible two-source example.

- [ ] **Step 1: Run frontend and Studio backend checks**

```bash
cd web-ui
npm ci
npm run sync:api
npm run build
npm test
npm run test:e2e
npm audit --omit=dev
cd backend
uv run --project . --extra dev pytest --cov=calc_flow_studio
```

Expected: every command succeeds and Studio backend coverage remains at or above
85%.

- [ ] **Step 2: Run the repository-required full verification groups**

From the repository root run exactly:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo llvm-cov --workspace --all-features --fail-under-lines 90
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
uv sync --extra dev
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check .
uv run ruff format --check .
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
cargo deny --locked check
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
```

Expected: all commands succeed. After `maturin develop`, remove any generated
`python/calc_flow/_native*.so` only if the command created one, then rerun the
relevant Python import/test from the prepared environment without committing the
generated library.

- [ ] **Step 3: Prove generated contracts and diff hygiene**

```bash
git diff --exit-code -- schemas/project-v2.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts
git diff --check
git status --short
```

Expected: generated contracts have no diff, diff check is silent, and status
contains only the intended committed plan/implementation state with no runtime
artifacts.

- [ ] **Step 4: Build the production frontend and restart managed Studio**

```bash
cd web-ui
npm run build
cd ..
./web-ui/scripts/start_web_ui.sh
curl -fsS http://127.0.0.1:8765/api/v2/catalog >/dev/null
curl -fsS http://127.0.0.1:5173/ >/dev/null
```

Keep the launcher's owning shell/session alive. Expected: both probes succeed at
`http://127.0.0.1:8765` and `http://127.0.0.1:5173`.

- [ ] **Step 5: Restore the live `two_upstream_demo` as a true two-input project**

PUT the same graph shape used by `twoSourceProject`, changing only:

```text
id = two_upstream_demo
name = Two upstream nodes demo
description = Two independent saved sources join in one downstream SQL node.
```

Use source IDs `left` and `right`, graph inputs `left_source` and
`right_source`, and the exact two branch plus join queries from Task 4. Do not
retain the old `source_input` fan-out node.

- [ ] **Step 6: Validate and run the live example through the saved-source path**

```bash
curl -fsS -X POST http://127.0.0.1:8765/api/v2/projects/two_upstream_demo/validate
curl -fsS -X POST -H 'Content-Type: application/json' \
  --data '{}' \
  http://127.0.0.1:8765/api/v2/projects/two_upstream_demo/runs
```

Poll the returned run ID with `GET /api/v2/runs/<id>` until terminal. Expected:
validation returns `valid: true`; run returns `completed`; `join_result.output`
contains two rows with totals `16` and `30`; the saved project GET shows two data
source cards' backing values and no source named `input`.

- [ ] **Step 7: Perform final scope review**

```bash
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
git status --short --branch
```

Expected: separate design and plan commits followed by the focused helper,
component, App, and browser commits; no unrelated or runtime files; clean status.
