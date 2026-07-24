# Studio Input Aliases and Resizable Panels Implementation Plan

> **Historical status:** Implemented and merged in PR #17. Unchecked boxes
> preserve the original execution plan; they are not current pending work.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users edit multiple SQL input aliases safely and resize every horizontally adjacent Studio panel with persisted, accessible controls.

**Architecture:** Keep graph-wide alias invariants in a pure immutable project transition and render aliases through a focused row editor. Add dependency-free resize primitives, pure layout/storage helpers, and one App-owned persisted layout whose safe widths feed the workspace and Results grids through CSS custom properties.

**Tech Stack:** React 19, TypeScript 5.9, React Flow 12, CSS Grid, Vitest, Testing Library, Playwright, browser `localStorage`, and `ResizeObserver`.

## Global Constraints

- Keep this frontend-only: do not change project format v2, backend routes, OpenAPI, generated API types, Cargo, or Python packages.
- Preserve immutable React and project updates; never mutate node, port, edge, source, or layout values supplied by callers.
- Add no npm dependency. The resize implementation is repository-owned and uses pointer events, keyboard events, CSS Grid, and `ResizeObserver`.
- Keep React Flow node-type maps outside render functions.
- Alias commits trim whitespace and reject empty or duplicate values without writing invalid project state.
- Alias rename updates matching explicit input ports and incoming edges; alias removal deletes only its matching port and incoming edges.
- Use `calc-flow-studio:panel-layout:v1` for browser-local layout preferences; do not serialize widths into a project document.
- Main widths: Toolbox defaults to 235 px within 200–420 px; Inspector defaults to 335 px within 280–640 px; Canvas retains at least 480 px.
- Results widths: Metrics defaults to 330 px with a 260 px minimum; Output retains at least 480 px.
- Resize handles use accessible vertical separator semantics, 16 px arrow steps, 48 px Shift+Arrow steps, Home/End bounds, and double-click reset.
- Every behavior change follows red-green-refactor and ends with a focused commit.

---

### Task 1: Pure Graph-Safe SQL Alias Operations

**Files:**
- Create: `web-ui/src/components/inputAliasEditor.ts`
- Test: `web-ui/src/components/inputAliasEditor.test.ts`

**Interfaces:**
- Consumes: `EditableProject` and generated SQL node, port, and edge shapes from `web-ui/src/types.ts`.
- Produces: `SqlInputAliasEdit`, `nextInputAlias`, `validateInputAlias`, and `editSqlInputAliases` for the row editor and `App`.

- [ ] **Step 1: Write failing helper tests**

Create tests that define the public behavior before the helper exists:

```ts
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
    pipeline: {
      ...project.pipeline,
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
        { source_node: 'a', source_port: 'output', target_node: 'join', target_port: 'left' },
        { source_node: 'b', source_port: 'output', target_node: 'join', target_port: 'right' },
      ],
    },
  };
};

describe('SQL input aliases', () => {
  it('generates the first available deterministic alias', () => {
    expect(nextInputAlias([])).toBe('input');
    expect(nextInputAlias(['input', 'input_2'])).toBe('input_3');
  });

  it('validates committed alias text without mutating the saved value', () => {
    expect(validateInputAlias('', 'left', ['left', 'right'])).toBe('Input alias is required');
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

    expect(updated.pipeline.nodes[0].operator).toMatchObject({ aliases: ['left', 'rhs'] });
    expect(updated.pipeline.nodes[0].input_ports[1]).toEqual({
      name: 'rhs',
      kind: 'table',
      required: true,
      schema: [{ name: 'id', data_type: 'int64', nullable: false }],
    });
    expect(updated.pipeline.edges.map((edge) => edge.target_port)).toEqual(['left', 'rhs']);
    expect(project.pipeline.nodes[0].input_ports[1].name).toBe('right');
    expect(project.pipeline.edges[1].target_port).toBe('right');
  });

  it('removes only the selected alias, port, and incoming edge', () => {
    const project = sqlProject();
    const updated = editSqlInputAliases(project, 'join', {
      type: 'remove',
      alias: 'right',
    });

    expect(updated.pipeline.nodes[0].operator).toMatchObject({ aliases: ['left'] });
    expect(updated.pipeline.nodes[0].input_ports.map((port) => port.name)).toEqual(['left']);
    expect(updated.pipeline.edges.map((edge) => edge.target_port)).toEqual(['left']);
  });

  it('adds a derived alias without materializing empty explicit ports', () => {
    const project = sqlProject();
    project.pipeline.nodes[0].input_ports = [];
    const updated = editSqlInputAliases(project, 'join', { type: 'add' });

    expect(updated.pipeline.nodes[0].operator).toMatchObject({
      aliases: ['left', 'right', 'input'],
    });
    expect(updated.pipeline.nodes[0].input_ports).toEqual([]);
  });
});
```

- [ ] **Step 2: Run the tests and record the expected red result**

Run:

```bash
cd web-ui
npm test -- src/components/inputAliasEditor.test.ts
```

Expected: FAIL because `./inputAliasEditor` does not exist.

- [ ] **Step 3: Implement the minimal immutable helper**

Create these exact public types and functions:

```ts
import type { EditableProject, NodeConfig, PortConfig } from '../types';

export type SqlInputAliasEdit =
  | { type: 'add' }
  | { type: 'rename'; alias: string; nextAlias: string }
  | { type: 'remove'; alias: string };

export const nextInputAlias = (aliases: readonly string[]): string => {
  if (!aliases.includes('input')) return 'input';
  let index = 2;
  while (aliases.includes(`input_${index}`)) index += 1;
  return `input_${index}`;
};

export const validateInputAlias = (
  draft: string,
  current: string,
  aliases: readonly string[],
): string | null => {
  const alias = draft.trim();
  if (!alias) return 'Input alias is required';
  if (alias !== current && aliases.includes(alias)) return 'Input aliases must be unique';
  return null;
};

const addExplicitPort = (ports: readonly PortConfig[], alias: string): PortConfig[] =>
  ports.length === 0
    ? []
    : [
        ...ports,
        { name: alias, kind: 'table', required: true, schema: [] },
      ];

export const editSqlInputAliases = (
  project: EditableProject,
  nodeId: string,
  edit: SqlInputAliasEdit,
): EditableProject => {
  const current = project.pipeline.nodes.find((node) => node.id === nodeId);
  if (!current || current.operator.kind !== 'sql') return project;

  const aliases = current.operator.aliases;
  const added = edit.type === 'add' ? nextInputAlias(aliases) : null;
  const renamed = edit.type === 'rename' ? edit.nextAlias.trim() : null;
  if (
    edit.type === 'rename'
    && validateInputAlias(renamed ?? '', edit.alias, aliases) !== null
  ) return project;

  const node = {
    ...current,
    operator: {
      ...current.operator,
      aliases:
        edit.type === 'add'
          ? [...aliases, added!]
          : edit.type === 'rename'
            ? aliases.map((alias) => (alias === edit.alias ? renamed! : alias))
            : aliases.filter((alias) => alias !== edit.alias),
    },
    input_ports:
      edit.type === 'add'
        ? addExplicitPort(current.input_ports, added!)
        : edit.type === 'rename'
          ? current.input_ports.map((port) =>
              port.name === edit.alias ? { ...port, name: renamed! } : port,
            )
          : current.input_ports.filter((port) => port.name !== edit.alias),
  } satisfies NodeConfig;

  const edges = edit.type === 'remove'
    ? project.pipeline.edges.filter(
        (edge) => edge.target_node !== nodeId || edge.target_port !== edit.alias,
      )
    : edit.type === 'rename'
      ? project.pipeline.edges.map((edge) =>
          edge.target_node === nodeId && edge.target_port === edit.alias
            ? { ...edge, target_port: renamed! }
            : edge,
        )
      : project.pipeline.edges;

  return {
    ...project,
    pipeline: {
      ...project.pipeline,
      nodes: project.pipeline.nodes.map((candidate) =>
        candidate.id === nodeId ? node : candidate,
      ),
      edges,
    },
  };
};
```

During implementation, retain the same semantics but remove non-null assertions
when straightforward local narrowing makes the code clearer and keeps Clippy-like
TypeScript lint quality.

- [ ] **Step 4: Verify green and the complete helper suite**

Run:

```bash
cd web-ui
npm test -- src/components/inputAliasEditor.test.ts
```

Expected: all SQL alias helper tests PASS.

- [ ] **Step 5: Commit the helper**

```bash
git add web-ui/src/components/inputAliasEditor.ts web-ui/src/components/inputAliasEditor.test.ts
git commit -m "feat: add graph-safe SQL alias operations"
```

### Task 2: Input Alias Rows and Node Inspector Integration

**Files:**
- Create: `web-ui/src/components/InputAliasEditor.tsx`
- Create: `web-ui/src/components/InputAliasEditor.test.tsx`
- Modify: `web-ui/src/components/NodeInspector.tsx`
- Modify: `web-ui/src/components/NodeInspector.test.tsx`
- Modify: `web-ui/src/App.tsx`
- Modify: `web-ui/src/App.test.tsx`
- Modify: `web-ui/src/styles.css`

**Interfaces:**
- Consumes: `SqlInputAliasEdit`, `validateInputAlias`, and `editSqlInputAliases` from Task 1.
- Produces: `InputAliasEditor` and a `NodeInspector.onSqlAliasEdit(edit)` callback integrated with immutable App project state.

- [ ] **Step 1: Write failing row-editor tests**

Create tests that simulate the keystroke workflow that the comma field cannot
support:

```tsx
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { InputAliasEditor } from './InputAliasEditor';

describe('InputAliasEditor', () => {
  it('adds and commits a second alias as an independent row', () => {
    const onAdd = vi.fn();
    const onRename = vi.fn();
    const { rerender } = render(
      <InputAliasEditor
        aliases={['left']}
        onAdd={onAdd}
        onRename={onRename}
        onRemove={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Add input alias' }));
    expect(onAdd).toHaveBeenCalledOnce();

    rerender(
      <InputAliasEditor
        aliases={['left', 'input']}
        onAdd={onAdd}
        onRename={onRename}
        onRemove={vi.fn()}
      />,
    );
    fireEvent.change(screen.getByLabelText('Input alias 2'), {
      target: { value: 'right' },
    });
    fireEvent.keyDown(screen.getByLabelText('Input alias 2'), { key: 'Enter' });
    expect(onRename).toHaveBeenCalledWith('input', 'right');
  });

  it('keeps invalid drafts local and restores the saved alias on Escape', () => {
    const onRename = vi.fn();
    render(
      <InputAliasEditor
        aliases={['left', 'right']}
        onAdd={vi.fn()}
        onRename={onRename}
        onRemove={vi.fn()}
      />,
    );

    const second = screen.getByLabelText('Input alias 2');
    fireEvent.change(second, { target: { value: 'left' } });
    fireEvent.blur(second);
    expect(screen.getByText('Input aliases must be unique')).toBeInTheDocument();
    expect(onRename).not.toHaveBeenCalled();

    fireEvent.keyDown(second, { key: 'Escape' });
    expect(second).toHaveValue('right');
  });

  it('removes the selected saved row without committing a dirty draft', () => {
    const onRename = vi.fn();
    const onRemove = vi.fn();
    render(
      <InputAliasEditor
        aliases={['left', 'right']}
        onAdd={vi.fn()}
        onRename={onRename}
        onRemove={onRemove}
      />,
    );

    fireEvent.change(screen.getByLabelText('Input alias 2'), {
      target: { value: 'temporary' },
    });
    fireEvent.pointerDown(screen.getByRole('button', { name: 'Remove input alias 2' }));
    fireEvent.click(screen.getByRole('button', { name: 'Remove input alias 2' }));
    expect(onRemove).toHaveBeenCalledWith('right');
    expect(onRename).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run the component test and record red**

Run:

```bash
cd web-ui
npm test -- src/components/InputAliasEditor.test.tsx
```

Expected: FAIL because `InputAliasEditor` does not exist.

- [ ] **Step 3: Implement the stateless list and row-local drafts**

Implement `InputAliasEditor` with this public contract:

```tsx
interface InputAliasEditorProps {
  aliases: readonly string[];
  onAdd: () => void;
  onRename: (alias: string, nextAlias: string) => void;
  onRemove: (alias: string) => void;
}
```

Use a keyed `InputAliasRow` child with `useEffect` to reset its draft when the
saved alias changes. Its `commit()` trims and calls `validateInputAlias`; Enter
calls `commit`, blur calls `commit`, Escape restores `alias` and clears the
error, and the Remove button prevents pointer-down focus transfer before
calling `onRemove(alias)`. Give errors stable IDs and connect them through
`aria-invalid` and `aria-describedby`.

- [ ] **Step 4: Replace the comma input and write failing integration assertions**

Update the SQL branch of `NodeInspector` to render:

```tsx
<InputAliasEditor
  aliases={node.operator.aliases}
  onAdd={() => onSqlAliasEdit({ type: 'add' })}
  onRename={(alias, nextAlias) =>
    onSqlAliasEdit({ type: 'rename', alias, nextAlias })}
  onRemove={(alias) => onSqlAliasEdit({ type: 'remove', alias })}
/>
```

Add `onSqlAliasEdit: (edit: SqlInputAliasEdit) => void` to the inspector props.
Before changing `App`, extend `NodeInspector.test.tsx` with a SQL node containing
`['left', 'right']`; assert two labeled fields and assert Add/Rename/Remove emit
the exact semantic operations. Run the focused test and expect it to fail until
the inspector implementation is complete.

- [ ] **Step 5: Integrate the semantic transition in App**

Pass the selected node ID through one functional update:

```tsx
onSqlAliasEdit={(edit) => {
  const nodeId = selectedNode.id;
  updateProject((current) => editSqlInputAliases(current, nodeId, edit));
}}
```

Add an `App.test.tsx` case that loads a SQL join project, renames `right` to
`rhs` through the visible row, and asserts the next PUT request contains the
renamed operator alias, explicit input port, and incoming edge. The original
loaded fixture must retain `right` after the UI change.

- [ ] **Step 6: Style the rows and verify all focused tests green**

Add compact `.input-alias-editor`, `.input-alias-row`, and
`.input-alias-error` rules. Rows use `minmax(0, 1fr) auto`, inputs keep
`min-width: 0`, and the Add action remains visible below the list.

Run:

```bash
cd web-ui
npm test -- \
  src/components/inputAliasEditor.test.ts \
  src/components/InputAliasEditor.test.tsx \
  src/components/NodeInspector.test.tsx \
  src/App.test.tsx
```

Expected: all focused alias tests PASS and no React act or accessibility
warnings are printed.

- [ ] **Step 7: Commit the alias UI**

```bash
git add \
  web-ui/src/components/InputAliasEditor.tsx \
  web-ui/src/components/InputAliasEditor.test.tsx \
  web-ui/src/components/NodeInspector.tsx \
  web-ui/src/components/NodeInspector.test.tsx \
  web-ui/src/App.tsx \
  web-ui/src/App.test.tsx \
  web-ui/src/styles.css
git commit -m "feat: edit Studio SQL aliases as rows"
```

### Task 3: Persistent Panel Layout Model

**Files:**
- Create: `web-ui/src/components/panelLayout.ts`
- Test: `web-ui/src/components/panelLayout.test.ts`

**Interfaces:**
- Consumes: browser `Storage`, measured workspace width, and measured Results width.
- Produces: `PanelLayout`, constants, parsing/persistence helpers, safe bound helpers, `usePanelLayout`, and `useElementWidth`.

- [ ] **Step 1: Write failing storage and bound tests**

Cover these exact cases:

```ts
expect(parsePanelLayout(null)).toEqual(DEFAULT_PANEL_LAYOUT);
expect(parsePanelLayout('{bad json')).toEqual(DEFAULT_PANEL_LAYOUT);
expect(parsePanelLayout(JSON.stringify({ version: 2, toolbox: 300, inspector: 400, metrics: 350 })))
  .toEqual(DEFAULT_PANEL_LAYOUT);
expect(parsePanelLayout(JSON.stringify({ version: 1, toolbox: null, inspector: 400, metrics: 350 })))
  .toEqual(DEFAULT_PANEL_LAYOUT);

expect(clampWorkspaceLayout(
  { toolbox: 420, inspector: 640, metrics: 330 },
  1180,
)).toMatchObject({ toolbox: 408, inspector: 280 });

expect(maxMetricsWidth(900)).toBe(414);
expect(maxMetricsWidth(600)).toBe(260);
```

Use the final implementation constants rather than repeating raw handle and
minimum widths inside the assertions where doing so makes the expected
calculation clearer.

- [ ] **Step 2: Run the model test and record red**

Run:

```bash
cd web-ui
npm test -- src/components/panelLayout.test.ts
```

Expected: FAIL because `panelLayout` does not exist.

- [ ] **Step 3: Implement exact layout types and safe parsing**

Export:

```ts
export interface PanelLayout {
  toolbox: number;
  inspector: number;
  metrics: number;
}

export const PANEL_LAYOUT_STORAGE_KEY = 'calc-flow-studio:panel-layout:v1';
export const PANEL_RESIZE_HANDLE_WIDTH = 6;
export const DEFAULT_PANEL_LAYOUT: PanelLayout = {
  toolbox: 235,
  inspector: 335,
  metrics: 330,
};
export const PANEL_LIMITS = {
  toolbox: { min: 200, max: 420 },
  inspector: { min: 280, max: 640 },
  metrics: { min: 260 },
  canvasMin: 480,
  outputMin: 480,
} as const;
```

`parsePanelLayout(raw)` accepts only a version-1 object containing three finite
numbers, then statically clamps individual widths. Any missing, malformed,
non-finite, or wrong-version payload returns a fresh default object.

`clampWorkspaceLayout(layout, containerWidth)` first clamps Toolbox and
Inspector individually, then constrains their combined size to
`containerWidth - canvasMin - 2 * handleWidth`. Reduce Inspector toward its
minimum first, then Toolbox. This deterministic rule preserves the content-heavy
Toolbox preference unless both sides must shrink.

`maxMetricsWidth(containerWidth)` returns the safe maximum after Output and one
handle, never below the Metrics minimum. `clampResultsLayout` applies it.

- [ ] **Step 4: Add best-effort storage and React hooks**

Implement `readPanelLayout(storage)`, `writePanelLayout(storage, layout)`, and
`usePanelLayout()` with guarded `try/catch`. The stored payload is:

```ts
JSON.stringify({ version: 1, ...layout })
```

The hook uses a lazy state initializer, a functional `setPanelWidth(name,
value)`, a `resetPanelWidth(name)`, and an effect that writes current safe
state. Export `useElementWidth<T extends HTMLElement>()` as a callback-ref plus
numeric-width hook. It reads the initial `getBoundingClientRect().width`, then
observes the mounted element and disconnects its `ResizeObserver` on replacement
or unmount.

- [ ] **Step 5: Verify the model and hook helpers green**

Run:

```bash
cd web-ui
npm test -- src/components/panelLayout.test.ts
```

Expected: all layout parsing, clamping, storage, and observer cleanup tests PASS.

- [ ] **Step 6: Commit the layout model**

```bash
git add web-ui/src/components/panelLayout.ts web-ui/src/components/panelLayout.test.ts
git commit -m "feat: persist safe Studio panel widths"
```

### Task 4: Accessible Resize Handles and Grid Integration

**Files:**
- Create: `web-ui/src/components/PanelResizeHandle.tsx`
- Create: `web-ui/src/components/PanelResizeHandle.test.tsx`
- Modify: `web-ui/src/App.tsx`
- Modify: `web-ui/src/App.test.tsx`
- Modify: `web-ui/src/components/ResultsPanel.tsx`
- Modify: `web-ui/src/components/ResultsPanel.test.tsx`
- Modify: `web-ui/src/styles.css`

**Interfaces:**
- Consumes: Task 3 layout values, bounds, reset callback, and element-width hook.
- Produces: three accessible separators and CSS Grid widths for the workspace, header, Output, and Metrics panels.

- [ ] **Step 1: Write failing handle interaction tests**

Render a handle with `value={300}`, `min={200}`, `max={400}` and assert:

```tsx
fireEvent.pointerDown(separator, { pointerId: 1, clientX: 100 });
fireEvent.pointerMove(separator, { pointerId: 1, clientX: 140 });
expect(onChange).toHaveBeenLastCalledWith(340);

fireEvent.keyDown(separator, { key: 'ArrowRight' });
expect(onChange).toHaveBeenLastCalledWith(316);
fireEvent.keyDown(separator, { key: 'ArrowRight', shiftKey: true });
expect(onChange).toHaveBeenLastCalledWith(348);
fireEvent.keyDown(separator, { key: 'Home' });
expect(onChange).toHaveBeenLastCalledWith(200);
fireEvent.keyDown(separator, { key: 'End' });
expect(onChange).toHaveBeenLastCalledWith(400);
fireEvent.doubleClick(separator);
expect(onReset).toHaveBeenCalledOnce();
```

Add a `grow="end"` test proving a 40 px move left grows a right-hand panel by
40 px. Assert `role="separator"`, `aria-orientation="vertical"`,
`aria-valuemin`, `aria-valuemax`, and `aria-valuenow`.

- [ ] **Step 2: Run the handle test and record red**

Run:

```bash
cd web-ui
npm test -- src/components/PanelResizeHandle.test.tsx
```

Expected: FAIL because `PanelResizeHandle` does not exist.

- [ ] **Step 3: Implement the minimal handle**

Use this contract:

```tsx
interface PanelResizeHandleProps {
  label: string;
  value: number;
  min: number;
  max: number;
  grow: 'start' | 'end';
  onChange: (value: number) => void;
  onReset: () => void;
}
```

Store only the active pointer ID, start X, and start value in a ref. Use pointer
capture on down, calculate physical separator motion on move, invert the delta
for `grow="end"`, clamp every emitted value, and clear/release capture on up or
cancel. Keyboard handling moves the physical separator, not an abstract value,
so ArrowLeft enlarges end-growing Inspector/Metrics and ArrowRight enlarges the
start-growing Toolbox.

- [ ] **Step 4: Add failing App layout tests before integration**

In `App.test.tsx`, clear local storage in `afterEach`, seed:

```ts
localStorage.setItem(
  'calc-flow-studio:panel-layout:v1',
  JSON.stringify({ version: 1, toolbox: 300, inspector: 410, metrics: 360 }),
);
```

Render `App`, then assert the Studio shell or workspace exposes `300px` and
`410px` through its custom properties. Fire an ArrowRight event on the Toolbox
separator, assert `316px`, and assert the stored payload is updated through
`waitFor`. Also seed malformed JSON and assert default widths render without an
exception.

- [ ] **Step 5: Integrate workspace separators and safe measurement**

In `App`, call `usePanelLayout()` once. Measure `.workspace`, derive a safe
layout with `clampWorkspaceLayout`, and reconcile clamped widths back to the
hook in an effect only when the measured width is positive and values differ.

Render this order:

```tsx
<aside className="toolbox panel">...</aside>
<PanelResizeHandle label="Resize Toolbox" grow="start" ... />
<section className="canvas-panel">...</section>
<PanelResizeHandle label="Resize Inspector" grow="end" ... />
<NodeInspector ... />
```

Expose `--toolbox-width` and `--inspector-width` on `.studio-shell`. Use the
Toolbox value for the header's first grid column and both values for the
workspace grid.

- [ ] **Step 6: Add failing Results tests and integrate its separator**

Extend `ResultsPanelProps` with:

```ts
metricsWidth: number;
onMetricsWidthChange: (width: number) => void;
onMetricsWidthReset: () => void;
```

Update existing tests with defaults, then add a completed-run test that locates
`Resize Metrics`, sends ArrowLeft, and expects the Metrics callback to receive a
larger value. Measure `.result-grid`, clamp against `maxMetricsWidth`, and emit a
safe correction in an effect only after a positive measurement.

Render the separator between `.output-stack` and `.metrics-stack`, expose
`--metrics-width`, and update `.result-grid` to
`minmax(480px, 1fr) 6px var(--metrics-width)`.

- [ ] **Step 7: Style all handles and panel grids**

Update `.workspace` to use five columns and `.result-grid` to use three. Remove
the fixed widths from the 1300 px media rule. Add a 6 px `.panel-resize-handle`
with a centered 1 px line, `cursor: col-resize`, touch-action disabled only on
the handle, and visible hover/focus/drag color. Retain internal panel overflow
and `min-width: 0` for Canvas, Output, and Metrics descendants.

- [ ] **Step 8: Verify focused layout tests and the full unit suite**

Run:

```bash
cd web-ui
npm test -- \
  src/components/panelLayout.test.ts \
  src/components/PanelResizeHandle.test.tsx \
  src/components/ResultsPanel.test.tsx \
  src/App.test.tsx
npm test
npm run build
```

Expected: focused tests, the complete Vitest suite, TypeScript, and Vite build
all PASS without warnings.

- [ ] **Step 9: Commit the resize UI**

```bash
git add \
  web-ui/src/components/PanelResizeHandle.tsx \
  web-ui/src/components/PanelResizeHandle.test.tsx \
  web-ui/src/components/ResultsPanel.tsx \
  web-ui/src/components/ResultsPanel.test.tsx \
  web-ui/src/App.tsx \
  web-ui/src/App.test.tsx \
  web-ui/src/styles.css
git commit -m "feat: resize Studio workspace panels"
```

### Task 5: Browser Workflow, Live Example, and PR Verification

**Files:**
- Modify: `web-ui/e2e/studio.spec.ts`
- Verify only: `schemas/project-v2.schema.json`
- Verify only: `web-ui/openapi.json`
- Verify only: `web-ui/src/api/schema.d.ts`

**Interfaces:**
- Consumes: all completed alias and resize UI behavior.
- Produces: browser regression coverage, a verified live two-source example, and an updated Draft PR #17.

- [ ] **Step 1: Write the failing browser workflow assertions**

Within the persisted two-source test:

1. click Add DataFusion SQL and assert `Input alias 1` contains `input`;
2. click Add input alias, assert `Input alias 2` contains `input_2`, rename it
   to `right`, commit with Enter, and delete the temporary node;
3. drag `Resize Toolbox` and `Resize Inspector`, recording the resulting panel
   widths;
4. run the existing join and wait for total 18 in the repository E2E fixture;
5. drag `Resize Metrics`, record its width, reload, and assert all three widths
   are restored within a 2 px tolerance; and
6. keep the existing exact saved inputs `left_source` and `right_source`.

Run the browser test before implementation is complete and confirm it fails on
the missing alias-row or separator control rather than an unrelated server
startup failure.

- [ ] **Step 2: Make only the minimal E2E-driven corrections**

Correct accessible labels, pointer capture behavior, width reconciliation, or
Playwright targeting only when the failure demonstrates a mismatch with the
approved design. Do not weaken exact alias, persistence, or join assertions.

- [ ] **Step 3: Run the complete frontend command group**

```bash
cd web-ui
npm ci
npm run sync:api
npm run build
npm test
CI=1 npm run test:e2e
npm audit --omit=dev
```

Expected: API sync is clean, build passes, all Vitest and Playwright tests pass,
and audit reports zero production vulnerabilities.

- [ ] **Step 4: Run the repository-required non-frontend verification**

From the repository root, use repository-local target/cache locations when the
managed environment requires them, then run:

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
cd web-ui/backend
uv run --project . --extra dev pytest --cov=calc_flow_studio
cd ../..
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
cargo deny --locked check
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
```

Expected: every required group passes, Rust line coverage remains at least 90%,
and Studio backend coverage remains at least 85%. Remove any generated
`python/calc_flow/_native*.so` after Python verification and prove the source
tree contains none.

- [ ] **Step 5: Verify generated contracts and repository scope**

```bash
git diff --exit-code -- \
  schemas/project-v2.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
git diff --check
find python/calc_flow -maxdepth 1 -type f -name '_native*.so' -print
git status --short --branch
```

Expected: generated contracts have no diff, whitespace check is clean, native
search prints nothing, and only intended committed files differ from `main`.

- [ ] **Step 6: Restart and verify the managed Studio**

Use:

```bash
./web-ui/scripts/stop_web_ui.sh
./web-ui/scripts/start_web_ui.sh
```

Keep the owning session alive. Verify HTTP 200 at `http://127.0.0.1:5173` and
`http://127.0.0.1:8765`. Load `two_upstream_demo`, confirm aliases `left` and
`right` are independently editable, validate with no issues, submit `{}` to its
run endpoint, and assert a completed output with totals 16 and 30.

- [ ] **Step 7: Push the exact verified head and update Draft PR #17**

Use the designated external git directory for every git operation:

```bash
git --git-dir=/tmp/calc-flow-multi-source-git-https/.git \
  --work-tree=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow \
  push origin feature/studio-multi-source-editor
gh pr view 17 --repo wegamekinglc/calc-flow \
  --json url,state,isDraft,headRefOid,mergeable
gh pr checks 17 --repo wegamekinglc/calc-flow
```

Expected: PR #17 stays open as a Draft, its head OID equals the locally verified
HEAD, it is mergeable, and GitHub checks start for that exact head. Report any
pending checks as pending rather than claiming they passed.
