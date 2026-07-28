# DAL-37 Data Source Dialog and Toolbar Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a large accessible Data Source editor dialog and align the Studio toolbar controls at desktop and narrow viewports without changing project or API contracts.

**Architecture:** `DataSourceEditor` owns a transient modal draft identified by the stable `DataSourceDraft.key`, while `App` continues to own committed UI text and later typed-project materialization. A focused `DataSourceDialog` component establishes the first internal native-dialog convention, and `App` exposes per-source pending-read keys so file reads and modal commits cannot overlap. Scoped CSS and Playwright geometry checks cover only the top toolbar and dialog.

**Tech Stack:** React 19, TypeScript 5.9, native HTML `<dialog>`, React Testing Library/Vitest, Playwright, existing CSS custom properties.

## Global Constraints

- Preserve modal draft → committed `sourceDrafts[index].dataText` → later typed project materialization as three distinct state boundaries.
- Validate only `inline_json` with the existing `JSON.parse` semantics and `Invalid inline JSON` message; keep `json`, `csv`, and `arrow_ipc` opaque.
- Do not add dependencies, backend/API/schema changes, generated API changes, broad formatting, or unrelated mobile redesign.
- Use the stable `DataSourceDraft.key` for async ownership; array indices are not identities.
- Disable a source's Edit opener while its file read is pending and prevent a file read from starting while that source's dialog is open.
- Use `1440x900` and `390x844` as the exact measured browser viewports.
- Preserve the controlling critique at `.codex/artifacts/critiques/dal-37-ui-button-alignment-data-source-dialog.md` unchanged.

---

### Task 1: Establish the internal Data Source dialog

**Files:**
- Create: `web-ui/src/components/DataSourceDialog.tsx`
- Create: `web-ui/src/components/DataSourceDialog.test.tsx`

**Interfaces:**
- Consumes: `format: DataSourceFormat`, `initialText: string`, `sourceLabel: string`, `onConfirm(text: string): void`, and `onDismiss(): void`.
- Produces: `DataSourceDialog`, whose only parent mutation is one successful `onConfirm` callback.

- [ ] **Step 1: Write failing dialog lifecycle tests**

```tsx
render(
  <DataSourceDialog
    format="inline_json"
    initialText="[]"
    sourceLabel="sample"
    onConfirm={onConfirm}
    onDismiss={onDismiss}
  />,
);
fireEvent.change(screen.getByRole('textbox', { name: 'Data source data for sample' }), {
  target: { value: '[{' },
});
fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
expect(onConfirm).not.toHaveBeenCalled();
expect(screen.getByText('Invalid inline JSON')).toBeInTheDocument();
```

Cover editor-first focus, local typing, valid confirmation, inline JSON failure, opaque formats, shared Escape/backdrop/close/Cancel dismissal, and forward/reverse focus containment.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
node node_modules/vitest/vitest.mjs run src/components/DataSourceDialog.test.tsx
```

Expected: FAIL because `DataSourceDialog` does not exist.

- [ ] **Step 3: Implement the minimal dialog**

Use `showModal()` when available, fall back to the `open` attribute in jsdom, focus the textarea after opening, handle `cancel`, validate only on Confirm, keep the dialog open on invalid JSON, detect real backdrop coordinates, and cycle Tab/Shift+Tab across enabled focusable descendants.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2 and require all dialog tests to pass.

### Task 2: Replace direct editing with a stable-key modal draft

**Files:**
- Modify: `web-ui/src/components/DataSourceEditor.tsx`
- Modify: `web-ui/src/components/DataSourceEditor.test.tsx`

**Interfaces:**
- Consumes: new `pendingSourceKeys: ReadonlySet<string>` prop.
- Produces: accessible `Edit data source <label>` buttons and bounded `Data <N> preview` regions; calls `onDataChange(index, text)` exactly once only after valid Confirm.

- [ ] **Step 1: Write failing editor integration tests**

```tsx
fireEvent.click(screen.getByRole('button', { name: 'Edit data source left' }));
fireEvent.change(screen.getByRole('textbox', { name: 'Data source data for left' }), {
  target: { value: '[1]' },
});
expect(props.onDataChange).not.toHaveBeenCalled();
fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
expect(props.onDataChange).toHaveBeenCalledOnce();
expect(props.onDataChange).toHaveBeenCalledWith(0, '[1]');
```

Also cover preview updates under a controlled harness, reopening from latest committed text after every discard path, exact opener focus restoration, pending-read opener disabling, and Load file disabling while the source dialog is open.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
node node_modules/vitest/vitest.mjs run src/components/DataSourceEditor.test.tsx
```

Expected: FAIL because the direct textarea still edits parent state and no Edit opener exists.

- [ ] **Step 3: Implement the modal-local stable-key state**

Store `{ key, initialText, format, label }` locally, resolve the current index by key at Confirm time, close if the key disappears, route every discard path through one callback, and restore the recorded opener after unmount.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2 and require all editor tests to pass.

### Task 3: Enforce per-source file-read ownership in App

**Files:**
- Modify: `web-ui/src/App.tsx`
- Modify: `web-ui/src/App.test.tsx`

**Interfaces:**
- Produces: `pendingFileReadKeys: ReadonlySet<string>` derived from per-key in-flight counts.
- Preserves: newest same-source file wins, replacement/format change rejects stale reads, persistence waits for reads, and a later confirmed manual edit wins.

- [ ] **Step 1: Adapt existing App regressions to the dialog and add the pending-owner failure**

```tsx
fireEvent.change(screen.getByLabelText('Load file 1'), {
  target: { files: [delayed.file] },
});
expect(screen.getByRole('button', { name: 'Edit data source sample' }))
  .toBeDisabled();
```

Replace direct `Data 1` interactions with Edit → modal editor → Confirm, and replace value assertions with bounded preview assertions. Keep all existing delayed-read and project-replacement cases.

- [ ] **Step 2: Run the focused App tests and verify RED**

Run:

```bash
node node_modules/vitest/vitest.mjs run src/App.test.tsx
```

Expected: FAIL because `App` does not expose per-source pending ownership and current tests still target the removed direct textarea.

- [ ] **Step 3: Add per-key in-flight counting**

Increment a `Map<string, number>` before each read, publish a fresh `Set` of active keys, decrement in `finally`, and pass the set to `DataSourceEditor`. Keep the existing global pending count and stable-key/token checks.

- [ ] **Step 4: Run App and full unit suites**

Run:

```bash
node node_modules/vitest/vitest.mjs run src/App.test.tsx
node node_modules/vitest/vitest.mjs run
```

Require all tests to pass with no unhandled errors.

### Task 4: Align toolbar controls and constrain the responsive dialog

**Files:**
- Modify: `web-ui/src/styles.css`
- Modify: `web-ui/e2e/studio.spec.ts`

**Interfaces:**
- Produces: scoped `.project-actions`, `.topbar-actions`, `.topbar-control`, `.data-source-dialog`, and dialog section layout rules.

- [ ] **Step 1: Write failing Playwright geometry/focus assertions**

Measure New, Import, Export JSON, Export YAML, Delete, Save, Validate, and Run preview at `1440x900` and `390x844`. Assert every box is inside the viewport, boxes do not overlap, equal explicit heights are used, and controls sharing a row share top/bottom coordinates. Measure the dialog shell, heading, editor, error, and action row; assert `scrollWidth <= clientWidth`, usable editor height, focus containment, discard restoration, and Confirm semantics.

- [ ] **Step 2: Run the focused browser test and verify RED**

Run:

```bash
node node_modules/@playwright/test/cli.js test e2e/studio.spec.ts --grep "Data Source dialog"
```

Expected: FAIL because the dialog and responsive toolbar rules do not yet exist.

- [ ] **Step 3: Add scoped styles and update the saved-source workflow**

Use an explicit shared 36px toolbar control height, contextual inline-flex Import/Delete overrides, and intentional wrapping. Remove the global narrow-viewport obstruction only at the focused breakpoint, keep the workspace desktop-sized/scrollable, and use viewport-bounded flexible dialog layout with a sticky/reachable action row.

- [ ] **Step 4: Run focused and full Playwright suites**

Run:

```bash
node node_modules/@playwright/test/cli.js test e2e/studio.spec.ts --grep "Data Source dialog"
node node_modules/@playwright/test/cli.js test
```

Require all browser tests to pass.

### Task 5: Full repository verification and delivery

**Files:**
- Verify only; no generated schema or API output may change.

- [ ] **Step 1: Run affected frontend gates**

```bash
node node_modules/vitest/vitest.mjs run
node node_modules/typescript/bin/tsc -b
node node_modules/vite/bin/vite.js build
node node_modules/@playwright/test/cli.js test
```

- [ ] **Step 2: Verify generated contracts and diff hygiene**

```bash
git diff --exit-code -- schemas/project-v2.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts
git diff --check
```

- [ ] **Step 3: Review all ten acceptance criteria against tests and diff**

Confirm each issue checkbox has implementation evidence, no backend/Rust/Python/generated/dependency file changed, and the critique hash remains `3853c00c5ebdad3ec7c8a2a47f6195cad3bf8694a628bc0c4c62369802626ea6`.

- [ ] **Step 4: Commit, push, and open an issue-linked PR**

Commit focused changes on `feature/dal-37-data-source-dialog`, push to `origin`, and open a non-merged PR whose body contains `## Summary`, `## Test plan`, and the DAL-37 issue identifier.
