# Studio Input Aliases and Resizable Panels Design

**Status:** Implemented and merged in PR #17 (historical design)

## Goal

Make SQL nodes practical for multi-upstream graphs by replacing the fragile
comma-separated alias field with an explicit input-alias editor. At the same
time, make the Studio's horizontally adjacent panels resizable so source data,
SQL, schemas, output tables, and metrics can receive enough width to remain
readable.

This is a frontend-only follow-up to the Studio multi-source editor. Project
format v2, backend routes, generated API types, and runtime semantics remain
unchanged.

## Current Failures

`NodeInspector` renders SQL aliases in one controlled text input. Every input
event immediately splits on commas, trims entries, removes empty entries, and
renders the normalized array again. When a user types a comma after the first
alias, that delimiter is removed before the second alias can be entered. A
complete comma-separated value can be pasted, but two aliases cannot be typed
normally.

The main workspace also uses fixed grid widths for Toolbox and Inspector, and
the Results panel uses a fixed Metrics width. Long source values, SQL queries,
schemas, output tables, and physical plans are therefore unnecessarily hard to
inspect even when the browser has usable horizontal space.

## Selected Approach

Use two focused, dependency-free UI primitives:

- an `InputAliasEditor` that edits one SQL alias per row and emits semantic
  add, rename, and remove operations; and
- a `PanelResizeHandle` plus pure panel-layout helpers that coordinate adjacent
  CSS Grid columns and persist safe widths in browser-local storage.

This is preferred over retaining a locally buffered comma-separated field
because separate rows remove delimiter ambiguity and make individual errors and
Remove actions clear. It is preferred over a generalized port editor because
output-port editing and external-provider configuration are outside this
request.

For resizing, native CSS `resize` cannot reliably coordinate both adjacent
columns or provide consistent keyboard and persistence behavior. A split-pane
package would add supply-chain and maintenance cost for three vertical
separators, so a small repository-owned implementation is more appropriate.

## SQL Input Alias Editor

`InputAliasEditor` receives the current alias strings and callbacks for Add,
Rename, and Remove. It renders a labeled text input and Remove button for each
alias, followed by an always-visible Add input alias action.

Each row keeps only its temporary text draft. Rename commits on Enter or blur;
Escape restores the saved value. A commit trims surrounding whitespace and is
rejected when the result is empty or duplicates another alias. Rejected drafts
stay visible with an associated inline error, while the project document keeps
the last valid alias. This avoids invalid React Flow handle identifiers and
prevents invalid intermediate keystrokes from leaking into persistence calls.

Add immediately creates the first available deterministic name. It tries
`input`, then `input_2`, `input_3`, and so on. Remove is immediate. Zero aliases
remain valid in the editor because DataFusion SQL can contain a query that does
not read an input table. Newly created SQL nodes retain their current default
single `input` alias.

## Graph-Safe Alias Operations

Alias edits affect more than the operator array, so `App` owns a pure immutable
project transition for each semantic operation. `NodeInspector` does not edit
edges directly.

Add performs the following changes:

1. append the generated alias to the SQL operator;
2. leave derived input ports empty when the node has no explicit ports; or
3. append a required table port with an empty schema when explicit input ports
   already exist.

Rename performs the following changes atomically:

1. replace the matching SQL operator alias;
2. rename the matching explicit input port while preserving its kind, required
   flag, and Arrow schema; and
3. rewrite incoming edges whose `target_node` and `target_port` identify the
   renamed alias.

Remove atomically deletes the operator alias, matching explicit input port, and
incoming edges targeting that alias. It does not alter unrelated edges or
upstream nodes. These transitions preserve caller-owned values and construct a
new project, pipeline, node array, and edge array only where required.

## Resizable Workspace

The main workspace receives separators between Toolbox and Canvas and between
Canvas and Inspector. The Results grid receives a separator between Output and
Metrics. Benchmark Comparison remains a full-width row and therefore needs no
horizontal separator.

The initial widths match the current layout:

- Toolbox: 235 px, clamped from 200 px to 420 px;
- Inspector: 335 px, clamped from 280 px to 640 px; and
- Metrics: 330 px, with a 260 px minimum.

Canvas and Output consume the remaining width. Canvas keeps at least 480 px and
Output keeps at least 480 px. A drag that would violate either neighboring
minimum stops at the boundary. Existing panel overflow rules continue to keep
large content scrollable inside the selected width.

The workspace and Results grids expose their selected sizes through CSS custom
properties. The header's first column uses the Toolbox width so its visual
alignment follows the left panel. React Flow remains mounted throughout a drag;
its existing `ResizeObserver` integration receives the container-size change.

## Resize Interaction and Accessibility

Each handle has `role="separator"`, vertical orientation, a readable label, and
a current numeric value. Pointer interaction uses pointer capture so dragging
continues outside the narrow handle. Pointer listeners and capture are cleaned
up at drag completion and component unmount.

When focused, Left and Right Arrow move the separator by 16 px. Holding Shift
moves it by 48 px. Moving the Toolbox separator right enlarges Toolbox; moving
the Inspector or Metrics separator left enlarges its right-hand panel. Home
moves the separator to the relevant panel's minimum, End moves it to that
panel's largest currently safe width, and a double click restores that
separator's default width. Handles show clear hover, focus, and active states
without covering panel content.

## Persistence and Bounds

Panel widths are a browser preference, not project data. Store one versioned
record under `calc-flow-studio:panel-layout:v1`:

```json
{
  "version": 1,
  "toolbox": 235,
  "inspector": 335,
  "metrics": 330
}
```

The layout initializes from this record and writes valid committed sizes after
interaction. Missing storage, unavailable storage, invalid JSON, wrong
versions, non-finite values, and out-of-range values all fall back to defaults
or safe clamped values without blocking Studio startup.

Container measurements are authoritative. On browser resize, restored widths
are clamped again so Canvas and Output retain their minimums. The adjusted safe
values become the active layout and are persisted, preventing every reload from
restoring an impossible arrangement.

## Error Handling

Alias validation errors are local to their row and identify empty or duplicate
names. Add always generates a valid unique value, so it cannot introduce an
error. Rename and Remove are synchronous immutable operations and do not create
network requests.

Layout persistence is best-effort. Storage read or write failures are ignored
after selecting an in-memory safe layout. A lost pointer event ends at the last
accepted width; no project state is involved and no calculation can be
corrupted.

## Testing

Every behavior change starts with a focused failing test.

- Add `InputAliasEditor` tests that type and commit a second alias without
  commas, reject blank and duplicate drafts, restore on Escape, and remove a
  selected row.
- Add pure project-transition tests proving that rename updates the operator,
  explicit schema port, and matching incoming edge while preserving unrelated
  values, and that Remove deletes only the corresponding port and edge.
- Extend `NodeInspector` tests to prove SQL nodes render one row per alias and
  request semantic operations rather than normalizing a comma-separated field.
- Add layout-helper tests for defaults, storage parsing, version rejection,
  finite-number validation, and clamping against neighboring minimums.
- Add resize-handle tests for pointer deltas, keyboard steps, bounds, double
  click reset, and listener cleanup.
- Extend App and Results tests to prove CSS column sizes update immutably and a
  stored layout is restored.
- Extend the Playwright workflow to resize both workspace and Results panels,
  reload, and confirm the widths remain within measurement tolerance.
- Preserve the existing two-source browser run and live
  `two_upstream_demo` validation, including joined totals 16 and 30.

Run focused Vitest tests first, then the full frontend unit suite, generated API
sync, build, Playwright workflow, and audit. Run the repository-required backend,
Rust, Python, supply-chain, generated-contract, and diff checks before the PR is
considered complete. No OpenAPI or project-schema change is expected.

## Acceptance Criteria

- A user can add and type two or more SQL input aliases without pasting a
  comma-separated string.
- Alias rows support accessible Add, Rename, and Remove actions with local
  blank and duplicate validation.
- Rename preserves and updates matching schemas and incoming edges; Remove
  clears only the matching schema port and incoming edges.
- Toolbox, Canvas, Inspector, Output, and Metrics widths can be adjusted through
  the three separators without violating their minimum readable widths.
- Resize handles work with pointer, touch-compatible pointer events, keyboard,
  and double-click reset.
- Safe widths survive reload through versioned local storage and recover from
  corrupt or impossible stored values.
- The true two-source Studio example still validates and produces totals 16 and
  30.
- All focused and required full verification passes, generated contracts stay
  unchanged, and no runtime artifact is committed.

## Commit Scope

Commit this approved design separately on the existing
`feature/studio-multi-source-editor` branch and update Draft PR #17. The later
implementation commits remain frontend-focused. Managed Studio project data,
logs, process state, and local panel preferences remain untracked.
