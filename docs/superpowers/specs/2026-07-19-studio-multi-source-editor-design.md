# Studio Multi-Source Editor Design

## Goal

Allow Calc Flow Studio users to add, remove, edit, save, validate, and run all
of a project's configured data sources. Replace the current single preview
override with project-backed source editors so a graph with two external
inputs can be configured and run entirely from the Studio.

After the frontend change, restore the `two_upstream_demo` project to a true
two-input example using `left_source` and `right_source`, then verify it through
the running Studio.

## Current Failure

The backend and project format already support multiple sources. A project
stores them in `data_sources`, and the run manager requires the saved source
names, submitted override names, and compiled graph inputs to be identical.

The frontend does not use that model. `App` owns one input name, one preview
format, and one data string. Every run therefore submits exactly one override:

```text
inputs: { [sampleInputName]: ... }
```

For a graph whose external inputs are `left_source` and `right_source`, the
browser submits only `input`, producing:

```text
run inputs must be ['left_source', 'right_source']; received ['input']
```

The current editor also cannot add or remove project data sources, and its
format choices describe run overrides rather than the canonical saved-source
formats.

## Selected Approach

Make `project.data_sources` the persisted source of truth and render one editor
card per source. Save, Validate, and Run materialize all valid editor drafts
into a new immutable project value before calling the API. Run persists that
project and submits an empty run request, allowing the backend to load the
complete saved source set.

This is preferred over maintaining a separate multi-input preview model because
separate saved and preview values can drift. It is preferred over extending the
run override API because the saved-source path already accepts all project
formats and requires no backend, OpenAPI, or generated-type changes.

## Source Editor Model

Add a focused `DataSourceEditor` component and pure source-editor helpers rather
than expanding the already large `App` component. The component receives the
current project sources, format-specific data drafts and validation errors, and
callbacks for immutable add, remove, field-edit, data-edit, and file-load
operations.

Each source card contains:

- a source ID field;
- a graph input field;
- a format selector for `inline_json`, `json`, `csv`, and `arrow_ipc`;
- a data textarea;
- a format-aware file picker; and
- a Remove action.

The section includes an Add data source action. New sources receive the first
available deterministic names in the form `source_N` and `input_N`, use
`inline_json`, and start with an empty record array. Generated names avoid
collisions with every source currently in the project.

Removing a source removes only that source and its draft. Studio will not
silently rewrite nodes or edges. A subsequent Validate reports the existing
`source_input_mismatch` issue when the configured sources no longer cover the
compiled external inputs.

## Drafts and Immutable State

Textareas need to represent temporarily invalid JSON while a user types, but a
project document must remain valid JSON data. Keep a small editor draft for each
source's data text while keeping source ID, input, and format changes in
`project.data_sources` through functional immutable updates.

Draft entries use an internal stable UI key rather than editable source IDs, so
renaming or temporarily duplicating an ID does not lose textarea state. Loading,
creating, or importing a project replaces the entire source/draft collection
from that project. Deleting a source removes its draft; adding one creates both
values in the same functional update path.

Before Save, Validate, or Run, materialization converts every draft according to
its saved-source format:

- `inline_json` must parse as JSON and is stored as the parsed JSON value;
- `json` is stored as text, preserving JSON documents or newline-delimited JSON;
- `csv` is stored as text; and
- `arrow_ipc` is stored as Base64 text.

Existing non-string `json` values remain editable by serializing them as
formatted JSON text when the project loads. File loading uses text for JSON and
CSV and Base64 for Arrow IPC. The implementation never mutates arrays, objects,
or caller-owned project values.

## Save, Validate, Run, and Checkpoint Flow

Every action that persists the editable project first materializes the current
editor drafts. This includes Save, Validate, Run, and checkpoint inspection for
an unsaved or edited project. If a source has invalid `inline_json`, the action
stops before any request, marks that source, and displays a concise error
identifying its source ID or card position.

On success:

1. build a new project containing every materialized data source;
2. update local project state to that value;
3. create or save the project through the existing project API; and
4. continue with validation or execution where requested.

Run calls the existing endpoint with an empty request object. The backend then
uses the saved `data_sources` values, including their configured source IDs,
input names, and formats. It receives the exact complete input set instead of a
single browser-preview override.

The existing project-level validation remains authoritative for duplicate IDs,
duplicate graph inputs, unsupported formats, invalid portable IDs, and source
coverage. The frontend adds only the immediate syntax validation needed to
avoid silently saving stale data when an `inline_json` draft is incomplete.

## UI and Accessibility

Replace the single `Preview input` panel with a scrollable `Data sources`
section. Cards use semantic labels containing the source index or ID so fields,
file controls, and Remove actions remain uniquely addressable in tests and by
assistive technology. The Add action is always visible. Empty projects display
a short explanation that graph inputs require matching data sources.

The sidebar remains usable at the existing desktop breakpoints. Source cards
stack their controls vertically, long data is contained by the textarea, and
the source list scrolls inside the available sidebar height rather than
expanding the entire workspace.

## Example Project

After the frontend is built and the managed Studio is restarted, update the
existing `two_upstream_demo` project to use two compiled external inputs:

- `left_source`, backed by its own inline JSON records;
- `right_source`, backed by its own inline JSON records;
- a left expression branch and a right expression branch; and
- a `join_result` SQL node consuming both upstream branch outputs.

The example must load with two source cards, validate successfully, and complete
a preview run from the UI-compatible saved-source path. Its output must
demonstrate that values from both sources reached the join, rather than using a
single external input fanned out to two branches.

The example is Studio runtime data, not a repository fixture, unless an existing
tracked fixture is discovered during implementation. Runtime state, logs, and
process metadata remain uncommitted.

## Testing

Every behavior change starts with a focused failing test.

- Add component/helper tests for rendering multiple sources, deterministic add,
  targeted removal, immutable edits, format changes, and per-source file loads.
- Replace the single-preview `App` contract test with a two-source project test.
  It must prove that Save persists both edited sources and Run submits an empty
  request instead of a one-name override.
- Add an App test proving that invalid `inline_json` identifies the affected
  source and prevents Save, Validate, and Run requests.
- Preserve coverage for loading, creating, importing, and switching projects;
  add assertions that their source drafts reset to the selected document.
- Extend the browser workflow to load the two-source example, observe both
  editors, run it, and observe a completed result where practical without
  coupling the repository test to persistent local runtime state.

Run the focused Vitest files first, then the complete frontend unit suite,
frontend build, and Playwright workflow. Because no route or model changes are
planned, `web-ui/openapi.json` and `web-ui/src/api/schema.d.ts` must remain
unchanged. Finish with the repository-required generated-contract and diff
checks.

## Commit Structure

Commit this approved design separately. The implementation will use a new
feature branch based on current `main` and keep frontend code, tests, and styles
in a focused implementation commit. Updating and exercising the local Studio
example is a verified runtime step and is not committed unless it changes an
already tracked fixture.

## Acceptance Criteria

- Studio renders every configured project data source and supports adding,
  removing, and editing sources without mutating prior state.
- Save persists source ID, graph input, format, and materialized data for every
  source.
- Invalid inline JSON is attributed to the correct card and blocks actions
  without persisting stale values.
- Run uses the saved complete source set and no longer emits a single `input`
  override.
- The true `left_source` plus `right_source` example validates and completes a
  real preview run after Studio restart.
- Focused and full relevant frontend verification passes, generated API
  artifacts do not change, and runtime artifacts are not committed.
