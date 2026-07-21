# Head Operator - Specification

## Source
- Issue: user request on 2026-07-21 ("Spec a new `head` operator for calc-flow: it passes through only the first N rows of each table batch it receives, dropping the rest; N is a positive integer configured at construction")
- Related docs: `docs/introduction.md` — "Batch contract", "Graph compilation", "Table execution", "Streaming and recovery"; `schemas/project-v2.schema.json` (v2 operator kinds); `AGENTS.md` (verification matrix)

## Problem Statement
Calc-flow's built-in table operators (`expression`, `sql`) can calculate, project, filter, and join table batches, but there is no first-class way to bound how many rows a node emits per batch. A user who wants only the first N rows of each batch — for previews, probe pipelines, or downstream cost control — must hand-write a `SqlOperator` query such as `SELECT * FROM input LIMIT N`. That is boilerplate, is invisible to the graph as a distinct row-bounding semantic, and cannot be validated or displayed as a bounded-row contract. Pipeline authors and Studio users building preview and debugging flows feel this gap.

## Goals
- A built-in `head` table operator that emits only the first N rows of every table batch it receives and drops the rest.
- N is a positive integer fixed at construction and carried in the operator's data-only configuration.
- The operator is available on every public surface that exposes built-in operator kinds: the Rust core operator/spec model, v2 project documents (`kind: "head"`), and the Python `PipelineBuilder`.
- Behavior is stateless and identical under micro-batch and streaming runners, including at-least-once redelivery.

## Non-Goals
- No cumulative or global head across batches (no cross-batch row-counting state).
- No array/external (NumPy/JAX) batch support.
- No offset/skip, tail, sample, or fraction-based variants.
- No changes to `BatchingSource` row/byte limits, runner semantics, or the checkpoint format.

## Functional Requirements
- **FR1** - Given a configured N > 0 and an input table batch with R rows, the operator emits a table batch containing the first min(N, R) rows in their original order; all remaining rows are dropped.
- **FR2** - Construction validates N: zero, negative, or non-integer values are rejected with an invalid-argument error naming the offending field.
- **FR3** - The operator declares exactly one required input port `input` and one output port `output`, both `BatchKind::Table` with no fixed schema; wiring a non-table batch to `input` fails port validation at compile time.
- **FR4** - The output batch preserves the input batch's Arrow schema exactly and carries the input's metadata (source identifier, sequence, JSON attributes) unchanged.
- **FR5** - An empty input batch (R = 0) yields an empty output batch with the same schema; a batch with R <= N rows is emitted unchanged.
- **FR6** - The operator's data-only configuration (N) round-trips losslessly through v2 project JSON (`kind: "head"`) and safe YAML import/export; documents with unknown fields on the node are rejected per the strict document rules.
- **FR7** - The Python `PipelineBuilder` exposes a functional `head(...)` method that returns a new builder, consistent with `expression`, `sql`, and `external`.
- **FR8** - The operator honors cooperative run cancellation between items, consistent with existing table operators.
- **FR9** - The compiled plan reports per-node duration and input/output row counts for the head node, consistent with `ExecutionPlan` metrics for existing nodes.

## Non-Functional Requirements
- **Performance** - No measurable regression on the existing benchmark scales (`overhead`, `small`, `standard`, `nightly`) measured with `CALC_FLOW_BENCHMARK_SCALE=<scale> uv run --extra benchmark pytest benchmarks --benchmark-only`; per-batch processing cost is proportional to the number of emitted rows, with no per-row expression evaluation.
- **State and checkpoints** - The operator is stateless: `snapshot()` returns null, `restore()` rejects non-null state, and `reset()` succeeds without effect. Existing checkpoints remain loadable; no checkpoint format or version change is introduced.
- **Compatibility** - The change is additive only: existing v2 project documents, checkpoints, and Python programs continue to work unchanged. `schemas/project-v2.schema.json`, the checked-in Studio OpenAPI document, generated TypeScript API types, and `python/calc_flow/_native.pyi` are regenerated or extended to cover the new kind; nothing is removed or renamed.

## Inputs and Outputs
| Name     | Type                          | Units    | Range / Constraints                      |
| -------- | ----------------------------- | -------- | ---------------------------------------- |
| n        | Rust `usize` / Python `int`   | rows     | integer > 0; fixed at construction       |
| input    | table `Batch` (Arrow)         | rows     | required; any Arrow schema; 0+ rows      |
| output   | table `Batch` (Arrow)         | rows     | first min(n, input rows) rows of input   |

## Acceptance Criteria
- [ ] Unit test: given input batches with 0, R < N, R == N, and R > N rows, when processed, then output row counts are 0, R, N, and N respectively and emitted rows equal the input prefix in order and content.
- [ ] Unit test: construction with N = 0 (and negative or non-integer values via the Python surface) is rejected with an invalid-argument error naming the `n` field.
- [ ] Compile test: wiring an array-kind batch output to the head `input` port fails compilation with a port kind mismatch.
- [ ] Round-trip test: a v2 project containing a `kind: "head"` node serializes and deserializes losslessly in JSON and via YAML import/export, and a document with an unknown field on the node is rejected.
- [ ] State test: `snapshot()` returns null, `restore()` with non-null state fails, `reset()` succeeds, and a checkpoint written before and restored after a run leaves head behavior unchanged.
- [ ] Python test: `PipelineBuilder.head(...)` returns a new builder, the project compiles via `Runtime.compile_project`, and `execute()` truncates each input batch as specified.
- [ ] Runner test: under the micro-batch runner with a forced sink failure, the redelivered batch produces identical head output (idempotent under at-least-once delivery).
- [ ] Verification matrix green for every touched surface per `AGENTS.md`: cargo fmt/clippy/test/doc and `cargo llvm-cov --workspace --all-features --fail-under-lines 90` for the Rust core and PyO3 crate; `pytest python/tests` plus ruff for the Python surface; Studio backend pytest and frontend `npm run sync:api`, `npm run build`, `npm test` when Studio surfaces change.
- [ ] Benchmark report: benchmark scales run before and after the change with no regression beyond run-to-run noise on at least the `overhead` scale; results attached to the PR.
- [ ] Documentation updated: `docs/rust-api.md`, `docs/python-api.md`, and `docs/api-reference.md` describe the `head` operator; JSON schema, OpenAPI document, and generated TypeScript types are regenerated.

## Open Questions
- Engine path: should `head` execute as DataFusion `LIMIT` SQL inside the run-scoped session (strictly honoring "table batches are calculated exclusively by Apache DataFusion"), or is a direct Arrow prefix slice an accepted exception for a built-in structural operator? The choice determines whether a head-only plan creates a DataFusion runtime and reports DataFusion metrics; for `cf-api-designer` and `cf-critic`.
- Studio scope: does this change add a dedicated `head` node to the Studio palette/inspector, or is schema-level support sufficient for now (v2 documents remain editable, validatable, and previewable)?
