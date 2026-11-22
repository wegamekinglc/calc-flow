# P1 convenience-stream defaults — specification

Date: 2026-11-15. Source baseline: `main` `39cab2fb`.
Branch: `feature/stream-convenience-defaults`. Plan:
streaming core re-architecture (`docs/plans/2026-11-15-streaming-core-rearchitecture.md`
on `feature/stream-session`), decision 5 and phase P1 (§7.2).
Status: revision 1; ready for critique.

## 1. Objective

Implement user decision 5 of the plan (approved 2026-11-15):

1. Convenience streams (`TableExpr.stream`, `Program.stream`, streaming
   `Program.execute`) default to **checkpointing off**.
2. Ordered event-time inputs advance their watermark **after every data
   batch** instead of on a 100 ms timer.

Both are visible behavior changes: results become visible earlier, and
convenience streams no longer create temporary checkpoint storage unless
asked to. Explicit `StreamingRunner` jobs and `StreamRuntimeConfig()` keep
their current defaults.

## 2. Current behavior (evidence)

- `python/calc_flow/stream.py:291` uses `_runner_config(self._request.config)`,
  i.e. `StreamRuntimeConfig()` with `checkpointing=True`
  (`python/calc_flow/runtime.py:773`); `stream.py:316-319` then creates a
  temporary managed checkpoint root. Docs: `docs/design.md:120`,
  `docs/python-api.md:448-452`, `docs/streaming-guide.md:848-860`.
- Checkpoint-on jobs take slower operator paths (plan §3.3: rolling loses
  in-place updates and keeps `RollingHistories`; `window_sum` 10M rows
  1,010 ms off vs 1,616 ms on-but-untriggered).
- `python/calc_flow/_stream_inputs.py:115-126`: a declared event-time input
  with no policy gets `BoundedOutOfOrderness(col, 1 µs, 100 ms)`. The native
  generator only emits on its timer (`runtime/streaming/progress/driver.rs:1380-1395`
  arms it; `driver.rs:2846-2905` fires it). `emit_interval` must be positive
  (`progress/prepare.rs:392-397`, `validate_duration`; `progress/generated.rs:152`).
  Docs: `docs/python-api.md:399-407`, `docs/streaming-guide.md:1040-1046`.

## 3. Design

### 3.1 Per-batch generated watermarks (Rust core)

`WatermarkPolicy::BoundedOutOfOrderness { emit_interval: Duration::ZERO, .. }`
becomes valid and means *emit after every data batch*:

- Validation (`normalize_generated_policy`) accepts a zero `emit_interval`;
  `max_out_of_orderness` and `idle_timeout` keep their positive-duration rules.
- The driver arms no watermark timer for a zero interval (startup arming at
  `driver.rs:1380-1395`, re-arming at `driver.rs:2890-2905`, durable restore at
  `driver.rs:1210-1217`).
- In the data branch (`driver.rs:2700-2732`), after `observe_batch` and the
  `ForwardData` emission, a zero-interval binding calls the generator's
  candidate computation (the same `on_timer` arithmetic) and, when it
  advances, appends `ForwardControl(Watermark)` plus the aggregate evaluation,
  exactly as the timer branch does (`driver.rs:2870-2885`). The watermark
  therefore follows its data in FIFO order on the source edge.
- Empty batches do not advance (unchanged `observe_batch`). Multi-source
  minimum, idle handling and generated-watermark arithmetic are unchanged.
- Durable progress keeps storing `Some(emit_interval)`; a zero interval
  restores with no watermark deadline. Checkpoint format and fingerprints are
  otherwise unchanged (the interval is already part of the policy identity).

Rustdoc on `WatermarkPolicy::BoundedOutOfOrderness` documents the zero
meaning. The Python `BoundedOutOfOrderness` dataclass accepts
`emit_interval=timedelta(0)` through its existing lowering
(`crates/calc-flow-python/src/continuous.rs:854`).

### 3.2 Convenience defaults (Python)

- `_source_config` (`_stream_inputs.py:115-126`): the default ordered policy
  becomes `BoundedOutOfOrderness(col, timedelta(microseconds=1), timedelta(0))`.
  This applies to iterables and materialized Arrow inputs alike.
- `StreamResults._start` (`stream.py:286-331`): when the request's `config`
  is `None`, use `StreamRuntimeConfig(checkpointing=False)`; an explicit
  `config` is honored unchanged (including `checkpointing=True` with its
  temporary root). Update the `StreamResults` docstring.
- `StreamRuntimeConfig`'s own default stays `checkpointing=True`, so explicit
  runners are unaffected.

### 3.3 Out of scope

`StreamSession` (separate P1 PR); the runner's single-source direct fan-out
for generated policies; any change to explicit runner defaults, project-v3
defaults or `SourceBinding` factories.

## 4. Tests (each starts red)

Rust (`progress` unit tests next to the existing driver/prepare tests):

1. `prepare` accepts a zero `emit_interval` and still rejects a zero
   `max_out_of_orderness` (update `watermark_policy_rejects_invalid_durations`,
   `progress/prepare.rs:779-800`).
2. A zero-interval generated binding emits `Watermark(max - delay)` right
   after each advancing data batch, in FIFO order after the data, and none for
   an empty or non-advancing batch; no watermark timer is armed.
3. Durable restore of a zero-interval binding arms no watermark timer and
   keeps emitting per batch.

Python:

4. A default `TableExpr.stream` over an async iterable with a rolling mean
   yields each batch's finalized rows before the next batch is supplied
   (gate the iterable on the consumer; no sleeps), and the latest timestamp
   stays open until a larger one arrives.
5. A default convenience stream creates no temporary checkpoint directory
   (observe `tempfile` usage or the job's checkpoint status); an explicit
   `StreamRuntimeConfig(checkpointing=True)` still does.
6. `BoundedOutOfOrderness(col, delay, timedelta(0))` passed explicitly works
   for an explicit `StreamingRunner` source binding.
7. Update existing tests that assert the 100 ms default or default temporary
   checkpoints (search `python/tests` for `100`/`milliseconds=100`,
   `checkpoint` status assertions on convenience streams) to the new
   defaults, keeping their original intent.

Focused commands:

```bash
cargo test -p calc-flow --lib runtime::streaming::progress
cargo clippy -p calc-flow --lib --tests -- -D warnings
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests/test_stream_results.py python/tests/test_arrow_stream_inputs.py python/tests/test_stream_entrypoints.py -q
uv run ruff check . && uv run ruff format --check .
git diff --exit-code -- schemas/project-v3.schema.json && git diff --check
```

## 5. Documentation (via `cf-doc-writer` after review)

`docs/python-api.md` (399-407, 448-452), `docs/streaming-guide.md`
(848-860, 1040-1046), `docs/design.md:120`, `docs/runtime-envelope.md`
progress/timer section, `docs/rust-api.md` watermark policy, `CHANGELOG.md`.

## 6. Measurement

Not a performance PR. The plan's P1 measurement covers sessions; the
convenience-stream latency effect is reported qualitatively (results no
longer wait for a 100 ms timer). No local benchmark is run for this PR.

## 7. Implementation decisions after critique (lead, 2026-11-15)

The [critique](../critiques/stream-convenience-defaults.md) is applied as an
implementation checklist; no further review round. Decisions:

1. **Red-first default test (blocking item).** Add a unit-level test that
   captures the `SourceBinding.watermark_policy` produced by
   `_source_binding` for iterable, materialized and sequence inputs and
   asserts `emit_interval == timedelta(0)`; it fails on main. Keep §4 test 4
   as a non-red end-to-end check.
2. **Durable restore.** A zero interval must restore with no watermark
   deadline: filter zero in `restore_source` (`progress/durable.rs:106-111`,
   `rearm`) so `driver.rs:1207` arms nothing. Add a durable unit case. §3.1's
   "keeps storing Some(emit_interval)" is corrected accordingly.
3. **Project-v3 omitted `emit_interval_ms`.** Its serde default is 0
   (`config.rs:351`); projects that omit it now run per batch instead of being
   rejected at prepare. Document this; no schema change.
4. **Explicit config must not silently re-enable checkpoints.** Python
   `StreamRuntimeConfig.checkpointing` becomes `bool | None = None`
   ("mode default"): convenience streams resolve `None` to `False`; explicit
   `StreamingRunner` resolves `None` to whether checkpoint storage was passed.
   Explicit `True`/`False` are honored and validated as today. Resolve `None`
   before every `_native()` use. The Rust `StreamRuntimeConfig` default is
   unchanged. Test both resolutions.
5. **Existing tests.** Pin temp-root lifecycle tests (e.g.
   `python/tests/test_stream_results.py:718`, `:911`, `:1017` and the other
   temp-root cleanup tests) to `StreamRuntimeConfig(checkpointing=True)` so
   they keep testing what they were written for.
6. **Measurement.** Not part of the implementation step. After review, run at
   most two paired cases with an existing maintained tool (a many-small-batch
   ordered rolling stream and a large-batch control) to attribute the
   per-batch watermark cost; recorded in the PR.
7. **Docs.** `cf-doc-writer` sweeps by grep (`100 ms`, `temporary`
   checkpoint wording, `emit_interval`) including `docs/introduction.md` and
   `README.md`; document the zero sentinel (zero means per batch) and that
   checkpoint-off convenience jobs admit operators without a proven
   checkpoint lifecycle.
8. **Decision 4 refinement (final).** Explicit `StreamingRunner` resolves
   `checkpointing=None` to `True`, keeping its current behavior (missing
   checkpoint storage still raises). Convenience streams resolve `None` to
   `False`. One shared helper resolves before any `_native()` use;
   `_native()` on an unresolved config treats `None` as `True`; the native
   layer only receives bools; non-bool, non-`None` values still raise
   `TypeError`. Update `python/tests/test_checkpointing.py:14-30`. This
   resolves critique items 12–14.
