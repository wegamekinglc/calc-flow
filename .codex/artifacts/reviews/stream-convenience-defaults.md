# Review: convenience-stream defaults (plan decision 5, P1)

**Branch:** `feature/stream-convenience-defaults` (head `e9882531`) → `main`
(`39cab2fb`). **Commits:** `fc3457d8`, `fb539607` (spec/critique), `be547b86`
(Rust per-batch generated watermarks), `e9882531` (Python defaults).
**PR:** none yet, so there are no check runs or prior reviews.
**Requirements:** `.codex/artifacts/specs/stream-convenience-defaults.md`
(§7 overrides earlier sections) and the critique checklist
`.codex/artifacts/critiques/stream-convenience-defaults.md`. The behavior
change itself (decision 5) is approved and not re-examined here.

## Verdict

**Approve with nits** for the code. I found no blocking code defects. Merge
still depends on three items outside the code:

1. `cf-doc-writer` must update the published docs and CHANGELOG (listed
   below). Several published statements now contradict behavior.
2. The PR body must note the benchmark baseline shift.
3. The spec §7.6 paired measurement must be recorded in the PR.

## Correctness review

### Rust per-batch generated watermarks

- **FIFO order, data then watermark (OK).** `evaluate_raw` (`driver.rs:2729-2740`)
  pushes `ForwardData` and then `emit_generated_watermark` pushes
  `ForwardControl(Watermark)` for the same binding. `dispatch`
  (`driver.rs:3482-3530`) sends emissions in order through one
  `outputs[binding]` route. `Progress` emissions are driver-internal and are
  never dispatched. `drive_serial` serializes drains, and `ReadyKey` sorting
  orders several batches in one drain as `[D1, W1, D2, W2]`, which the new test
  asserts.
- **Aggregate evaluation (OK).** The timer and data paths share one helper,
  `emit_generated_watermark` (`driver.rs:2905-2936`). On the data path,
  `AggregateInput::Data` (which reactivates an idle binding) runs before
  `AggregateInput::Watermark`. Multi-source minimum logic in `aggregate.rs` is
  unchanged.
- **Idle timers (OK).** `rearm_idle` on data is unchanged. With a zero interval,
  only the idle timer is on the heap. The test
  `zero_emit_interval_with_idle_timeout_arms_only_the_idle_timer` covers this.
- **Startup and timer invariant (OK).** `watermark_timer_interval()`
  (`prepare.rs:135-143`) returns `None` for zero, so `start_running`
  (`driver.rs:1384-1396`) arms nothing. The zero guard in `next_phase_deadline`
  is kept and can no longer be reached.
- **Validation (OK).** `validate_emit_interval` (`prepare.rs:435-441`) accepts
  only a zero `emit_interval`. Positive values keep the u64-nanosecond check:
  the test now rejects `Duration::MAX`. `max_out_of_orderness` and
  `idle_timeout` keep their positive-duration rules.
- **Checkpoint cut (OK).** Per-batch emission happens in the same drain step as
  `observe_batch`, so at any serialized cut `last_emitted` equals the candidate
  for `observed_max`. The manifest shape is unchanged.
  `emit_interval_ns` is part of the binding identity (`prepare.rs:644`), so a
  positive-interval checkpoint cannot restore into a zero-interval job. It
  fails closed with `CheckpointMismatch`.
- **Durable restore (OK; critique item 1 resolved).** `restore_source`
  (`durable.rs:96-107`) uses `watermark_timer_interval()`, so a zero interval
  gets no watermark deadline and `restore_durable` (`driver.rs:1207`) arms
  nothing. Tests exist at the unit level (`durable.rs`) and the driver level.
- **Failure attribution (OK).** An arithmetic failure on the data path becomes a
  `SelectedItemError` with the data `ReadyKey` and path
  `runtime.progress.generated`. The scratch state is discarded and the whole
  drain fails before dispatch, so no data from that drain is forwarded. Every
  receipt errors. `zero_emit_interval_arithmetic_error_fails_the_whole_drain`
  covers this.
- **No second emission path (OK).** `can_use_direct_source_progress`
  (`runner.rs:3784-3796`) admits only `SourceProvided`, so generated policies
  always go through the coordinated driver.

### Python `checkpointing=None` resolution

- **Every convenience entry point (OK).** `TableExpr.stream` →
  `_stream_table`, `Program.stream` → `_stream_program`, and streaming
  `Program.execute` → `Program.stream` all reach `StreamResults._start`, which
  calls `_resolved_config(config, False)` (`stream.py:293`). Both `_native()`
  calls (`stream.py:294`, `:298`) and the temp-root branch (`stream.py:319`)
  see the resolved bool. Resolving again inside `StreamingRunner` changes
  nothing.
- **Explicit runner unchanged (OK).** `_resolved_config(config, True)`
  (`runtime.py:1320`) applies. The project-plan branch sets `checkpointing=True`
  explicitly (`runtime.py:1314`). The storage checks (`runtime.py:1322-1327`)
  read the resolved bool, and a bare `StreamingRunner(plan, {}, {})` still
  raises. Tested.
- **Native boundary (OK).** `_native()` lowers an unresolved `None` as `True`
  (`runtime.py:784`). The native layer receives only bools and still rejects
  `None` (`test_native_checkpointing_requires_a_boolean`). Values that are
  neither bool nor `None` raise `checkpointing must be a bool or None` at every
  entry point.
- **Default ordered policy (OK).** `_source_config` (`_stream_inputs.py:120-127`)
  covers iterable, materialized and sequence inputs. The test that fails on
  `main` is `test_stream_inputs.py`.

## Blocking issues

None in code.

## Non-blocking findings

1. **Temp-root assertions in default-mode tests now pass trivially.** In these
   tests, `list(tmp_path.iterdir()) == []` holds because no root is ever
   created:
   - `test_stream_results.py`:
     `test_stream_source_failure_wakes_consumer_with_safe_native_error` (`:276`),
     `test_stream_cancels_idle_source_and_cleans_owned_tasks` (`:303`),
     `test_stream_full_output_queue_applies_backpressure_and_cancels` (`:469`),
     `test_stream_start_failure_closes_only_acquired_iterators` (`:501`),
     `test_stream_close_at_native_start_completion_cancels_before_entry_returns`
     (`:1076`), and
     `test_stream_composite_rolling_emits_while_source_remains_open` (`:1136`).
   - `test_symbolic_late_output.py`:
     `test_one_owned_iterator_routes_scripted_watermarks_and_closes` (`:510`).

   Spec §7.5 asked to pin "the other temp-root cleanup tests". Root removal is a
   single `finally` in `StreamResults._close_sources`, and the pinned race tests
   cover it, so the risk is low. However, root cleanup with a live temporary
   root no longer has coverage on the source-failure, idle-cancel and
   start-failure paths.
   **Fix:** parametrize at least the source-failure, idle-cancel and
   start-failure tests over `config=[None, _TEMPORARY_CHECKPOINTS]`. In the rest,
   drop the trivially true assertion or pin the test.
2. **No multi-source case at zero interval.** The aggregate code is untouched and
   the shared helper feeds it, so this is optional. A Rust test with two
   zero-interval bindings in one drain would pin per-binding `ForwardControl`
   ordering and confirm that `Progress` follows the active minimum. Place it next
   to `zero_emit_interval_emits_a_watermark_after_each_advancing_batch`.
3. **ASOF test parametrization is acceptable.**
   `test_convenience_path_workspace_failure_surfaces_reason_detail`
   (`test_asof_stream_error_transparency.py:232`) now accepts either failure
   phase. I reproduced the behavior and confirmed it does not depend on the
   cadence change:

   | Mode                               | Runs | Aggregate | Index |
   |------------------------------------|------|-----------|-------|
   | Checkpoint-off, default policy     | 6    | 3         | 3     |
   | Checkpoint-off, explicit 100 ms    | 8    | 4         | 4     |
   | Checkpointed                       | 6    | 6         | 0     |

   "Aggregate" means "ASOF aggregate workspace" and "Index" means "expiration
   index" in the failure detail.

   The `reason_code == "asof_workspace_limit_exceeded"` assertion stays strict
   in both modes, and the checkpointed case keeps the exact phase. This is a
   pre-existing property of checkpoint-off ASOF: which limit trips first depends
   on how the sources interleave. Suggested low-priority follow-up issue:
   document that the phase detail is not stable under checkpoint-off, or
   sequence `dims` before `facts` in the test.
4. **Underflow test change is acceptable.**
   `test_stream_default_watermark_underflow_fails_without_forward_progress`
   (`test_stream_results.py:1534`) no longer waits for `feed.waiting`. The feed
   never ends, and `feed.read == 1` is asserted, so the failure can only come
   from watermark generation, not EOF. The original intent is kept: no output,
   `watermark_micros is None`, and tasks and the feed are closed.
5. **Performance and measurement.** Each advancing batch now sends one watermark
   control through every operator. Many-small-batch ordered streams pay one
   finalization pass per batch, where a 100 ms tick used to batch them. Run the
   spec §7.6 two-case paired measurement and record it in the PR. Baseline
   shift to note in the PR and CHANGELOG:
   - `benchmarks/arrow_source.py::_collect_tables` (`:139`): an explicit
     `StreamRuntimeConfig(edge_budget=...)` now resolves to checkpoint-off.
   - `benchmarks/test_stream_operators.py:190` and
     `benchmarks/test_program_engine.py:63`: both defaults flip (checkpointing
     and cadence).
6. **Nit: `test_stream_inputs.py`.**
   `test_default_ordered_policy_emits_watermarks_after_every_batch` builds all
   three inputs, including an unused `_feed()` generator, eagerly in a dict.
   Building only the selected kind would be cleaner. The import of the private
   `_source_binding` is required by spec §7.1, so it is acceptable.
7. **Pre-existing, out of scope.** With ruff 0.16.0,
   `uv run ruff format --check .` reports
   `docs/bounded-global-ordering.md`, which this branch does not touch. Every
   `.py` file in the diff is clean.

## Documentation and CHANGELOG (route to cf-doc-writer)

Statements that now contradict behavior:

- `docs/python-api.md:399-407`: "100 ms emission interval". Now the watermark
  advances after every batch (zero interval).
- `docs/python-api.md:448-452`: "Every convenience stream owns temporary managed
  checkpoint storage". Now checkpointing is off by default and
  `StreamRuntimeConfig(checkpointing=True)` opts in to a temporary root.
- `docs/python-api.md:431` mentions temporary checkpoints for the convenience
  path; check whether it still fits.
- `docs/python-api.md` needs new documentation of
  `StreamRuntimeConfig.checkpointing: bool | None = None`:
  - Convenience streams resolve `None` to off, and `StreamingRunner` resolves it
    to on.
  - A config that only tunes other fields no longer turns checkpoints on.
  - The error text changes to "must be a bool or None".
- `docs/streaming-guide.md:848-861`: the default is described as a temporary
  checkpoint root and `checkpointing=False` as the opt-out. Both are now
  inverted.
- `docs/streaming-guide.md:1040-1046`: "100 ms emission interval" and "The
  native timer keeps running while the next iterable read is suspended".
- `docs/streaming-guide.md:1070-1071`: "Durations must be positive". A zero
  `emit_interval` now means per batch.
- `docs/streaming-guide.md:1630`: "Checkpointing is enabled by default". It now
  depends on the entry point, and the `checkpointing=False` convenience example
  is redundant.
- Convenience-stream "temporary checkpoint/state" wording: `docs/design.md:120`,
  `docs/introduction.md:115`, `README.md:66`, `docs/api-reference.md:131-132`,
  `docs/asof-join-guide.md:463`, `docs/relational-frontends.md:42`,
  `docs/projects-guide.md:87`, `docs/symbolic-workflows.md:66`,
  `docs/examples.md:51`, `docs/symbolic-design.md:78`, and
  `examples/README.md:221`. The connector and explicit-runner mentions
  (`docs/examples.md:107`, `examples/README.md:72`/`:206`,
  `docs/connectors/{postgresql,websocket,http}.md`) use explicit temporary roots
  and are unaffected.
- `docs/runtime-envelope.md:85` and `:754-755`: add that a zero interval arms no
  watermark timer and emits behind each advancing batch, and that restore arms
  no watermark deadline for it.
- `docs/rust-api.md`: document the zero-interval meaning on
  `WatermarkPolicy::BoundedOutOfOrderness`. The rustdoc in `continuous.rs` is
  already updated.
- Project-v3: an omitted or zero `emit_interval_ms` now runs per batch instead
  of failing prepare. Document this in `docs/projects-guide.md` and
  `docs/connectors/kafka.md`. The schema is unchanged.
- Add a `CHANGELOG.md` 2026-11-15 entry covering:
  - the per-batch default cadence and the zero-interval sentinel;
  - checkpoint-off convenience defaults, and that checkpoint-off jobs admit
    operators without a proven checkpoint lifecycle;
  - the `checkpointing` field type change (`bool` to `bool | None`), which
    affects downstream Studio readers;
  - explicit runners unchanged;
  - project-v3 omitted `emit_interval_ms`;
  - the benchmark baseline shift.

## Verification (this review, isolated target dir)

Commands used
`CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/stream-convenience-defaults-target`.

- `cargo test -p calc-flow --lib runtime::streaming::progress`: **103 passed**,
  0 failed. This includes all six new tests: three `zero_emit_interval_*`,
  `durable_restore_of_zero_emit_interval_*` in both `driver` and `durable`, and
  `generated_policy_accepts_zero_emit_interval_as_per_batch`.
- `cargo clippy -p calc-flow --lib --tests -- -D warnings`: clean.
  `cargo fmt --all --check`: clean.
- `uv sync --extra dev --no-install-project` and `uv run --no-sync maturin
  develop`: installed.
- `JAX_PLATFORMS=cpu uv run --no-sync pytest -q` on `test_stream_inputs.py`,
  `test_checkpointing.py`, `test_stream_results.py`,
  `test_stream_entrypoints.py`, `test_asof_stream_error_transparency.py`,
  `test_arrow_stream_inputs.py` and `test_continuous_runtime.py`:
  **178 passed**.
- The other changed test files (`test_bounded_analytic_over.py`,
  `test_bounded_global_distinct.py`, `test_bounded_global_ordering.py`,
  `test_snapshot_sql.py`, `test_symbolic_late_output.py`): **149 passed**.
- I repeated the eight timing-sensitive tests (per-batch rolling, underflow, ASOF
  workspace, explicit per-batch runner, equal-times frontier, composite rolling)
  in ten runs; all ten passed. Eight more runs failed at import, before any test
  ran: a concurrent agent deleted the shared `_native*.so` mid-loop. That was an
  environment problem, not a test failure. I rebuilt and repeated.
- `uv run ruff check .`: clean. `ruff format --check`: see finding 7.
- `git diff --exit-code main -- schemas/project-v3.schema.json`: no diff.
  `git diff --check main`: clean.
- I removed the generated `python/calc_flow/_native*.so`.
- Not run: the full Rust and Python suites, coverage, examples and benchmarks.
  There is no PR, so there is no CI.
