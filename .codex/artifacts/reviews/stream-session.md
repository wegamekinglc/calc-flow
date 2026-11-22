# P1 StreamSession review

**Branch:** `feature/stream-session` (worktree `target/worktrees/stream-session`), base `59beb0e9`.
**Reviewed:** product code at `84067482`. HEAD later moved to `3676a502`, which only adds
`benchmarks/stream-session/*` measurement records. I only skimmed those; they are not part of this review.
**Requirements:** `.codex/artifacts/specs/stream-session.md` revision 2 and plan §4.1/§7.2.
Critique checklist items 1–10 are addressed. The synchronous API replaces item 4.
**No PR yet:** no check runs or prior reviews to consult.

## Summary

The branch adds a synchronous, caller-thread `StreamSession` in Rust: a current-thread runtime,
session FIFO edges and a topological drain that reuses `dispatch_message`/`finish_operator`. It
also adds the `_StreamSession` PyO3 state machine, the Python `StreamSession` with
`TableExpr.session`/`Program.session`, and the approved running-loop rule amendment. The core
reuses the runner's per-operator code faithfully. Validation is staged on a cloned
`InputProgress` and committed only after a successful drain. Admission fails closed through an
exhaustive match.

## Build and test results

Commands below were run from the worktree with `CARGO_TARGET_DIR=target/stream-session-target`.

- `cargo fmt --all --check`: clean.
- `cargo test -p calc-flow --lib runtime::streaming::session`: 18 passed.
- `cargo test -p calc-flow --doc StreamSession`: 1 passed.
- `cargo clippy -p calc-flow -p calc-flow-python --lib --tests -- -D warnings`: clean.
- `LD_LIBRARY_PATH=… cargo test -p calc-flow-python --lib session`: 5 passed.
- `maturin develop` plus `pytest python/tests/test_stream_session.py`: 8 passed. The generated
  `_native*.so` was removed afterwards.
- I ran extra probes against the built extension; nothing was committed. They confirmed:
  - an operator failure raises `ExecutionError`, then `RuntimeError("StreamSession failed earlier (…)")`;
  - after `finish`, calls raise `RuntimeError("… is finished")`;
  - `close()` while a call is in flight: that call still returns its rows, and later calls raise "closed";
  - a re-entrant `push` from a UDF raises "busy" and does not deadlock;
  - table and `Batch` inputs with schema metadata are normalized;
  - `RecordBatch` and empty-table inputs work.
- Not run: the full Rust/Python suites, `cargo doc -D warnings`, ruff (CI).
- Incident: one of my earlier cargo invocations ran from `target/stream-session-p1`. Cargo resolved
  that path to the main checkout and overwrote same-hash calc-flow artifacts in the shared target
  dir with `main` code. I repaired it: touched (mtime only) the three crate `lib.rs` files in the
  worktree and rebuilt. Every result above comes from the repaired, worktree-built artifacts.

## Blocking issues

None.

## Should fix (small, recommended before merge)

- **`session.rs` — `ordered_maximum` (deviation 6).** The watermark check compares the first row in
  nanoseconds: `first * unit_nanos <= watermark_micros * 1000`. The runtime converts nanosecond event
  time to `EventTime` with `div_euclid(1_000)` (`EventTime::import_timestamp_unit`). So a row at
  `T µs + 1..999 ns` passes validation, but downstream it is treated as *at* `T`, which is late.
  The intent of B1 is that such a row is a usable validation error, not a session failure.
  - Fix: convert `first` with the runtime's import and require `> watermark`. This is identical for
    s/ms/µs, and checked overflow becomes a validation error.
  - Why not blocking: rolling only reads µs event time (`late_output::identity::event_time`), so this
    is reachable only through ns input → casting expression → rolling.
- **`python/tests/test_stream_session.py` — failed-session path is untested.** No committed test
  covers the binding's `is_failed()` → `Completion::Failed` mapping or the §4.6 rows "operator
  failure → `ExecutionError`" and "call on failed → `RuntimeError`". The inline tests pass explicit
  `Completion` values, so they miss it. If the mapping were inverted on the failure side, the session
  would silently return to `Idle`.
  - Add one test: under `SourceProvidedWatermarks`, push `t=0` with `watermark=t0`, then push a late
    row at `t=0`. Expect `ExecutionError`, then `RuntimeError` matching "failed earlier".

## Nits

- **`crates/calc-flow-python/src/session.rs` — `PyStreamSession::call`.** A panic that escapes
  `operation` (operator panics are contained; staging or export bugs are not) leaves the state `Busy`
  forever. Every later call then says "another call is running". Use a small guard that stores
  `Failed("panicked")` on unwind.
- **`session.rs` — `SessionInput::stage_push` / `drive`.** `EnvelopeCost` is computed three times per
  push: once in `sequenced_source_message` (discarded), once in `validate_message`, once in
  `SessionEdgeSender::push`. Reuse the returned cost and enqueue the input message without
  revalidating. Separately, `drive` clones output batches (`as_data().cloned()`); take ownership
  instead. Both are small next to the measured 127 µs native call, but they cost something on every
  push.
- **`session.rs` — generated-watermark errors.** `GeneratedWatermarkState` errors (underflow,
  EventTime range) reach session callers with runner paths
  (`sources.<id>.watermark_policy.*`), not `session.watermarks.<id>.*`. They are still validation
  errors, so this only affects the field path in the message.
- **Test gap (optional).** Commit-only-on-success is covered for `max_event_time`, not for generator
  state. One `BoundedOutOfOrderness` case would pin it: an oversize push with a later timestamp is
  rejected, and the next push's watermark is unaffected.

## Deviations judged

1. Public `StreamSession::is_failed()`: **accept.** The binding is a separate crate and needs it.
   Document it in `docs/rust-api.md` (or mark it `#[doc(hidden)]` if it should stay off the
   advertised surface).
2. Prepared table `Batch` skips the Python pre-check: **accept.** Only `Port::validate` produces a
   pre-run `Compile` error. The binding maps it to `ValueError` only when the session is not failed,
   and Python retries once, only for a `Batch` that metadata normalization changes. Verified by probe.
3. One source route per binding: **accept.** `external_inputs` is
   `BTreeMap<String, PortEndpoint>` and `project_source_routes` emits one route per binding. Python
   enforces one native ingress per dynamic input with the same check `stream()` uses.
4. Python test 4 drives `_native._StreamSession` directly: **accept.** The busy state belongs to the
   binding, and the symbolic API has no blocking-UDF hook.
5. Field paths (`session.inputs.<id>`, `.watermark`, `.event_time`; schema under
   `session.inputs.<id>`): **accept.** Document them.
6. Watermark comparison in ns: **change.** See "Should fix" above.
7. `large_futures` and `needless_pass_by_value` allows: **accept.** Both carry reasons and match the
   spec'd by-value `push(batch)`.

Open question, `close()` during an in-flight call: **accept the current behavior.** The call
finishes, returns its rows (or raises its own error), and drops the session on the caller's thread.
`close()` returns immediately, and later calls raise "closed". Document it in
`docs/python-api.md`.

## Checked and fine

- Topological drain matches the runner. For non-join operators, `ingress_admission::run` reduces to
  `run_operator_loop`. Per-ingress FIFO goes through `dispatch_message`, then `acknowledge`.
  `finish_operator` runs once after every ingress has seen EOF. Union drains A before B.
- Metadata stamping reuses `sequenced_source_message`: sequence from 0, attributes kept, no cursor,
  empty batches stamped like the runner. The late side-output test matches the runner, including
  `_cf_late_source`/`_cf_late_sequence`.
- `Ordered`:
  - null, decreasing-within-batch and decreasing-across-pushes rows are rejected;
  - rows at or before the watermark are rejected;
  - empty batches are no-ops;
  - an explicit watermark combines with the generated one as a max;
  - `Disabled` rejects explicit watermarks;
  - a regressing watermark is rejected, and an equal one is a no-op.
- Failure handling:
  - post-validation errors poison the session;
  - panics are contained around each dispatch and `finish_operator`, and reset panics at
    construction;
  - panic messages reuse `panic_message`, which bounds them.
- Runtime handling:
  - current-thread runtime with timers only;
  - the Tokio-context guard runs before any work;
  - `Drop` uses `shutdown_background` when inside a runtime context.
- Budgets:
  - the input envelope is checked at staging;
  - every session edge applies `reject_oversize`;
  - `budget()` feeds `effective_output_budget`;
  - the `reject_oversize` extraction from `EdgeQueueCore` does not change behavior.
- Admission:
  - one dynamic input, no statics, no native sequence nodes, no `ExactlyOnce`;
  - Rolling (minus the indicator/MACD layouts), Expression and Union only;
  - the exhaustive `CompiledStreamOperator` match fails closed for future variants.
- Binding:
  - the state mutex is never held across `detach` (temporary guards, always taken with the GIL), so
    lock order is GIL → state;
  - plan-owner GC traversal is kept;
  - outputs are exported per `RecordBatch`.
- Python:
  - policy mapping and `idle_timeout` rejection;
  - static/array input rejection before compile;
  - zero-row outputs come from cached `collect_schema` tables, and aliases share tables;
  - construction rejects a running loop without an `_async` hint, and calls on an existing session
    run inside a loop;
  - AGENTS.md and `.claude/rules/code-style.md` amendments match spec §6.3 word for word.
- Harness: `WarmSessionScenario` drives `Program.session(watermarks={…: SourceProvidedWatermarks()},
  config=…)` and `push({input: Batch}, watermark=…)` through the public API on one thread.

## Documentation consistency (route to cf-doc-writer, spec §9)

1. `docs/python-api.md`: "Stream sessions" next to "Streaming results".
   - Cover `TableExpr.session`/`Program.session`, `push`/`advance`/`finish`/`close`/context manager,
     and the result shapes (all outputs including empty tables; aliases share tables).
   - Policy mapping: `None` → Ordered or Disabled; SourceProvided → Explicit;
     `BoundedOutOfOrderness` ignores `emit_interval`; `idle_timeout` must be `None`.
   - The §4.6 error classes.
   - The running-loop exception and that a large push blocks the loop.
   - `close()` during an in-flight call.
   - Late rows dropped by `late_policy="drop"` are unobservable; checkpoint fields are ignored;
     admission errors name physical node IDs.
2. `docs/streaming-guide.md`: session vs stream. `Ordered` with an explicit watermark rejects rows
   at or before it; the generated and explicit watermarks combine as a max.
3. `docs/rust-api.md`:
   - `StreamSession`, `SessionWatermark` (`#[non_exhaustive]`), `SessionOutput`, `is_failed()`;
   - the synchronous contract and the Tokio-context `InvalidArgument { field: "session" }`;
   - the admission and validation field paths.
4. `docs/runtime-envelope.md`, session path:
   - admission list;
   - per-envelope oversize on every edge, and no queue backpressure (the bounded-FIFO paragraph
     applies to runners and streams only);
   - no checkpoint, even with `checkpointing=true`;
   - panics surface as `Operator { message: "panicked: …" }`, not `TaskPanicked`;
   - errors are not redacted.
5. `docs/api-reference.md`: `calc_flow.StreamSession` and the Rust exports.
6. `docs/introduction.md`: "Choose how to declare and execute".
7. AGENTS.md: Python architecture bullet for `python/calc_flow/session.py`.
8. `CHANGELOG.md`: an Unreleased entry.
9. `docs/plans/2026-11-15-streaming-core-rearchitecture.md`: P1 execution record, including the
   ≤20 µs gate miss and the §7.2 stop-rule outcome.

## Verdict

**Approve with nits.** No blocking issues. I recommend landing the two "Should fix" items (the
µs-floor watermark comparison and the Python failed-session test) before merge; both are a few
lines.
