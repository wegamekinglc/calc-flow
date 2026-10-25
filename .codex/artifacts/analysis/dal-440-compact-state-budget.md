# DAL-440 / T3: compact SQL state budgets

Baseline: `926c4f3cd182d163887c16107d5e345c651319b7`, merged T2 / PR #497.
This stage reuses its ordered-last state, layout-5 identity and restore paths.

## Contract and accounting

`StreamRuntimeConfig.sql_compact_state_budget` is optional and defaults to
`None`. Each aggregate SQL node gets an independent ownership pool layered over
its existing DataFusion pool. `max_rows` counts live groups, including null-only
groups. Byte leases conservatively pay for plans, encoded keys, owned key/value
and ordering scalars, results, container capacity, dirty tracking, checkpoint
metadata, encoded segments and bounded logs. Old state and transaction/restore
candidates pay simultaneously; failed candidates release their leases.

Input/output buffers, expression workspaces, proof headroom, decoded validation
buffers and serialization scratch retain their separate engine reservations.
IPC buffers pay the engine pool through their ownership lease once. This is an
owned-state budget, not a bound on process RSS or serialization peak memory.

The old `sql_state_budget` still counts cumulative accepted input rows and
logical bytes. Both budgets apply when configured. Neither budget enters SQL
identity or the semantic plan fingerprint. An absent new budget does not add a
field to the historical runtime-config hash; configured budgets enter only that
diagnostic hash.

An explicit compact budget rejects unsupported aggregate shapes, registered
UDFs and unsupported execution models. Declared schemas permit preflight
admission checks; otherwise admission happens before the first nonempty input
can commit or emit. Nonaggregate SQL is unaffected. Group admission follows the
input predicate, so filtered-out keys do not consume the group limit.

Full and delta restore rebuild native groups and their fees from actual keys,
typed scalars, capacities and segment contents. Checkpoint metadata cannot supply
an authoritative ownership fee. Group checks apply before importing new keys;
the byte pool pays before allocation. Nonempty retained checkpoints conflict
with an explicit compact budget and require a new lineage plus replay. Restore
errors leave the previously installed state intact.

Rust runtime/runner wiring and the Python config use the existing `StateBudget`.
PyO3 accepts the paired `sql_compact_state_max_groups` and
`sql_compact_state_max_bytes` fields, with positive, JSON-safe integer bounds.
The existing `_native.pyi` mapping signature remains valid and unchanged.

## Focused validation

Builds reuse the preserved task's Cargo target tree with
`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0`. Binding builds/tests use
CPython 3.13.9 via `PYO3_PYTHON`; native test execution also sets the interpreter's
library directory in `LD_LIBRARY_PATH`.

- `cargo test --locked -p calc-flow --lib -- compact_budget test_checkpointing_hash_records_disabled_mode_without_changing_the_plan test_compact_storage_two_roundtrips_continue_and_recapture_every_prefix test_global_float_where_coalescer_boundaries_and_async_cold_cuts --nocapture`:
  12 passed, 2.26 seconds test execution. Covers group/byte boundaries, candidate
  copies, filtering, old quota, unsupported aggregates, row-local SQL, sync/async
  capture rejection, held snapshots, full/delta restore and atomic rejection.
  Includes the historical fixed hash and SUM/COUNT/coalescer recovery controls.
- `cargo test --locked -p calc-flow-python --lib test_compact_budget_native_pairs_and_parameter_validation -- --test-threads=1 --nocapture`:
  1 passed; absent/present pairs, either missing field, bool/float/string,
  zero/negative and JSON-unsafe limits are checked.
- `cargo clippy --locked -p calc-flow -p calc-flow-python --lib --tests -- -D warnings`:
  passed on the final Rust content.
- `RUSTDOCFLAGS="-D warnings" cargo doc --locked -p calc-flow -p calc-flow-python --no-deps`:
  passed.
- `uv tool run --from 'maturin>=1.14.1,<2.0' maturin build --locked --profile dev --out target/wheels --interpreter /home/wegamekinglc/anaconda3/bin/python`:
  built a local test wheel. Installed only into `target/python-current`, with
  imports verified to use that wheel's Python code and native extension.
- `PYTHONPATH=target/python-current JAX_PLATFORMS=cpu python -m pytest python/tests/test_stream_results.py -q -k 'compact or preserves_retained_epochs_position or uses_configured_state_budget'`:
  five unchanged cases passed. The multi-node test initially supplied an iterable
  where `Program.stream` requires a named mapping; only its fixture was corrected.
- `PYTHONPATH=target/python-current JAX_PLATFORMS=cpu python -m pytest python/tests/test_stream_results.py -q -k test_stream_compact_budget_applies_independently_to_each_aggregate`:
  corrected multi-node case passed. Together these provide six Python cases,
  including actual native group rejection and per-node independent budgets.
- Rust formatting, changed Python Ruff lint/format checks, whitespace and
  unchanged Cargo.lock/project-schema checks passed. Lizard found no new or
  increased function above complexity 8; all three new modules satisfy it.

T2's unchanged semantic oracle evidence is reused. Full regression, platform
checks and the existing coverage gates remain with CI. Broader corruption and
fault-injection acceptance stays in T4; current recovery already enforces the
new budget independently of that later stage.

## Preserved work

The original 21 tracked changes and four untracked files remain in the original
DAL-431 checkout on `agent/cf-implementer/5ceb4eb9b8b4`. Stash
`32edcee9716fcd7a4d0cb595d69b86967fdb7c37` and its patch/archive backups remain
available. Relevant budget wiring was adapted selectively onto the merged T2
baseline; unrelated historical WIP and its six diagnostics are not part of this
stage. No native binary or build artifact is committed.

## CI restore-refund regression on 2026-10-26

The coverage job failed in
`test_persistent_dictionary_duplicate_ids_preserve_prior_payload_error` before
measuring the coverage floor. Its Managed route compared a pool with no idle
executor against a pool with a reusable gather generation. Global worker
capacity pressure can retire the checkpoint's executor; Managed restore can
create another one. Generation ownership is distinct from candidate ownership.
The installed state, writer and attempt fees were unchanged, and the existing
final `close_and_drain`/zero-pool assertions still passed before the outer
comparison failed. Join/gather production code was unchanged by the T3 commit.

Unmodified isolated runs passed with a warm generation, both normally and with
coverage instrumentation. To reproduce the CI condition deterministically, the
fixture now explicitly retires the idle checkpoint executor before recording
its baseline. Before the matching post-restore wait was added, an instrumented
four-CPU run reproduced exactly `50572 -> 69740` pool bytes and
`(16384, 0, 0) -> (16384, 19168, 0)` control fees, retaining the expected prior
payload error. A two-CPU control reproduced the same failure with its smaller
17776-byte generation.

The final fix compares both observations after executor retirement. A test-only
helper requires an empty work slot and a completed initialization, invokes the
existing generation retirement transition, and awaits the existing cleanup
notification. Worker joins and generation credit release precede that cleanup
completion. Job admission stays open. All original assertions, including exact
funding equality, state/history identity, error priority, continued processing,
validation-before-dictionary-grant and final zero-pool checks, remain unchanged.
There is no added sleep, retry, tolerated difference, production behavior change
or gate exemption.

Reproduction/validation uses Linux, Rust 1.94.0, cargo-llvm-cov 0.8.7,
`CARGO_BUILD_JOBS=3`, and `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0`.
Instrumented runs use the CI core feature selection (`--all-features`) and
`cfg(coverage)`/LLVM coverage flags. Four-CPU affinity gives the same generation
charge observed in CI; connector services are not exercised by these core tests.

- `cargo llvm-cov test --locked -p calc-flow --lib --all-features --no-report -- persistent_dictionary_restore_tests test_close_and_drain_waits_for_generation_owner_release cleanup_observer_keeps_escaped_output_funded_until_install --nocapture`:
  10 passed, 0.05 seconds execution under four-CPU affinity, including the exact
  failing case, all five duplicate-ID routes, managed/direct restore, prior
  payload/dead-row error precedence, cancellation, refunds and generation cleanup.
- `cargo test --locked -p calc-flow --lib --all-features duplicate_id_tests -- --nocapture`:
  2 passed, 0.03 seconds execution without coverage instrumentation.
- `cargo clippy --locked -p calc-flow --lib --tests --all-features -- -D warnings`,
  `cargo fmt --all --check` and `git diff --check`: passed. Static comparison also
  confirms that every original assertion line is unchanged, no function gains
  complexity above 8, and CI workflows, coverage scripts, Cargo.lock and the
  project schema are unchanged.

The full matrix and 90% combined coverage floor remain CI responsibilities;
partial local instrumentation does not establish that floor. T3's accepted
budget, Python/wheel and other unaffected validation evidence remains valid.
