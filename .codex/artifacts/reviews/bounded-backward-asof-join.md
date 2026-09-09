# Branch Review: bounded backward ASOF Join

**Branch:** `feature/bounded-backward-asof-join`
**Implementation base:** `eda1583751abbd1ca4d246fcb8ee6b70f57d9b09` (#259)
**Review state:** final source review of the isolated feature worktree;
focused implementation, final wheel, affected consumer, and generated-contract
checks complete. Approved for the requested PR handoff; CI is not yet green.

## Summary

The change adds the independent `stream_asof_join@1` operator, native typed
predecessor selection and bounded DataFusion output materialization, managed
checkpoint recovery, Python expression/builder declarations, and Studio project
and status contracts. Review cross-references the same-slug specification
FR1–24/AC1–18, API note, and critic's C1/C2/C3 requirements.

The assigned implementation worktree is already isolated from the user's
primary checkout and remains implementation-owned. No feature PR or immutable
feature commit exists at this review stage, so there are no feature check runs
or prior GitHub reviews to consult. The parent verified that #259 merged as
`06c0223` on `main`, with an identical source tree to the implementation base.
Publication must attach the reviewed feature to its final commit and PR base.

## Build and Test Results

The following are actual focused local results reported by the implementation
and independent test owners and reconciled with the reviewed source. They are
not full-suite or CI results. Cargo checks use the isolated worktree with
`CARGO_HOME` at the repository's `target/cargo-home`,
`CARGO_TARGET_DIR=target/cargo`, `CARGO_PROFILE_DEV_DEBUG=0`, and
`CARGO_INCREMENTAL=0`.

- Rust: **Passed, focused scope.**
  `cargo test --locked -p calc-flow --test stream_asof_join_validation`:
  **7 passed**; `stream_asof_join_properties`: **5 passed**;
  `stream_asof_join_boundaries`: **3 passed**;
  `stream_asof_join_state`: **21 passed**.
  `cargo test --locked -p calc-flow --lib operator::asof::`:
  **10 passed**, including output-edge failure, counter overflow, cancellation,
  shared-pool release, malformed IPC, and typed-encoding oracle checks.
- Runtime: **8 operator-task and 8 managed-runner test functions passed**
  through focused invocations under
  `runtime::streaming::{operator_task,runner}::tests::asof_tests`.
  The later focused backpressure rerun also passed its new assertions that
  edge queue depth, charged rows, and charged bytes return to zero.
- Recovery validation:
  `cargo test --locked -p calc-flow --test stream_asof_join_restore_corruption`
  passed **4 tests / 29 mutation vectors**. The affected metadata matrix also
  passed its focused rerun after the final early shape guard.
- Resources:
  `cargo test --locked -p calc-flow --test stream_asof_join_resources -- --nocapture`
  passed **5 tests**, comprising four unchanged-size traces and one allocation
  regression. The run took **88.02 seconds** in this local development profile.
- Compatibility: frozen Rust inner vectors **5 passed**; frozen Python inner
  vectors **5 passed**. Original inner validation **16 passed** and state
  **10 passed** at the compatibility baseline. Fixtures were not re-recorded.
- Python: the initial focused builder, symbolic ASOF, actual stream, and
  existing relational DAG command passed **128 tests**, followed by one added
  input-transformation case. There are **25 actual native stream cases**.
  Review fixes add two alternate-time rejection cases, six lost-ordering
  rejection cases, and two valid-ordering cases; all passed their focused
  reruns. After installing the final freshly built native wheel, the five
  ASOF/relational/inner targets passed **144 tests**; example22 again exited 0,
  withholding output at equal watermark 105 and producing [10.2, None].
- Studio backend: **14 ASOF metrics/SSE/OpenAPI checks**, **5 ASOF project
  round-trip/validation checks**, **4 existing inner validation checks**, and
  directly affected status/capability/error consumers passed. The final-wheel
  repeat of the two new ASOF backend modules passed **19 tests**.
- Studio frontend: **12 inspection tests** and **6 SSE-hook tests** passed;
  targeted TypeScript compilation and ESLint passed. The SSE test parses actual
  JSON carrying full-width integer strings, null watermarks, and booleans.
- Format/lint: final native library Clippy and targeted library/resource/
  corruption-target Clippy passed with `-D warnings`; owned Rust formatting,
  affected Python Ruff checks/formatting, and the complexity ratchet passed.
  `cargo clippy --locked -p calc-flow-python --lib --no-deps -- -D warnings`
  passed for the binding. Its initial relative interpreter-path invocation
  failed before analysis and was corrected to the managed absolute interpreter.
  A subsequent invocation including dependency lint failed on the unchanged
  connector `options::positive_option` dead-code diagnostic with the default
  file-only feature selection. The reviewer confirmed connector source is
  identical to the base and its call sites are behind other connector features.
  Core and changed native targets were linted separately. This dependency-only
  limitation remains recorded; no source suppression or baseline change was used.
- New failures: **None unresolved in the reviewed source.** Earlier RED and
  diagnostic failures are explained below. Final package/contract and scoped
  binding lint checks passed; the existing dependency-only warning is identified
  separately above.
- Regressions: **None observed in the focused scope.** Full regression,
  cross-platform runs, Rust 90% coverage, Studio 85% coverage, and required
  performance gates remain CI obligations.

Additional local detail is in `target/asof-evidence/{stage-0,stage-1,native-final,resources}.md`
and `target/asof-python-evidence/summary.md`. This committed review includes
the essential results and limitations so it does not depend on ignored files.

## Blocking Issues

**None.** Final wheel, generated-contract repeat, affected consumer verification,
and scoped binding lint passed. All review findings and acceptance-test gaps
below are closed.
The following blocking findings were resolved and their final source reread:

- **`operator/asof/workspace.rs`, `admission.rs`, `duplicate_fallback.rs`:**
  schema/type/null → late → duplicate → budgets now holds even when an identity
  reservation fails. A cancellable typed fallback detects duplicates without
  a temporary index. Signed/unsigned extrema and UTF-8 block boundaries are
  compared against Arrow's independent `RowConverter`.
- **`operator/asof/workspace.rs`, `metadata.rs`, `checkpoint.rs`:**
  preflight now includes schema/field metadata; fingerprints are cached;
  fixed-shape metadata validation runs before borrowed serde decoding.
  Malformed 2 MiB values no longer cause a large diagnostic allocation.
  Restore charges temporary identity-only UTF-8 validation as well as payload
  decoding. Empty terminal snapshots restore with a valid one-byte limit.
  Prepared and decoded workspace reservations remain owned until installation.
- **`operator/asof/codec.rs`:**
  trusted canonical schema-frame digests precede Arrow conversion. The complete
  IPC sequence is exactly Schema → one RecordBatch → EOS; compression,
  additional schemas/batches, variadic buffers, invalid flat field nodes,
  overlapping/misaligned buffers, and out-of-body lengths are rejected before
  Arrow can allocate or reach the identified assertions. Arrow 58.3.0 sorts
  metadata keys when encoding, so equivalent metadata maps produce stable
  schema digests.
- **`operator/asof/checkpoint.rs` — `validate_progress`:**
  restored identity-only entries must satisfy the same safe payload-GC
  threshold as normal execution. Already-finalizable pending rows, impossible
  frontiers, expired tombstones, inconsistent terminal state, and native-owned
  progress are rejected before installation/readiness.
- **`operator/asof/output.rs` — `CandidateTables`:**
  RAII cleanup deregisters both candidate tables on normal return, failure,
  and dropped futures. Focused tests retain and reuse the runtime after
  cancellation and assert that registrations and pool reservations are gone.
- **`operator/asof/schema.rs`:**
  the explicit flat payload allowlist rejects nested, dictionary, run-end, and
  view types, including the `FixedSizeList<Null, 1_000_000>` amplification case.
  Invalid flat type parameters are rejected. Thirty-three representative flat
  types pass actual IPC, checkpoint/restore, and DataFusion output.
- **`operator/asof/spec.rs`, `project_store/asof.rs`:**
  raw `late_policy` is required through both raw validation and direct serde;
  constructors still default to `Error` and emit the explicit wire field.
- **`symbolic/analyzer.py`:**
  ASOF → inner consumes the proven event-time column. Post-ASOF stateful
  filter/with-columns reject lost event-time/entity/sequence evidence.
  The reviewer independently reproduced the missing-ordering failure before
  the parent supplied the source correction and focused RED/GREEN evidence.

The resource test also exposed quadratic checkpoint copying from per-field
`reserve_exact`: a 1,000-row state charged 2,972,088 bytes caused
5,125,251,020 cumulative allocated bytes in one preparation. Exact preallocation
under the existing reservation fixed the regression; all five resource tests
then passed. The earlier interrupted traces ended with exit 130 and are not
counted as passing runs.

## Style Issues

No unresolved convention violation was found. New behavior stays in the native
engine; Python declarations and caller-owned containers remain immutable.
There are no added dependencies, weakened unsafe-code restrictions, complexity
waivers, or executable project payloads. Changed Markdown tables were aligned;
local Markdown target checks and `git diff --check` passed.

## Test Coverage and Requirement Reconciliation

| Requirements                | Acceptance | Source and focused evidence                                                                                                                                                                                                                                   |
|-----------------------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FR1–4, FR20, FR24           | AC1, AC15  | Independent kind/state/schema definitions; strict Rust/raw/Python declarations; validation7; flat payload round-trip; immutable builder tests; frozen inner vectors and structural contract comparison                                                        |
| FR5–7                       | AC2–3      | Inclusive latest predecessor, unmatched left preservation, right reuse, typed ties, canonical order; native properties5 and boundaries3; eight oracle permutations; actual native Python streams                                                              |
| FR6, FR8–10, FR23           | AC4–6      | Native time extrema and EOF; operator-task8; progress without aggregate emission; idle/reactivation; data before C−1 frontier; actual downstream rolling, cross-section, and Union                                                                            |
| FR11–12                     | AC7–8      | Atomic late/duplicate precedence, equality on-time, retained identity-only duplicate detection, post-EOF rejection; real L105/R100/T10 with left WM1000/right WM90; conservative GC and restore threshold checks                                              |
| FR13–14                     | AC9, AC16  | State rows/bytes and separate aggregate workspace; tiny-slice and metadata allocation vectors; repeated wide candidate chunk reduction; actual output-edge failure and MAX-history overflow; cancellation/pool/edge release; resource5                        |
| FR15–17                     | AC10–11    | Native state21; private progress checks; managed runner8 across pending/publication/fault/terminal cuts; missing/corrupt persisted segments; restore corruption4 with 29 version/config/schema/counter/gauge/order/duplicate mutations                        |
| FR17–19, FR24               | AC12       | Reset preserves shared snapshots; deterministic logical counters and terminal equalities; PyO3 integer-extrema check; backend ASOF14 and frontend SSE6 preserve i64/u64 decimal strings, nulls, and bools                                                     |
| FR20–23                     | AC13–14    | Root/fluent/advanced builders; 25 actual native stream cases; logical-name binding and SourceBinding policy ownership; capability fail-closed matrix; fan-out owner identity; approved compositions and explicit rejected paths; explain/optimizer boundaries |
| FR24 and verification scope | AC15, AC17 | Frozen native/Python inner vectors, existing inner directed tests, Studio import/view/save, generated schema/OpenAPI/TS comparison, scoped format/lint/type checks; final wheel and second contract generation passed                                         |
| Documentation and delivery  | AC18       | Example22 and synchronized guides explain finality, late policy, resource limits, and delivery; final specialist verdict and PR publication/one CI snapshot remain publication steps                                                                          |

The specification's former planned runtime/Python test target names were
reconciled to the actual inline runtime modules and focused Python files.
Equivalent coverage is recorded above; nonexistent target names are not
reported as executed.

The 29 corruption mutations cover state/layout/accounting/encoding versions,
actual tolerance and schema-metadata fingerprint changes, forged gauges and
charges, logical counters/sequence/terminal/progress, and duplicate/descending
left identities, right keys, and right identities. Each checks the failure
category, unchanged live status and checkpoint bytes, and the original later
matching answer.

## Resource Evidence

All four deterministic traces define 100,000 inputs, split 50,000/50,000 across
exactly 1,000 keys. Schema is non-null UInt64 key, UTC-us time, Int64 sequence,
and UTF-8 payload. Batches contain at most 256 rows; rounds advance 15 seconds
through left time 735,000,005us. Tolerance is 60,000,000us. Limits are 100,000
state identities, 67,108,864 charged state bytes, a separate 67,108,864-byte
workspace ceiling, and output edges of 10,000 rows / 67,108,864 bytes.
Dependencies are DataFusion 54.0.0, Arrow 58.3.0, Tokio 1.52.3, and
allocation-counter 0.8.1.

| Trace     | Accepted inputs | Final output rows | Peak state rows | Peak charged bytes | Peak segment capacity | Outcome                            |
|-----------|-----------------|-------------------|-----------------|--------------------|-----------------------|------------------------------------|
| Advancing | 100000          | 50000             | 6000            | 17368088           | 7584024               | Oracle matches; EOF releases state |
| HotKey    | 100000          | 50000             | 6000            | 17368088           | 7584024               | Oracle matches; EOF releases state |
| Stalled   | 24000           | 1000              | 23000           | 66486088           | 29123024              | Explicit state-limit failure       |
| Wide      | 3000            | 1000              | 2000            | 38486088           | 18859024              | Explicit workspace-limit failure   |

Payloads are 8 bytes; Wide uses 16,384 bytes after the first round.
HotKey assigns 900/1,000 rows to key 0 after its first uniform round.
Advancing/HotKey advance both watermarks to round time + 6us;
Stalled/Wide freeze the right watermark at 6us. Failing traces stop at the
first failure: their complete emitted prefix is exactly logical left identities
1–1,000, equal to the oracle, with no partial admission from the rejected Batch.
All successful outputs match the independent oracle without duplicate left IDs.
Successful per-watermark right history is at most 4,000 rows.

Snapshot capacities are observed owned segment allocations, not every native
allocation. Reset invalidates weak references to the prepared segments and
clears state gauges. Current-thread heap peaks were respectively 30,995,050,
30,836,032, 103,647,076, and 63,329,810 bytes; these include the harness,
Arrow, DataFusion, and operator work. **They are neither RSS nor workspace
high-water measurements. Workspace peak was not measured.** Its ceiling,
failure behavior, and release are established by shared-pool reservation paths
and the focused private lifecycle/allocation tests. No throughput, latency,
or 64 MiB process-memory guarantee follows from these results.

## Documentation Consistency

The second project-schema/OpenAPI/TypeScript generation produced no drift.
Structural comparison confirms every old project definition and ordered operator
alternative is unchanged, apart from the one additive ASOF operator alternative.
The final native wheel was installed under `target/`; no native module was left
in the source package.


The ASOF guide, Python/streaming/project/Rust/API references, example22, and
normative architecture text agree with strict dual-watermark finality,
conservative C−1 progress, idle waiting, typed identity, the flat payload boundary,
resource failure behavior, and source/sink delivery limits. Preparation is
documented per accepted output chunk: each compaction scans retained state,
so total handler work includes repeated preparations. The low-budget duplicate
fallback is an exceptional scan and does not imply a throughput guarantee.

Ordinary at-least-once sinks can replay physical writes; operator-level one-row
per logical left identity does not provide an external exactly-once guarantee.
Studio uses its existing generic import/view/save flow rather than promising
a new ASOF editor.

## Verdict

**Approve — implementation review complete.** All reviewed source blockers,
acceptance-test gaps, final wheel checks, generated-contract checks, and scoped
lint checks are closed. The unchanged dependency-only lint limitation and unrun
full matrices are recorded above.

PR publication and its one nonblocking CI snapshot remain the parent's AC18
delivery steps; attach this review to the final feature commit and the identical
merged base. Required CI, cross-platform, coverage, and performance results must
be green before any separately authorized merge. This review neither authorizes
merging nor claims pending or absent CI is green.
