# Streaming core rearchitecture — closing review

Status: **final specialist review passed; Join performance gates failed and GitHub CI is pending**. This is not a merge-ready claim. The overall P0–P7 architecture and absolute performance plan remains incomplete.

## Review history

The first full read-only review returned **Request Changes** for three production blockers, repository cleanup issues and open performance/CI gates. Subsequent CI restored Linux, Windows, macOS and crate-release checks and exposed additional complexity, runtime and checkpoint failures. Fixes below are anchored to the current dirty worktree; only checks explicitly listed as completed are claimed.

## Production findings and remediation

### Durable ordered sessions mixed timestamp units

`StreamSession::restore_input` interpreted an ordered raw Arrow timestamp as microseconds. `SessionInput::watermark_manifest_state` also failed to persist the generator's actual observed maximum.

Remediation:

- `SourceV4` optionally persists `SessionOrderedMaximumV4 { unit, value }`; runner manifests are unchanged because the field is omitted when absent.
- Ordered restore validates the raw maximum and declared unit and rejects missing or inconsistent state.
- `GeneratedWatermarkState::max_observed_micros` supplies the generator's real microsecond state independently of ordered raw values.
- Regression coverage exercises all four units, positive and negative values, empty batches, non-advancing pushes and repeated checkpoints for ordered and bounded policies. The durable suite passed 23 tests.

### Hot Join bypassed admission and lifecycle constraints

`HotJoinCore::process` skipped admission, retention, state limits and output budgets; `RunCore::evict` skipped hot history.

Remediation:

- Every hot/generic batch now shares one admission pass. Hot execution is admitted only when every row is admitted and retained and the first-batch state charge is within limits; other inputs migrate and use RunCore.
- `hot_batch_charge` uses production `KeyStore` routing and the existing morsel/run/bucket/expiration charge formulas. No test-only key encoder remains in production.
- Hot output splits recursively by the edge row/byte budget. Empty joins use one zero-row record batch.
- Hot history migrates on progress and applies current ingress eviction. Late/null metrics, state-limit errors and next-row IDs follow the shared admission path.
- Sixteen hot tests passed. The zero-match regression passed separately.

### Capability inventory contradicted checkpoint layouts

The public catalog and tests now report Rolling layouts 1–7, ASOF layout 11 and Join layout 12. SQL layouts 3–6 are synchronized in `docs/python-api.md`. Session documentation now explains generated-watermark units and managed-storage checkpoint configuration. The native binding capability suite passed 41 tests.

### Inline source edges panicked under coordinated progress

Runtime initialization selected inline/session source edges and then tried to convert them to physical source edges. Inline execution is now admitted only when direct progress is valid; coordinated progress creates physical edges from the start. Fusion test entrypoints explicitly preserve the direct path. The 179 targeted Python binding regressions passed.

### Checkpoint settlement had two independent defects

- Paused-clock tests resumed Tokio time immediately before releasing the real v4 persistence gate, rather than increasing timeout settings.
- Manual checkpoint settlement no longer requires `current_epoch` to differ from the completed epoch. Durable source acknowledgement still precedes success; later maintenance failures keep their diagnostic phase without turning a successful checkpoint into a failure.

The targeted checkpoint tests passed. The independent native Join ack regression also passed after releasing the completed ack before `close_and_drain`, confirming that the fixture held the native capture and reservation while the close waited for their release.

## Complexity and CI cleanup

- Parameter objects and helper extraction removed the newly introduced Rust `too_many_*` suppressions and Ruff complexity increments. The committed complexity baseline was not updated to hide debt.
- Codacy-annotated functions in `calc-flow-python`, the grouped-SUM example and ASOF admission were reduced to the required limit by the implementation specialist, with focused tests passing.
- Arroyo and historical Join harnesses use explicit evidence errors and smaller controller/audit functions. Subprocess calls retain fixed argv with reviewed `nosec`/`nosemgrep` annotations.
- Unreferenced raw checkpoint logs with trailing whitespace or blank EOF lines were deleted. `git diff --check origin/main` passes.
- `Cargo.lock` retains only the `libz-sys` 1.1.31 → 1.1.33 security update for RUSTSEC-2026-0544, not an unrelated dependency migration.

## Verification completed locally

- Hot Join: 16 tests plus zero-match passed. The zero-match test was rerun independently after review: one test passed and 3,171 were filtered out.
- Durable sessions: 23 tests passed.
- Rust complexity specialist: grouped-SUM 11, PyO3 session 7, PyO3 config 12 and focused ASOF recovery tests passed; full `--tests` Clippy passed.
- Python: capability 41 and the other 179 targeted binding tests passed. Profiling 55 tests plus 2 subtests, P3 45, checkpoint 11 and release 23 passed; numeric/session combined tests passed.
- Full Ruff lint, changed-file Python formatting, Rust formatting, complexity ratchet, spec references, schema drift and full-PR whitespace checks passed.
- Arroyo pure-function fixture, oracle, summary and explicit-error checks passed.

## Final specialist re-review

The original reviewer inspected `6a2abc08` and found no new production blockers. Its initial zero-match fixture finding was withdrawn after checking the actual one-row right input, join time bounds and the recorded passing test. Final verdict: **no new blocking findings in production code or tests**. This source review does not establish performance or CI success.

A subsequent focused review of `0e9ed0b8`, `0e2bf509`, `32fe3c11` and `fd62fc27` also returned **no new blocking findings**. It confirmed the durable source-ack contract, ASOF/Rolling/Join layout alignment, typed Rolling error projection and the diagnostic attribution limits. Code review passes; new GitHub checks and the absolute performance gates remain independent acceptance requirements.

A separate documentation check caught an attempted release-rule exception introduced while addressing Codacy wording. The guidance now says release inspection rejects frozen `src/calc_flow`, fixtures and guidance directories, matching the existing inspector without an exception.

## Current-head performance conclusion

- Sealed Join wheel, revision `6a2abc08`: static Join P50 **619.785 ms** against the 30 ms limit; interval Join P50 **2,058.241 ms** against the 350 ms limit. Both cases passed all 10 complete output oracles. Acquisition took 64.513 seconds and the separate release build took 837.820 seconds. Both absolute gates failed; semantic fixes are retained and no timing-driven samples were added. Evidence: `benchmarks/join-asof/p5-review-fixes-20261122/result.json`.
- Sealed ASOF binary, revision `6a2abc08`: balanced **−17.30% (improved)**, fixed128 **−3.15% (no-confirmed-regression)**, skew **−18.69% (improved)** and restored-skew **−21.39% (improved)**. The fixed 160-observation protocol passed its maintained validators. Acquisition took 1.070 seconds and the separate build took 757.790 seconds. Evidence: `benchmarks/asof-latest-probe/p5-latest-20261121/outcome.json`.

## Join regression diagnosis

One diagnostic observation per scenario used a sealed `benchmark-internals` wheel and the same complete output oracle. These are not new P50 samples.

- Static Join did not execute `hot_execute`; `run_route`, `run_materialize` and `run_probe` took 499.109, 111.669 and 33.798 ms. When the opposite ingress has ended, `retain_threshold` returns `i128::MAX` and every row becomes `retain=false`; requiring every row to be retained incorrectly rejects the hot path even though the current batch can still join.
- Interval Join still executed the native BandJoin. `hot_execute`, `hot_emit` and shared admission took 775.573, 5.592 and 24.510 ms of a 2.908-second diagnostic observation. The remaining uncovered region includes the new production-KeyStore `hot_batch_charge` and other uninstrumented runtime work. No separate charge timer was present, so exact per-function attribution is not claimed.

Evidence: `benchmarks/join-asof/p5-review-fixes-20261122/diagnosis.json`.

## CI-driven follow-up

- The final two Codacy findings were reduced without behavior changes; the Arroyo summary regenerated all eight historical entries identically and the NumPy binding suite passed 37 tests.
- Manifest retention scans now ignore only files that disappear between enumeration and reading, while preserving other errors. The new helper regression and original runner test passed on Linux; this addresses the Windows/Linux `NotFound` race.
- ASOF analysis and its tests now use the native layout 11 contract. Rolling layout 7 is covered by the real `ROLLING_STATE_TABLE_LAYOUT_VERSION` implementation. The related 215 Python tests passed, including the v4 native Join manifest assertions and all nine native-table indicator tests.
- The typed Rolling batch path now projects invalid input data as an operator error with node context instead of a raw `ConfigError`. An ineffective TableOperator alternative was removed rather than retained.
- M5 fault expectations now distinguish `CompletedCommit`, which precedes required durable source acknowledgement and therefore cannot return success, from Retention/Compaction maintenance failures after acknowledgement. All four CompletedCommit modes passed. This updates the outdated test contract rather than weakening the documented acknowledgement guarantee.
- Full all-feature Clippy, Rust formatting, complexity ratchet, full Ruff lint, spec references, schema drift and full-PR whitespace checks passed after these changes.

## Open verification and acceptance

- Required GitHub checks are pending on the repair commits. Earlier failed checks are not treated as current-head results; Codacy must also be scanned again.
- Rolling G1–G3, Join absolute gates, P4 10M, P6 parallel scaling, P3 M3/M4, full ASOF inventory, Arroyo burst alignment and the complete P0–P7 plan remain unmet or unmeasured. No overall completion or merge-ready claim is made.
