# Streaming core rearchitecture — closing review

Status: **pending final verification**. This is not an approval or a merge-ready claim. The overall P0–P7 architecture and absolute performance plan remains incomplete.

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

- Hot Join: 16 tests plus zero-match passed.
- Durable sessions: 23 tests passed.
- Rust complexity specialist: grouped-SUM 11, PyO3 session 7, PyO3 config 12 and focused ASOF recovery tests passed; scoped Clippy passed.
- Python: profiling 55 tests plus 2 subtests, P3 45, checkpoint 11 and release 23 passed; numeric/session combined tests passed.
- Ruff, formatting, complexity ratchet, spec references, schema drift and full-PR whitespace checks passed.
- Arroyo pure-function fixture, oracle, summary and explicit-error checks passed.

## Open verification and acceptance

- Full `--tests` Clippy and checkpoint/native-Join regressions passed. Capability 41 tests and the other 179 targeted Python tests passed. Final specialist re-review is pending.
- Required GitHub checks are not green on the current dirty head. Earlier failed checks are not treated as current-head results.
- Hot Join and ASOF performance must be re-measured after semantic changes. Old sealed values are not current-head evidence.
- Rolling G1–G3, P4 10M, P6 parallel scaling, P3 M3/M4, full ASOF inventory, Arroyo burst alignment and the complete P0–P7 plan remain unmet or unmeasured.
