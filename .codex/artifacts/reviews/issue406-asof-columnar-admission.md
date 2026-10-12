## PR #411 Review: ASOF right columnar admission

**Branch:** `feature/issue406-asof-columnar-admission` → `main`
**Implementation base:** `8ee5b71b7b862d4e44bd0be390d182cf950d17f7`
**Current base:** `8ce5380bd6b6704473cba4ca1083820dd1059e42`
**Reviewed:** 2026-10-12, initial implementation and the two-file complexity
refactor at `5e08b60425f611f4d755cf62fe36f7f322d1cb0a`, plus the analysis updates.
**Scope:** Source correctness and functional verification. The overall performance
goal remains paused. The initial review preceded PR creation; no new GitHub status
snapshot was taken for this incremental review. Required CI remains a separate
merge gate.

### Summary

Eligible ordered right inputs retain typed time and UInt64 sequence columns with
remapped key IDs instead of allocating canonical row descriptors. The fast path
proves that each touched key follows its actual retained maximum, preserves the
existing fallback and admission transaction, and reconstructs canonical identities
only when a delta journal needs them.

The complexity refactor extracts record preparation, key remapping and the retained
maximum predicate. Differential test helpers extract seed/restore and left probes.
Validation, fallback, cancellation, processing order and resource cleanup remain
equivalent to the previously reviewed implementation.

### Build and Test Results

Existing implementation logs were inspected without repeating native builds or
passing tests. The first three commands used the main checkout's
`target/issue406-join-hotpaths-build` as `CARGO_TARGET_DIR`.

```bash
cargo test -p calc-flow --lib operator::asof --all-features -- --test-threads=1
cargo test -p calc-flow --lib operator::asof::admission::identity_tests::columnar::direct --all-features -- --test-threads=1
cargo clippy -p calc-flow --lib --tests --all-features -- -D warnings
cargo fmt --all --check
```

- Rust: **Passed** within the reviewed scope. `asof-final.log` records 341 passed,
  zero failed, in 25.86 seconds. The final projection/COW checks in
  `direct-final.log` record eight passed, zero failed, in 8.56 seconds.
- Clippy: **Passed**, with completion in 49.78 seconds in `clippy-final.log`.
  Formatting completed with exit zero; `format.log` contains no diagnostics.
- After the complexity refactor, `codacy-direct.log` records eight passed, zero
  failed, in 8.63 seconds after a 1 minute 12 second build. `codacy-clippy.log`
  records successful completion in 26.64 seconds. The implementer also confirmed
  `cargo fmt --all --check` completed with exit zero. These checks used the same
  focused test and Clippy commands shown above; the reviewer did not repeat them.
- Python, Studio backend and Studio frontend: **Not touched**; suites not run.
- New unresolved failures: **None**. The earlier parallel test attempt exhausted
  existing native-worker capacity and was not counted as passing. Intermediate
  new-test failures were traced to repeated checkpoint epochs and a snapshot
  retained by the test; the corrected tests pass.
- Regressions: None identified in the reviewed source and focused verification.
  Full regression and coverage suites were not run locally; this review does not
  establish the combined Rust 90% coverage gate.
- Separate scheduled performance failure: the morning 10M streaming `group_by`
  regression remains unresolved. Its candidate predates this PR, and the local
  diagnostic does not replace the original CI verdict.

Logs are under the main checkout's
`target/issue406-asof-columnar-performance/`. The implementer recorded the initial
RED result: 4,097 canonical right-row descriptors where the new assertion requires
zero.

The reviewer independently confirmed these checks were clean after the unrelated
Cargo lockfile ordering change was restored:

```bash
git diff --exit-code -- Cargo.lock schemas/project-v3.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts
git diff --check
```

### Blocking Issues

None remain in the source and functional scope.

The review finding originally in `admission/right.rs::plan` was addressed:
dictionary-key interning in `RecordColumns::prepare` now calls
`check_input_cancellation` periodically, preserving the previous cancellation
opportunity for high-cardinality inputs.

### Style Issues

None remain. The analysis artifact's table alignment was corrected, and the
changed Markdown has final newlines and no trailing whitespace.
The two Codacy complexity findings were addressed with focused helper extraction;
this source review does not claim a fresh remote Codacy result.

### Test Coverage

- Differential checks compare fast and forced-legacy state, row references,
  counters, inventory, output, and checkpoint metadata and segment bytes.
- Coverage includes independent record dictionaries, empty batches, nonzero slice
  offsets, UInt64 extremes and ties, restored maxima, ordered and general
  identity-only history, and unordered, wide-key and mixed-late fallbacks.
- Delta recovery explicitly requires three journal changes and two frames across
  epochs 1 and 2, then verifies exact continuation output after restore.
- Held bucket references exercise copy-on-write. Nullable payload cases cover
  left-only, right-only and empty output projections with whole and sliced inputs;
  caller data and Arrow owner counts remain unchanged.
- Duplicate priority, explicit workspace refusal and cancellation retain committed
  counts and inventory and refund temporary credit. The generated all-feature
  unit harness remains at 2,048 registrations.

### Documentation Consistency

The implementation satisfies the work item's specification. State counts use
per-key admission totals even when generic descriptors are empty; typed sequences
retain the existing little-endian storage; delta edits reconstruct canonical
identities from the same compact payload mapping. Existing owner accounting,
capacity projection, checkpoint index length and infallible commit boundaries
remain in place.

No public API, checkpoint format, generated contract, dependency or default memory
limit changed. The existing runtime-envelope documentation remains consistent.
The analysis artifact records the paused performance scope and makes no timing
claim for this candidate. Historical timing values were not remeasured as part
of this review.

The morning benchmark investigation accurately preserves the separate evidence:
809 CI result rows contain one regression, with a +11.663% combined p50 change and
both round interval lower bounds above +11%. The reviewer independently checked
the original result, the two wheel and native seals, the fixed local plan and
driver/harness provenance, all local measured-sample oracles, and the 119.327-second
diagnostic result. Its three comparisons and 600-second budget stayed fixed; the
467 recorded resource samples show no observed native build overlap.

The WSL A/B result of +0.885% is a non-reproduction on another environment, not
evidence that the CI failure was invalid or that performance is equivalent.
The documented single recovery run remains follow-up evidence to evaluate;
neither its completion nor a resolved root cause is claimed here. The rebase's
release version changes do not relabel either measured product artifact.

### Verdict

**Approve** for the reviewed source and functional change. Required CI must still
pass before merge. This verdict does not resume the paused performance goal or
claim performance acceptance; the scheduled benchmark failure investigation is a
separate task.
