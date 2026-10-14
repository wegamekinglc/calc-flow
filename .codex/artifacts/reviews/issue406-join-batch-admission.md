## Branch Review: issue 406 Join batch admission

**Branch:** `feature/issue406-join-batch-admission`

**Reviewed head:** `327a6bbfaa162bddddc97f08ce312b99e393bd6a`

**Base:** `bb142320062995aeb318ad406c901288537fd766`

**Scope:** 37 changed files, including 32 Rust files. This is the incoming batch
ownership/selection increment within phase 1.1, following test/spec commit
`53be8035`. It is not a fresh review of the prior delivery branch or the separate
release-profile increment.

### Summary

Incoming Join admission now uses record segments, normalized event-time columns,
packed retention bits and lazy sparse selections instead of owning row records.
Borrowed consumers retain distinct physical, probe and copied-payload positions.
Larger metadata is reserved before classification; background transfers detach
required record parents or decline to serial execution. Existing retained-row,
checkpoint, fee, output-order and commit contracts remain in place.

The final review rechecked the advisory's two pending-record defects and their
corrections, sparse promotion rollback, consumer migration and the mutable outer
batch ownership refactor. No unresolved correctness or ownership blocker was
found at the reviewed head. The withdrawn legacy state-funding concern remains
an explicit caveat, not a newly introduced regression.

### Build and Test Results

Owner execution evidence is under
`target/issue406-join-batch-admission-20261014/`. The reviewer inspected results
and source without repeating passing checks.

- Rust focused tests: **Passed**. `focused-final.log` records 38 selected tests,
  zero failures and 2,011 filtered out, after 4m18s compilation and 2.22s
  execution. `focused-final.json` records exit zero and 262.014580329s process
  elapsed, together with the complete command and exact 38 filters.
- Scoped core Clippy: **Passed**. `cargo clippy -p calc-flow --lib --tests --
  -D warnings` completes in 1m53s. This uses default features; it does not
  establish workspace/all-feature lint coverage.
- Formatting, generated project-schema no-drift and whitespace: owner records
  **Passed**. No project schema, dependency, toolchain, public API,
  fingerprint or checkpoint-layout change belongs to this increment.
- Python/current-native integration, release/package builds, full regression
  CI, cross-platform checks and the combined 90% Rust line gate:
  **Not run/unverified** at this head.
- New unresolved failures: **None** in the final scoped evidence. Earlier REDs,
  fixture mistakes, intermediate compile failures and Clippy failures remain
  distinguished in the analysis rather than being reported as passing runs.

The final test command is a core lib invocation with the explicit OR list of 38
fully qualified filters, followed by `--exact --nocapture --test-threads=1`.
Its authoritative full argv is in `focused-final.json`; it is not a full Join
or workspace test run. `build-environment.json` records the shared target
`target/issue406-join-hotpaths-build`, two build jobs and no Rust flags, wrappers,
incremental or dev/test profile overrides. The final ownership/future refactor
changed the previously tested paths, so this focused rerun was justified.

The reviewer independently derived the changed Rust path set, matched all 32
working-tree and committed files to `complexity-final.json`, and recomputed its
sorted `filename + ASCII space + sha256 + LF` aggregate:

```text
dda80c61ca8560652d13399ea614ffd57cea068f259bf439be5a806e98f2b16b
```

The manifest records 283 changed/new functions, maximum CCN 8 and no parser
omissions. This source identity supports the review handoff, not a performance
or coverage verdict.

### Blocking Issues

None remain for this increment.

**Resolved — `AdmittedBatch::segment_index`:** lookup now searches only the
initialized record prefix. `begin_record` installs each segment with its current
admitted range, so not-yet-classified suffix records cannot break the binary
search ordering. The authentic mixed-record RED observed the prior panic;
`test_batch_admission_mixed_records_copy_and_borrow` now exercises an actual
successful first copy followed by a borrowed record, preserving IDs and fees.

**Resolved — `AdmittedBatch::release_table` / `complete_record`:** a pending
record prevents table release, and completing the final record triggers another
release check even when it admitted no rows. Independent controls cover manual
replacement followed by a required borrowed record and the omitted last-record
owner disappearing after caller release. The nested preflight correction also
removes production one-row RecordBatch views while preserving dictionary output
and budget behavior.

### Style Issues

None outstanding. The ownership/future-size corrections use fixed inline slots,
prepaid optional spill and borrowed outer boundaries. They add no production
future boxing, lint allowance, lint expectation or workspace-lint weakening.
Expected errors remain Results; private capacity/owner assertions express
invariants established by checked preflight and construction. The owning
AdmittedRow fixture and row-view adapter are test-only.

### Test Coverage

The structural controls combine actual allocation observation, independent
parent-owner checks, segment counts, sparse-capacity/write counters and nonempty
output parity. Dense input has zero sparse allocation/writes. The inline bounds
cover 64 physical rows/four records, with separate 65-row/five-record spill
controls and exact-credit versus one-byte-short refusal. Retention capacities
cover inline and multiword storage. Sparse promotion is stopped after genuine
partial-prefix writes and cooperative yielding; cancellation or abandonment
rolls back its uninstalled prefix and releases reservation on batch drop.

Semantic controls cover sparse sliced Int8 SQL with an actual execution witness,
complete Arrow output, physical ID gaps and full fees; all timestamp units;
null/late/retention boundaries; word/record boundaries; earlier conversion,
metric and ID errors; and state/match-limit atomicity. Native framing, composite
keys and both input directions retain their reference controls. Existing legacy
Int8/date codec behavior is preserved; this is not a claim that every historical
codec limitation has been fixed.

Real mixed-parent Count/Fill controls stop native workers while required input
and old state remain alive. They prove dropped/replaced parents disappear after
caller release, no output/state commit occurs, reset waits for cleanup, all
owners refund, and a fresh operation can proceed. Selected-copy input/result
controls likewise cover cancellation, abandonment, actual admission refusal and
retry, source immutability, omitted/pending-parent release, positive reservation
and pending drain until worker release, then pool zero. These helpers are
actually called by the passing native-lookup and materialization registrations;
they are not inferred from compilation alone.

The reused checkpoint controls perform post-restore batches with dropped
physical ID gaps through authentic layout 1 and layout 2 migration and strict
layout 3 recovery. They compare output, status, sequence and full V1 charges.
Serial/nonserial SQL fallback, paid-scratch refusal/retry and zero-headroom
acceptance are exercised after the mutable batch ownership refactor.

Source review confirms the following boundaries:

- Dense ordinal, original record position/global fee index, durable ID and
  replacement payload position are derived separately. Native key lookup and
  matched-pair routing use admitted ordinals; IDs and fee selection use original
  positions. Replacing a parent only changes payload addressing.
- Sparse promotion uses the prepaid worst-case capacity and an owned rollback
  guard. Packed retention and spill/control capacities use checked arithmetic;
  there is no unfunded heap fallback for new admission metadata. Mandatory
  workspace failure precedes classification and state/match-limit accounting.
- Borrowed row access returns references/scalars without allocating or cloning
  payload parents. SQL probe construction selects per segment, while nested
  output costing measures Arrow slices directly. Persistent StoredRow and
  checkpoint-row processing remain their existing separate representations.
- Native/SQL ownership transfer detaches selected record parents before capture.
  Refusal takes the serial path; successful recovery reinstalls the returned
  batch before continued matching. Cancellation or future Drop leaves taken
  owners with the existing awaited cleanup mechanism rather than fabricating
  an empty successful result or committing staged state.
- Selected-copy workers hold selected columns plus funded primitive selection
  metadata, not the original whole TableBatch. Owned output capture requires a
  detached owner; borrowed-only payloads select serial gather. Enlarged native
  and copy worker controls remain accounted through their actual owned types.

### Documentation Consistency

The specification, critique, implementation analysis and normative guides agree
on inline bounds, preflight error priority, detach-or-serial behavior and unchanged
logical schemas/fees. The analysis preserves expected RED evidence, labels invalid
Float64/schema/nullability/manual-journal fixtures as fixture failures, and records
subsequent corrections accurately. The final test argv is available in the named
JSON evidence rather than implied to be a broad suite.

The legacy funding caveat is explicit: base admission already allocated an
unreserved per-record Arc<RecordBatch> and column vector, then shared it with
retained rows. `prepare_retained_parents` moves that allocation later and skips
existing owners/records without retained rows. It does not introduce a funded
Rowed state promise. Detach reservations cover worker ownership; independently
funded Shared/restored selected payloads retain their lifetime funding. Requiring
a new state-owned reservation here would change the preserved state contract.

Offset-derived/vectorized charge sums, persistent key IDs, retained state chunks
and journals remain unfinished. No timing, whole-phase allocation/RSS, release,
current-native Python, CI or complete phase 1.1 claim follows from the structural
controls. All eight original performance targets and the full issue remain active;
stopped acquisitions were not resumed.

### Verdict

**Approve — final local specialist source review of
`327a6bbfaa162bddddc97f08ce312b99e393bd6a`.**

This approves this committed incoming-batch increment and its scoped evidence,
not the entire prior delivery tree, full phase 1.1, performance acceptance,
required CI/cross-platform checks, combined 90% coverage, release/new-native
Python integration, merge readiness or issue 406 completion. No tests, builds,
measurements, CI query, remote mutation, commit or product/test edits were
performed by the reviewer. Only this report was written.
