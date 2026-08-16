# Public A6 Continuous Runtime Implementation Evidence

## Scope and authority

This artifact records the completed Public A6 delivery on
`main@5deb3b4daa2b647353595feb556413adf52b89cd`. The controlling design chain is:

1. `../specs/a6-public-continuous-runtime.md`;
2. `../api-notes/a6-public-continuous-runtime.md`;
3. `../critiques/a6-public-continuous-runtime.md`;
4. the accepted D1 delivery deviation recorded below.

Public A6 exposes the source-driven continuous runtime through the Rust crate
root, PyO3, and Python; gives one owning job responsibility for lifecycle and
managed checkpoint recovery; removes the legacy continuous runner and public
checkpoint-document surfaces; and preserves Studio's private `/api/v2`
checkpoint document wiring. Production connectors, project v3, Studio job
routes/UI, savepoints, rescaling, schema migration, and the `3.0.0` release
remain M6 or M7 work.

## Frozen design-package integrity

The API note still hashes to the critique's reviewed SHA-256
`faae44001e1d85a579fa85ae6c16c15c353d770d0bb0238c60e1e611c779d40e`.
The current specification hashes to
`43a8465e5d512afad7b23156eba2566ba9faa173a17b761fad8685b7c1a45a73`,
while the critique records the pre-closure reviewed hash
`d2730be9dd135ad95e57db052b87e515ec722d468b4c4bc9b34dab26b9735948`.
The exact diff removes the nine-line stale `Open Questions` request for the API
note and independent critique after those artifacts were supplied, replacing
it with `None.`; it changes no decision, signature, acceptance criterion, or
runtime requirement. This completion artifact does not rewrite the frozen
specification, API note, or critique.

## Delivery chain

- [PR #113](https://github.com/wegamekinglc/calc-flow/pull/113) froze the A6
  specification, API note, and zero-blocker six-axis critique for issue #94.
- [PR #119](https://github.com/wegamekinglc/calc-flow/pull/119) delivered the
  Rust public facade tracked by issue #102.
- [PR #120](https://github.com/wegamekinglc/calc-flow/pull/120) delivered the
  PyO3 owning-job bridge tracked by issue #103.
- [PR #121](https://github.com/wegamekinglc/calc-flow/pull/121) delivered the
  Python async lifecycle and typing tracked by issue #104.
- [PR #122](https://github.com/wegamekinglc/calc-flow/pull/122) removed the
  legacy Rust, PyO3, and Python continuous surfaces tracked by issue #105.
- [PR #123](https://github.com/wegamekinglc/calc-flow/pull/123),
  [PR #124](https://github.com/wegamekinglc/calc-flow/pull/124),
  [PR #125](https://github.com/wegamekinglc/calc-flow/pull/125), and
  [PR #127](https://github.com/wegamekinglc/calc-flow/pull/127) completed and
  hardened the cross-surface, fault-catalog, and soak evidence tracked by issue
  #106.
- [PR #129](https://github.com/wegamekinglc/calc-flow/pull/129) integrated the
  current-main delta, closed all actionable review findings, and completed the
  final gate tracked by issue #107.

The final integration candidate was
`b05ed44bcf2686298241d313e903ec05cf5fdcb7`, with tree
`599968d11dbe5b94c02ce20c4faa2ae415a31907`. It merged as
`5deb3b4daa2b647353595feb556413adf52b89cd`, with tree
`31fe6b301404522e134ed2df4cac981b5303a446`.

## Acceptance mapping

### Ownership, lifecycle, and managed root

- `crates/calc-flow/tests/continuous_public.rs` proves crate-root construction,
  consuming `start(self)`, the non-cloneable owning job, one managed root, and
  the public status/checkpoint/shutdown/cancel/wait contract.
- Inline streaming runtime tests cover pure construction, launch-gate
  cancellation, root and lineage leases, lexical/symlink/cross-process root
  conflicts, failed-start cleanup, drop/reaper convergence, and terminal zero
  ownership.
- `python/tests/test_continuous_runtime.py` and
  `python/tests/test_native_stub.py` prove the matching async-only Python
  lifecycle, active-loop blocking rejection, signatures, cancellation, GC,
  and native/Python ownership convergence.

### Recovery and checkpoint completion

- Empty-lineage, `R + 1`, sparse selection, corrupt-candidate fail-closed,
  terminal resume, and scoped per-sink recovery tests exercise the managed v3
  manifest path.
- Manual, periodic, and terminal checkpoint tests cover bounded FIFO,
  single-flight ordering, non-coalescing epochs, observer drop, and completion
  only after manifest installation, parent sync, and all sink acknowledgements.
- Real directory-sync failure and process-restart tests cover both possible
  installed-unknown outcomes without claiming absence, while preserving
  recovery-required intent and stable public error coordinates.
- `crates/calc-flow/tests/a6_cross_surface.rs` and
  `python/tests/a6_cross_surface_worker.py` pass bidirectional Rust-to-Python
  and Python-to-Rust restart compatibility using one managed namespace.

### Capability, identity, and safe projection

- Whole-job preflight proves exact source coverage, stable source/sink IDs,
  descriptor and cursor identity, replay positioning, watermark capability,
  delivery capability, byte/row bounds, reachable-path guarantee derivation,
  and participant coverage before connector lifecycle work.
- Foreign cursors, descriptor drift, lossy or non-positioning sources,
  volatile operators, invalid participant IDs, and insufficient sink retention
  fail closed before governed effects.
- Public lifecycle failures project only as
  `CalcFlowError::Streaming(StreamingError)` and the corresponding Python
  exception. Field censuses and canaries cover display/debug/source chains,
  status, outcome, logs, metrics, diagnostics, serialization, Python cause and
  context, manifest recovery fields, and two-sink isolation.

### Cutover, fault catalog, and repository gates

- Rust exports, PyO3 registration, Python exports and stubs, examples,
  benchmarks, Studio package startup, v2 checkpoint routes, documentation,
  migration guidance, and release smoke agree on the sole A6 lifecycle. Removed
  public names have compile/import failure coverage and no aliases.
- `crates/calc-flow/src/runtime/streaming/soak.rs` contains and executes all 48
  unique `m5_fault/<point>/<mode>` cases from the specification. The catalog
  covers 12 checkpoint boundaries across I/O, panic, cancellation, and process
  restart, including absent, durable, and both installed-unknown restart
  outcomes.
- The final candidate passed 17 of 17 exact-head GitHub check runs, all four
  Coveralls statuses, Codacy with zero issues/annotations, and review with zero
  unresolved threads. The current-main merge tree was conflict-free.

## Exact-head soak evidence

The replacement final-head soak ran the crate-root public runner for 1,200
measured seconds and produced the durable bundle in
[PR #129 comment #5304132298](https://github.com/wegamekinglc/calc-flow/pull/129#issuecomment-5304132298):

- schema `calc-flow.m5-checkpoint-soak-evidence.v2`;
- 39,967 bytes;
- SHA-256
  `cb4efe30e2aac8fa3be907de18f8fc6897d28af0cef598e8429e766c7d1e2c73`;
- 120 ordered ten-second samples across three child process generations and two
  successful OS-process restarts;
- restored epochs `null`, `78`, and `155`, followed by terminal epochs `78`,
  `155`, and `234`;
- 28,357 accepted source records and 56,714 two-sink output records, with zero
  missing or duplicate records;
- bounded task, queue, state, manifest, RSS, and checkpoint resources;
- zero terminal tasks, charged edges, live/reaper registries, or temporary
  artifacts.

The bundle was independently downloaded, byte-hashed, schema-validated, bound
to the exact candidate commit/tree, and recomputed before merge.

## Accepted D1 delivery deviation

The A6 design required the public cutover to reach `main` through one atomic
integration. The actual history merged Rust facade PR #119 before the remaining
PyO3, Python, cutover, and integration stack. Cheng Li explicitly accepted that
non-atomic history in
[PR #119 comment #5305964523](https://github.com/wegamekinglc/calc-flow/pull/119#issuecomment-5305964523).

This acceptance is limited to repository history. It does not waive runtime
correctness, review, CI, Codacy, cross-surface, fault-catalog, soak, or
current-main merge requirements; all of those gates subsequently passed on the
final PR #129 candidate.

## Completion status

Public A6 is complete. Issues #102 through #107 may close against this evidence
and the merged delivery chain. M6 starts from the existing A6 public lifecycle
and owns connector implementations, project v3, Python connector/capability
integration, and Studio v3 job surfaces. M7 owns final performance, security,
packaging, documentation, version, and release gates.
