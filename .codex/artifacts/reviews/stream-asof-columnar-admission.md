# Branch Review: ASOF columnar admission

**Author:** Cheng Li | **Branch:** `feature/stream-asof-columnar-admission-main`
| **Base:** `764843e634ae1a1da7a5b010095c3017349a25f4` | **Files:** 7

Source commit: `e3e3e11f7982ae7144d2b760b4b77bdddc20c060`.
Source tree: `eae2b2f4b5e97cb13a458dd7c5cae681ba27358b`.
Reviewed documentation head: `085b706141a1e198414e178d8665a55fec3f76d5`.
Reviewed documentation tree: `40b0848f1bbd842b112d295732dd5f29a8e772a6`.
The latter changes only analysis table alignment and inclusion of the existing
approved critique. All tested production code, tests, and changelog remain
byte-identical to the source seal.
This is a local branch source review; no PR check runs or prior GitHub reviews
were available for this A4 source. Earlier A3, gather, and Join approvals are
the parent boundary and are not reopened here.

## Summary

Eligible left admission constructs its chunk directly from validated admission
ranges, removing the borrowed identity/position vector and repeated order and
position validation. The fast path reuses the existing duplicate probe rather
than adding another admission scan. It retains the first-occurrence dictionary,
owned time and native-width integer sequence columns, funding, and private
checkpoint layout.

The review follows `code-style`, `AGENTS.md`, [FR16](../specs/stream-join-asof-acceleration.md),
the [approved critique](../critiques/stream-asof-columnar-admission.md), and the
normative introduction and ASOF guide. Changed Rust files and their complete
context, new tests, analysis, and changelog were read. The final source was also
checked in a clean detached review worktree at the exact commit above.

## Build and Test Results

- Rust: **Passed**, for the scoped implementation checks below. No full
  workspace regression, all-feature lint, or coverage was run locally.
- Python: **Not touched**; package tests and native imports were not run.
- Studio backend: **Not touched**; tests were not run.
- Studio frontend: **Not touched**; build and tests were not run.
- New failures: none unresolved in the final scoped Rust and documentation
  checks. The initial Markdown alignment issue was corrected at the reviewed
  documentation head.
- Regressions: none observed in the selected checks; this is not a full CI or
  performance result.

The implementer observed these commands finish successfully. GREEN stdout was
not separately archived; the counts below explicitly transcribe those tool
results rather than claiming a raw GREEN log. All Cargo commands used
`CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target`
and `CARGO_BUILD_JOBS=2`.

```bash
cargo test --locked -p calc-flow --lib \
  operator::asof::admission::identity_tests::columnar:: \
  -- --test-threads=1 --nocapture
cargo test --locked -p calc-flow --lib \
  operator::asof::admission::identity_tests:: \
  --skip operator::asof::admission::identity_tests::columnar:: \
  --skip operator::asof::admission::identity_tests::parallel_cpu:: \
  --test-threads=1
cargo test --locked -p calc-flow \
  --test stream_asof_join_properties --test stream_asof_join_boundaries \
  -- --test-threads=1
cargo test --locked -p calc-flow --lib \
  operator::asof::admission::identity_tests::columnar::journal_segments_and_restored_continuation_match_legacy \
  -- --test-threads=1 --exact
cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
cargo fmt --all --check
```

Results: 15 columnar unit tests, 23 directly affected admission/CPU tests, eight
properties, and three boundaries passed: 49 unique tests. After extending the
wire fixture, its single focused recheck passed without repeating the unchanged
groups. Scoped Clippy, format, contract drift, and whitespace checks passed.

The reviewer independently checked exact commit/tree identity, a clean detached
worktree, whitespace, generated contracts against the sealed parent, Markdown
structure, and the test-first evidence. No Cargo command, native import, release
build, or performance measurement was repeated during this review.

The actual RED output in root `target/issue363-a4/red.log` records successful
compilation followed by the two expected borrowed-row failures, 1,024 and
64,000 against zero. The interrupted initial GREEN build exited 130 before
tests; later fixture compile/style corrections are not additional behavioral
RED evidence. The Lizard body comparison in root
`target/issue363-a4/complexity.json` covers 74 added/modified functions, with
maximum complexity eight. Its sole whole-file warning is unchanged legacy
`ChunkData::prepare`, complexity eleven; no suppression or threshold changed.

## Blocking Issues

None. The source correctness, ownership, and documentation findings are
addressed at the reviewed head.

## Style Issues

None unresolved. The original analysis allocation table had misaligned pipes.
The documentation correction pads all cells and separator dashes to shared
column widths. The reviewer verified identical pipe positions for its header,
separator, and both data rows. No Cargo checks were repeated for this correction.

## Test Coverage

- `inspect_duplicate_identities` preserves the legacy duplicate decision:
  sorted input is nondecreasing; zero duplicates makes that order strict, and
  `skip_resident` proves the first identity is beyond the actual live left
  maximum. `LeftState` maintains that maximum across chunk insertion, prefix
  removal, and restore. The proof compares canonical encoding bytes, not key
  handles, accepted-row counters, or physical arrival order.
- Eligibility also requires nonempty, fully accepted left input, one integer
  sequence, one integer or UTF-8 key, and microsecond event time. Late filtering
  cannot create eligibility after compaction. Reversal, overlap, duplicate,
  composite and string-sequence fixtures retain fallback or the original
  refusal; null validation remains before this private proof.
- The differential matrix covers 0/1/1,024 rows, 1/3/5/64 keys, short/long and
  empty/non-ASCII UTF-8, signed/unsigned integer widths and extrema, equal-time
  ties, and record cuts. It compares identities, references, first-occurrence
  dictionaries, all six actual capacities, owner inventory, payloads, and raw
  integer storage. Inline `AdmissionRef.key_index` remains zero; the chunk's
  IDs still come from `intern_chunk_key` with its original capacity growth.
- Full/projected/zero-output-column payloads and larger sliced backing use the
  unchanged payload decision. Fast chunks own time values and the existing
  integer prefix copy; no direct external backing retention or new type support
  is added. The original unknown-external-owner safeguard remains intact.
- Same-epoch metadata and complete segment equality cover admission, repeated
  capture, and restored continuation. The strengthened fixture commits a real
  512-row prefix, checks output sequence `0..512` without sorting, and compares
  live/restored state, metadata, segments, and hashes afterward.
- Tight-budget differential checks cover immediately below/at identity,
  payload, worker descriptor, and aggregate install thresholds, plus a 32 MiB
  pool. They compare failure category, status, peak reservation, and complete
  refund. Existing duplicate/workspace precedence tests also passed.
- Entry and final-commit retirement tests cover dropped waits, cancellation,
  reset/restore, operator drop, retained ownership/credit, and managed drain.
  The additional columnar abandoned-worker case uses the same owned executor.
  No registration, reservation acquisition, install, or retirement boundary is
  bypassed by the constructor switch.
- Actual constructor counters travel with chunks and include the detached
  worker thread, alongside caller future-poll counters. The 64,000-row case
  removes one allocation and 1,024,000 allocated bytes; retained constructor
  bytes are unchanged. These are bounded-fixture allocation observations,
  not whole-job RSS or timing gains.

## Documentation Consistency

The analysis and changelog describe the bounded construction change and retain
the existing public API, supported types, backing policy, limits, fingerprints,
and checkpoint contracts. The ASOF guide's ownership, typed sequence, output,
and recovery descriptions remain consistent; no unrelated public-guide or
generated-contract change is required. Dependency manifests, supply-chain
configuration, and agent definitions are unchanged.

The documentation correction includes the already-approved critique unchanged,
with Git blob `097047951e9241594f57fc75a84ea52ffb7db2dc`. The analysis relative
link now resolves in the reviewed commit. The reviewer neither edits nor stages
the critique or author analysis.

Performance and peak process RSS have not been measured for this source. The
FR16 planning target of 100 ms per million rows remains unverified. Source
approval does not imply performance acceptance, green remote checks, or merge
readiness.

## Verdict

**Approve**, for source at the reviewed documentation head, with production
code bound to the tested source seal. No source, style, or documentation finding
remains unresolved. Performance measurement, required CI, and remote review
resolution remain separate publication and merge gates; this verdict makes no
claim that those gates have passed.
