## Branch Review: issue 406 signed/date key codec prerequisite

**Branch:** `feature/issue406-signed-date-key-codec`

**Reviewed head:** `c27db37e5ec5389143e875c40e746d7ea1d719e5`

**Full-delta base:** `327a6bbfaa162bddddc97f08ce312b99e393bd6a`

**Scope:** four changed files, including two Rust files. Review includes the
initial test commit `e549f5a4` and the correction, not only the last commit's diff.
The approved batch-admission implementation is not reopened by this review.

### Summary

Join already accepted Int8, Date32 and Date64 keys, but retention failed on
negative Int8 conversion and date-array downcasts. The correction encodes Int8
as its signed little-endian byte and reads dates through their concrete Arrow
primitive types. It preserves existing V1 framing, fees and previously successful
encodings without changing native eligibility, schemas or checkpoint versions.

Replacing the earlier test's accidental-error expectation is an intentional
phase 1.2 prerequisite. The reviewed change makes accepted keys usable for
retention and recovery; it does not claim completion of the dictionary work.

### Build and Test Results

Owner evidence is under `target/issue406-signed-date-key-codec-20261015/`.
The reviewer inspected source and results without rerunning passing checks.

- Focused behavior verification: **Passed as a union of seven controls**.
  `green.log` initially records six passes and one fixture failure after 2m34s
  compilation and 0.34s execution. `corrected-green.log` records the changed
  stateful control passing for all three key types after 1m18s compilation and
  0.05s execution, 78.675923454s subprocess elapsed. The six unchanged passes
  were retained rather than repeated; the initial seven-test command remains
  classified as failed overall.
- Scoped core Clippy: **Passed**. The final command is
  `cargo clippy -p calc-flow --lib --tests -- -D warnings`, with default features.
  `clippy-final.json` records exit zero and 177.778903969s subprocess elapsed.
  This is not workspace/all-feature lint verification.
- Formatting, schema no-drift and whitespace: owner records **Passed**. The
  reviewer independently matched all four changed files to committed/disk bytes,
  checked final newlines/whitespace and confirmed Cargo.lock and project schema
  equal the base.
- Python/current-native integration, native package/release builds, full
  regression CI, cross-platform checks and combined 90% Rust coverage:
  **Not run/unverified** at this head.
- New unresolved failures: **None** within this scoped review.

`red.log` preserves four genuine production failures: negative Int8 does not fit
u8, and both date types fail the old primitive downcast. The stateful control
also reaches the negative Int8 defect. Compilation took 6m58s and execution
0.01s; recorded command elapsed is 418.263561180s. These were executed behavior
failures, not fixture compilation errors.

The initial GREEN stateful failure is accurately retained: the fixture captured
V1, cleared its test journal, then improperly switched the same operator to V2
without validated migration, producing inconsistent history/metrics. The corrected
fixture restores that authentic V1 snapshot into another operator, prepares its
actual migration and captures V2 before testing both restores. Production codec
source did not change for that fixture correction.

Initial Clippy found only a test helper's needless pass-by-value. Its final
`&dyn Array` signature and literal call sites preserve the slice/assertion body;
final Clippy compiles that test-only adjustment. Unchanged behavior controls were
not repeated solely for the borrow correction. The production source hash in both
GREEN metadata records equals the final committed production file hash.

The exact selected test argv, source snapshots and build environment are recorded
in `green.json` and `corrected-green.json`. The seven-test invocation uses explicit
fully qualified filters followed by `--exact --nocapture --test-threads=1`; it is
not a full Join suite. No build/performance result is inferred from its duration.

The reviewer independently checked the changed Rust path set and both committed
file hashes against `complexity-final.json`, then recomputed the sorted
`path + ASCII space + hex SHA-256 + LF` aggregate:

```text
ae3fad337d1ef96fdc8d55ea0164a3f009ee9a3638cd509dffb290bbfaa1eb26
```

Its 12 changed/new functions have maximum CCN 8 and no explicitly mapped
function omissions. The source seal identifies this local review scope.

### Blocking Issues

None found.

### Style Issues

None outstanding. Small signed, unsigned and string helpers keep dispatch within
CCN 8 and retain explicit Result errors. There is no dependency, unsafe exception,
lint suppression, production boxing or workspace-rule change. Caller arrays and
schemas remain read-only.

### Test Coverage

Literal complete V1 blocks independently check sliced Int8/Date32/Date64 arrays
at minimum, negative, zero and maximum values. They bind tags 2/12/13, empty
timezone framing, little-endian u32 lengths and exact signed value bytes. Slice
sentinels ensure the codec reads selected offsets rather than unsliced storage.

The stateful control retains the pre-existing non-retained acceptance case, then
requires successful retention and exact V1 fees. It generates authentic layout-1
state, restores/prepares layout-2 migration, restores each snapshot and checks
continued SQL matching with complete output/schema equality, sequence, retained
bytes and emitted counters. Owned cleanup is awaited after dropping operators.
Existing all-native-type SQL/order/frame comparisons, frozen V1 captures and
encode-once charge reuse also pass.

Source review confirms nonnegative Int8 bytes remain identical to the old u8
conversion. Date32/Date64 use Date32Type/Date64Type with the existing i32/i64
little-endian helper; their day/millisecond values undergo no event-time
normalization. Other signed/unsigned, Boolean, string and timestamp encodings
retain their bytes, and unsupported encoding errors retain their prior variant
and message. The type-tag table, append_key_block framing and logical charge
calculation are unchanged. No native eligibility route is added for these types.

Previously valid snapshots are covered by unchanged frozen fixtures. This review
makes no claim that formerly failing keys produced valid historical retained
snapshots or that every unrelated codec behavior was exhaustively tested.

### Documentation Consistency

The October 15 changelog entry describes the corrected retention/recovery
behavior and unchanged tags, fees, layouts, schemas and versions. The analysis
explicitly explains the removed accidental-error expectation, distinguishes the
valid RED from the invalid snapshot fixture, reports the seven-control union
accurately and leaves broader acceptance pending.

No schema/API/configuration or state-version extension is required: the key types
were already accepted and had assigned V1 tags. Prior batch review scopes and
historical performance records remain unchanged. All eight performance targets,
phase 1.2 dictionary implementation and whole-issue completion remain active.

### Verdict

**Approve — final local specialist source review of
`c27db37e5ec5389143e875c40e746d7ea1d719e5`.**

This approves the codec prerequisite and its focused evidence only. It does not
certify full phase 1.2, new-native Python/release integration, required CI and
cross-platform checks, combined 90% coverage, performance, merge readiness or
issue 406 completion. No tests, builds, measurements, CI query, remote mutation,
commit or product/test edits were performed by the reviewer. Only this report
was written.
