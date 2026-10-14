# Issue 406: signed and date key codec prerequisite

Status: seven focused behavior controls and scoped Clippy pass; final specialist
review is pending.
The tests were prepared at `b7768684` on `bb142320`, then rebased unchanged to
`e549f5a43d02d53350de905d355d20eb4144525b` on the stable batch-admission
commit `327a6bbfaa162bddddc97f08ce312b99e393bd6a`. Final specialist batch
review approves that source. No timing or target verdict is established. All
eight issue targets remain active.

## Concrete defect and intended behavior

At the inspected base, Join accepted Int8, Date32 and Date64 key schemas. Its V1
key encoder nevertheless converted signed Int8 through `u8::try_from`, rejecting
negative values, and downcast Date32/Date64 arrays through Int32/Int64 Arrow
types. Equal physical width does not make those Arrow array types interchangeable.
Retaining these keys therefore reported an internal error. Non-retained fixtures bypass
encoding and already succeed; they remain valid acceptance controls.

The phase 1.2 dictionary requires a correct canonical equality reference for all
accepted key types. Correct this prerequisite separately, rather than making a
new dictionary reproduce accidental encoder errors. Previously successful V1
bytes, tags, framing, logical charges and checkpoint contracts must remain
unchanged.

Int8 uses its one signed little-endian byte: -1 is `ff`, -128 is `80`, and
nonnegative values keep their existing byte. Date32 uses `Date32Type` and its
four signed day-count bytes; Date64 uses `Date64Type` and its eight signed
millisecond-count bytes. Dates are key values, so event-time normalization must
not be applied to them. Their existing tags remain 2, 12 and 13, respectively.
The block remains tag, timezone length and bytes, value length and bytes, with
the two lengths encoded as little-endian u32 values.

This intentionally replaces the old regression's expectation of an internal
error for retained keys. It does not broaden the accepted schema types, change
null admission, select a new native route, or change V1/V2/V3 checkpoint layouts,
project schema, fingerprint, version or release settings.

## Focused verification prepared

Three independent codec regressions check sliced arrays at minimum, negative,
zero and maximum values against literal complete V1 blocks. The stateful
regression preserves the non-retained acceptance control, then requires
retention, exact full V1 fees, V1 and V2 capture/restore, continued SQL matching,
complete output/schema equality, sequence and state counters, and owned cleanup.
Existing frozen checkpoint fixtures for previously successful keys remain
unchanged.

Execution started after the batch-admission implementer's build handles settled,
using its unchanged pinned toolchain and shared debug build settings. The four
selected tests, exact head, source hashes and environment are recorded under
`target/issue406-signed-date-key-codec-20261015/`. A compile or fixture failure is
not behavior RED. Implementation follows observed codec failures,
then only affected checks and required module formatting/lint/complexity checks.
Final specialist review is required before integration. No historical stopped
performance acquisition will be resumed.

## RED and correction

The four selected tests all failed for the intended production defects on
`e549f5a4`: Date32 and Date64 each reported the primitive-array type mismatch;
negative Int8 and its retained-state control each reported that the signed key
does not fit u8. The command took 418.26356117997784 seconds, including 6 minutes
58 seconds of compilation; test execution took 0.01 seconds. This is actual
test execution, not a fixture or compilation failure.

The correction reads signed Int8 bytes directly and downcasts the two date
types correctly. Small signed, unsigned and string helpers keep the changed
functions within CCN 8 without altering the other value encodings. GREEN adds
the existing all-native-type SQL/order/frame comparison, frozen V1 checkpoint
fixtures and encode-once charge control to the four original regressions.
The source is frozen during these seven selected tests; no broader matrix,
native package build or performance measurement is running.

The first GREEN invocation passed six controls after compilation of 2 minutes
34 seconds, executing in 0.34 seconds (154.569270128 seconds including the
command). The stateful control then rejected a V2 snapshot created by directly
switching from the test-only V1 producer on the same operator. That V1 capture
clears the pending journal without installing V2 history, so the subsequent
snapshot's empty history disagrees with its nonempty metrics. This was a test
fixture error, not another codec defect or a valid production RED.

The corrected control restores the authentic V1 snapshot into another operator,
runs its actual migration preparation and captures V2. It then restores both
snapshots and compares continued matching. Only this changed control is rerun;
the six unchanged passing controls are retained with their exact evidence.

`corrected-green.log` passes that stateful control for all three types, including
exact retained fees, authentic V1 migration to V2, both restored output schemas
and contents, sequence/counter checks and awaited owned cleanup. Compilation
took 1 minute 18 seconds and execution 0.05 seconds; recorded subprocess elapsed
time is 78.67592345399316 seconds. The focused union is seven passing controls.

Formatting, generated-schema no-drift and whitespace checks pass. The final
changed-function audit recognizes 12 functions in two Rust files, maximum CCN 8.
The final source seal is
`ae3fad337d1ef96fdc8d55ea0164a3f009ee9a3638cd509dffb290bbfaa1eb26`.
The first scoped Clippy invocation found only a test helper taking an unconsumed
`ArrayRef` by value. Its final signature borrows `&dyn Array`, with the same
slice and literal assertions. This test-only lint correction leaves production
bytes and all seven behavior assertions unchanged; those passing controls were
not repeated. The final core lib/test Clippy passes in 177.77890396898147 seconds,
with no lint suppression, workspace-rule change or production boxing. The
changed test file's format check, schema no-drift and whitespace check also pass.
Required CI, coverage, new native-package integration and performance acceptance
remain pending; no merge readiness or complete phase 1.2 is claimed.
