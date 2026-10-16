# DAL-345 nullable result funding correction

Fixes the independent review's P1 on PR #453, starting at
`e49031f627342b957a19e90e1aac3060d4405782` (Rust source `e5987c9c`).
The nullable correction's source is `889566de72a81dcc65a75ac634e58f97b9bc6ba6`.
The subsequent preflight correction is recorded below. The original
[validation and resource receipt](issue438-join-morsels.md) retain their source
provenance; no paired performance observations were repeated or relabeled.

## Reproduction and fix

The new Rust regression restores five matching opposite rows containing valid
key/time and 50 all-null Float64 columns, then admits 64,000 non-null incoming
key/time/Int64 rows. This smaller input reaches the same 320,000 run visits,
four units of 80,000 matches and eight chunks per unit as the review example.
The finite runtime pool is 1 GiB. Restored payload owners are checked explicitly.

Before the fix, the actual Rust result was **35,744,640 bytes per unit** by
`RecordBatch::get_array_memory_size`, against a surviving result lease of only
**16,682,240 bytes per unit**. This includes Arrow buffer capacities and array
controls; the review's 35,700,000-byte values/validity calculation excluded the
extra allocation/control bytes. The focused test failed at `backing <= lease`
(one failed test, 1.01 seconds after build).

Morsel result funding now uses projected physical widths independently of V1
state fees. Null fixed-width values retain their full width. Boolean/validity
bitmaps are bounded by one byte per row; selected Utf8/LargeUtf8/Binary/LargeBinary
offset spans account for variable values. FixedSizeBinary uses its declared
width. Duplicate/reordered projections are charged per emitted column. The
existing twofold buffer growth headroom and 4 KiB per array/chunk cover padding,
controls and terminal offsets. Overflow or an unsupported physical shape declines
the optional optimization. The 50-column case now estimates 81,002,240 bytes per
unit, declines before result allocation and emits through the prepaid fallback.

Before publication, actual Arrow array memory must also fit the admitted result
lease. This defensive check rejects an unexpected kernel capacity increase while
scratch is still alive. Rejected backing is destroyed before result credit.
The 16 MiB/unit, 64 MiB total result and 512 MiB scratch caps remain unchanged.
V1 logical fees, error priority, public APIs and checkpoint layouts are unchanged.

## Focused evidence

All builds use Rust 1.88.0, `CARGO_BUILD_JOBS=2`,
`CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0` and unchanged Cargo.lock.

```bash
# Before implementation: one failed test, reporting all four underfunded units.
cargo test --locked -p calc-flow --features benchmark-checkpoint-diagnostic \
  --lib test_morsel_restored_wide_nullable_backing_requires_safe_fallback \
  -- --test-threads=1 --nocapture

# After correction: affected morsel behavior and five new regression tests.
cargo test --locked -p calc-flow --features benchmark-checkpoint-diagnostic \
  --lib morsel -- --test-threads=1 --nocapture
cargo clippy --locked -p calc-flow --lib --tests --all-features -- -D warnings
cargo fmt --all --check
git diff --check
git diff --exit-code -- schemas/project-v3.schema.json
```

Affected morsel checks: **13 passed, 1 ignored** (the existing opt-in paired
diagnostic). Core all-feature lib/test Clippy, format, whitespace and generated
schema drift checks pass. Lizard 1.23.0 inspects 43 functions in the two touched
production modules and two new test modules; maximum CCN is 8, with no exemptions.

The five new tests cover the restored wide case's cap refusal and full fallback
output; narrower ordinary and reordered/duplicate projected restored output;
scratch/result pool refusal; success and downstream failure; live result leases
through collector suspension; no state commit after failure; complete refunds
after cleanup; and unchanged caller input/checkpoint status. Narrow ordinary
units hold 4,505,472 bytes with 10,309,376-byte leases; projected units hold
2,583,168 bytes with 5,891,072-byte leases. Scratch is already refunded before
these result-only comparisons and remains absent throughout emission.

Direct physical tests compare actual take/interleave allocation with the funding
bound for Float32/64, Decimal128/256, Boolean, Null, FixedSizeBinary and all four
byte-array forms, using sliced sources, nulls, selected/non-selected values and
projection. An injected actual-capacity mismatch verifies destruction and refund.
Affected pre-existing morsel tests retain ordinary/projected, both-side ordering,
state/limit priority, native refusal, cancellation/deadline/panic/reset and
blocked/rejecting collector coverage.

The original 396 Join, 20 integration and 30 gather lifecycle results remain
evidence for unchanged code. Full regression, cross-platform and 90% coverage
remain CI gates. Conservative funding can send additional nullable/wide shapes
to fallback; performance of those shapes was not measured. The original
development diagnostic's process CPU increase of about 20–21% remains a
limitation, and release/#443 performance acceptance remains unmeasured.

## Materializing preflight ordering correction

The follow-up review found that FixedSizeBinary could enter fused ranges before
result admission. The range planner's generic branch materializes bounded Arrow
chunks to validate their sizes. Its width predicate excludes FixedSizeBinary,
although the fused output predicate previously accepted that type. The source
for this follow-up is the commit containing this section; it starts at `889566de`.

A new real-worker regression uses a 1 GiB runtime pool, five projected opposite
rows with FixedSizeBinary(1 MiB), and 64,000 incoming rows. Only 20 incoming rows
per unit match in time: 320,000 estimated visits yield 400 actual matches, 100
per unit. The projected state has funded RestoredV2 owners. Before the fix, all
four units recorded **67,108,976 bytes** of peak actual Arrow array memory while
the entire batch held only **731,552 bytes** of morsel scratch. Exactly one
morsel admission attempt occurred (scratch); no result admission occurred.
The new no-worker-copy assertion failed (one failed test, 0.46 seconds).

Fusion now shares the exact `supports_flat_ranges` predicate used to select the
non-materializing planner. Shapes requiring copying preflight decline fusion
before deferred work or fused workspace admission. Of the previously supported
flat output forms, FixedSizeBinary is the only form excluded by this predicate:
primitive/temporal/decimal, Null, Boolean and the four byte-array forms retain
the non-copy planner. Dictionary, view and nested output were already excluded.
The fast fit check also performs no materialization. This keeps all fused range
planning on widths/selected offsets/validity, before funded result construction.
The existing split route and its bounded serial preflight remain in place.

After the fix all four real fill workers report **0 copied array bytes, 0 morsel
credit and 0 morsel admission attempts**. The fallback emits all 400 full-width
binary rows with their original five-row order and validates every output byte.
The regression also checks caller input equality, successful state publication
and complete cleanup refunds. A second test covers simultaneous match/state
limits (match error wins), a byte budget smaller than one binary row, and a sink
failure after one accepted chunk: no retained left state, row IDs or emitted-row
metrics commit, the accepted sequence is preserved, and all credit refunds.
The array peak counter is test-only; no production tracing was added.

Focused commands, using the same Rust/build environment recorded above:

```bash
cargo test --locked -p calc-flow --features benchmark-checkpoint-diagnostic \
  --lib morsel_tests::materializing_preflight -- --test-threads=1 --nocapture
target/debug/deps/calc_flow-abab16b2e36df81b \
  test_morsel_gathers_before_fill_unit_settlement --test-threads=1
target/debug/deps/calc_flow-abab16b2e36df81b \
  operator::join::materialization::tests:: --test-threads=1
cargo clippy --locked -p calc-flow --lib --tests --all-features -- -D warnings
cargo fmt --all --check
git diff --check
git diff --exit-code -- Cargo.lock schemas/project-v3.schema.json
```

The two new regressions, one retained fusion check and five directly affected
range-planning tests pass (8 tests total). The 13-pass nullable/morsel result
regressions retain their original provenance; the two new tests complete in
3.25 seconds after build. Core all-feature lib/test Clippy, format, whitespace
and lockfile/schema drift checks pass. Lizard 1.23.0 checks 52 functions in
`morsel.rs`, `materialization.rs` and the new regression module: maximum CCN 8,
without exemptions. The nullable/morsel result
evidence above remains scoped to `889566de`; the funding implementation is
unchanged by this correction. The same prior broad evidence and CI boundaries
apply. FixedSizeBinary now always uses the split route; performance of this
fallback and release/#443 targets remains unmeasured. No performance samples,
budget changes, public interfaces or checkpoint changes were introduced.
