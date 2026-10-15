# Join columnar retained-state charges

Issue: DAL-339 / GitHub #430, part of #406 phase 1.1.
Starting `origin/main`: `aaf793a3259b3e3d2e18e30eb5e13e06b1141665`.

Retained-row construction now computes frozen V1 payload fees by column in
64-position stack blocks. Non-null fixed widths accumulate one shared constant;
nullable widths use validity bits. Strings and binary arrays read offsets or view
lengths. Nested lists and maps sum child ranges using bitmap populations and
offset differences over valid runs; structs, unions, dictionaries and runs follow
their logical child references. No Arrow cell slices, owning incoming row
descriptors, child-fee vectors or additional heap workspace are constructed.

The block follows the admitted segment's current payload positions, including
sparse selection and copied parents. Retention bits select which positions need
fees. Existing physical IDs and native-key ordinals remain independent of those
positions. The result feeds the existing `StoredRow` charge and state-limit checks.

Speculative column failures set a row bit. The unchanged V1 row oracle replays
that row only after its original key check, preserving its error and priority.
Matching still precedes retained-state validation. All successful row charges use
checked arithmetic. Production state representation, checkpoint bytes/layout,
project fingerprint, output cost model and public bindings are unaffected.

V1 currently downcasts dictionary keys and run ends to Int32. Other physical
widths retain their original error path; a null dictionary key still costs one
validity byte. Arbitrarily nested encoded values still require logical reference
traversal. Structural counters establish no timing or RSS acceptance verdict.

## Verification

The starting implementation made 195 scalar cell-charge calls for the focused
65-row, three-column sliced nullable fixture. The original V1 functions remain
available for differential tests and error replay.

Build settings: pinned Rust 1.88.0, repository-local `target/`,
`CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0`, locked Cargo dependencies. The initial cold test
build took 11 minutes; acquisition of performance measurements was not run.

With those build settings, the following checks passed:

```bash
cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
cargo test --locked -p calc-flow --lib columnar_charge -- --test-threads=1
cargo fmt --all --check
git diff --cached --check
git diff --exit-code -- schemas/project-v3.schema.json
```

The new fee tests passed 10/10 (2,097 filtered), executing in 0.05 seconds after
a 5 minute 39 second candidate compile. They cover sliced/multi-record input,
both sides, all fixed-width families, six byte/string layouts, nested/encoded
payloads, sparse null/late/retained coordinates, empty/all-rejected input, V1
unsupported dictionary behavior, byte limits, overflow and partial-block
cancellation/abandonment. The fixed-width controls make zero scalar cell-charge
calls; a 1,000-value non-null child range reads two offsets and allocates nothing.

The same compiled binary passed 29 existing affected controls (2,078 filtered)
in 1.15 seconds, including real generic SQL, mixed copied/borrowed parents,
layout 1/2/3 recovery, timestamp units, counter/ID error priority, finite-pool
refusal, cancellation, output limits and composite-key full fees:

```bash
target/debug/deps/calc_flow-71dd66c0d944c938 --test-threads=1 \
  operator::join::tests::optimization_tests:: \
  operator::join::tests::columnar_state_tests::test_batch_admission \
  operator::join::tests::batch_timestamp \
  operator::join::tests::output_gather_tests::test_batch_admission_metric_and_conversion_keep_first_row_error \
  operator::join::tests::output_gather_tests::test_earlier_timestamp_error_precedes_later_row_id_overflow \
  operator::join::tests::state_row_limit_failure_is_atomic_with_typed_reason \
  operator::join::tests::match_limit_failure_is_atomic_with_typed_reason \
  operator::join::tests::variable_width_payloads_are_charged_and_limited_deterministically \
  operator::join::tests::retained_row_key_is_encoded_once_and_reused_for_its_charge \
  operator::join::tests::columnar_state_tests::test_owned_copy_denied_funding_keeps_legacy_admission_and_pool_zero \
  operator::join::tests::output_gather_tests::test_nested_dictionary_output_discards_unused_values_under_same_budget \
  operator::join::tests::output_gather_tests::test_multi_parent_nested_dictionary_fallback_discards_unused_values \
  operator::join::tests::persistent_dictionary_semantics_tests::test_persistent_dictionary_composite_sparse_slices_keep_nested_fees_and_order
```

Full regression, cross-platform and the Rust 90% coverage gate belong to existing
CI. No performance target acceptance or merge-readiness verdict is claimed.

## Codacy follow-up

The follow-up starts at PR #447 head
`7a497ebc85c2ee75c71d830a58319bbc315f3e45`; freshly fetched `origin/main`
remains `aaf793a3259b3e3d2e18e30eb5e13e06b1141665`.
Codacy check run `116493748987` reports cyclomatic complexity above its limit
of 8 in `range_sum` (14), `view` (11), and `encoded` (9).

Extract the validity-block accumulation, list-view child range sum, and union
child sum into private helpers. The existing checked additions, validity-byte
accumulator, child coordinates and scalar-error replay remain in place. No
check configuration, exemptions, thresholds, state layout or public API changes.

Lizard 1.23.0 reproduces all three original warnings. After the refactor,
`range_sum` is 8, `view` is 6, and `encoded` is 8; the extracted helpers are
7, 4, and 2 respectively. All functions in both affected files satisfy the
unchanged limit:

```bash
lizard -C 8 crates/calc-flow/src/operator/join/admission_charge.rs \
  crates/calc-flow/src/operator/join/admission_charge/nested.rs
```

This is a behavior-preserving refactor. Reuse the broader original validation
above; no additional benchmark collection or performance claim is needed.

With the same Rust 1.88.0 build settings as the original implementation:

```bash
CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo test --locked -p calc-flow --lib columnar_charge -- --test-threads=1

target/debug/deps/calc_flow-71dd66c0d944c938 --test-threads=1 \
  operator::join::tests::output_gather_tests::test_batch_admission_metric_and_conversion_keep_first_row_error \
  operator::join::tests::output_gather_tests::test_earlier_timestamp_error_precedes_later_row_id_overflow
```

The rebuilt candidate passes 10/10 fee tests and 2/2 error-priority controls.
The fee tests include sliced nested/view/union/dictionary payloads, frozen V1
parity, zero-allocation/prefix counters, byte-limit and overflow refusal, null
and sparse selections, and cancellation/abandonment. No duplicate tests were
added for the extracted helpers.

The focused Clippy check and static checks also pass:

```bash
CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
cargo fmt --all --check
git diff --check
git diff --exit-code -- schemas/project-v3.schema.json
```

New-head remote Codacy/CI and review status is recorded in the PR handoff;
the original head's CI/review evidence is not a verdict on the repaired head.
