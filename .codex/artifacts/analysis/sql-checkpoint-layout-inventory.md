# SQL checkpoint layout inventory

DAL-510 / S0a-1. Baseline: `1e6bb3f6489409bc19f47f3e8afd9b1a705a3a7c`.
An explicit fetch confirmed HEAD and origin/main at this SHA before changing
production code. This evidence was generated in the dedicated task checkout;
the historical DAL-441/442 results only guided selection of existing tests.

The change adds layout 5 to the Python SQL capability and updates its exact
catalog assertions. `state_version=2`, capability schema 5, native formats,
bindings, stubs, graph construction and execution remain unchanged. No new
inventory API is needed: a regression obtains the inventory from real managed
native captures, verifies segment SHA-256 and aggregate results, and compares
the observed layouts to the public catalog. The streaming guide distinguishes
operator state version, layout, and native accumulator semantics.

## Native provenance

The task checkout built `calc-flow-python` with
`cargo build --locked -p calc-flow-python`, reusing the existing Cargo cache.
Build settings: `CARGO_BUILD_JOBS=3`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0`, and
`PYO3_PYTHON=/home/wegamekinglc/anaconda3/bin/python` (CPython 3.13.9).
Build completed successfully in 4m43s. The extension and this checkout's Python
adapters were copied under `target/sql-checkpoint-layout-inventory/python`;
both imported module paths were verified there. All 52 Python adapter files
were checked byte-for-byte against the candidate checkout. No extension was
put in source.

- Native SHA-256: `b48ee1dc9f4621b79eb87ae79f746c894d1ee0ca65568897d56e99b20eef8857`.
- Native source inventory SHA-256: `39efefe1e4eb559e12105d1016ce7731fb4cd2351cf009b8c4b86e9dbdd8b18f`.
  This hashes sorted Git-tracked paths (NUL-terminated) and file bytes under
  the three crates and vendor, plus Cargo.toml and Cargo.lock.
- `git diff 1e6bb3f6489409bc19f47f3e8afd9b1a705a3a7c -- crates vendor Cargo.toml Cargo.lock`:
  empty. The metadata-only candidate uses these same native sources and binary.

## RED and GREEN

Commands used `PYTHONPATH=target/sql-checkpoint-layout-inventory/python`
and `JAX_PLATFORMS=cpu`.

```bash
python -m pytest python/tests/test_sql_checkpoint_layout_inventory.py -q \
  --basetemp=target/sql-checkpoint-layout-inventory/red
```

RED: 1 failed in 0.79s, after all three captures and result/digest checks
succeeded: catalog `{3, 4}` differed from observed `{3, 4, 5}`. An earlier
fixture attempt used a logical source name with the explicit runner and was
corrected to use the plan's public physical binding IDs before accepting RED.

The same command with `green` as basetemp passed (1 test, 0.79s) after the
single capability tuple correction. No native rebuild was needed for that
Python metadata change.

```bash
python -m pytest python/tests/test_capabilities.py -q \
  -k 'empty_runtime_capabilities_are_frozen_and_session_scoped or capability_catalog_reports_every_durable_state_layout'
```

2 passed, 39 deselected in 0.83s. A broader attempt on this module had
40 passed and 1 failed because the interpreter lacks optional `jax`; that
array-provider case is outside this SQL change. No test was weakened or
dependency environment changed to hide that limitation.

## Identity and byte comparison

The RED and GREEN runs generated fresh managed checkpoints using the same
native binary. For each layout, a comparison of their manifests verified
identical pipeline fingerprints, runtime config hashes, source/sink entries,
operator entries and state checksums. Each segment was read from both roots
and compared byte-for-byte. Only the manifest creation time is outside this
comparison.

| Capture                            | Layout   | Native semantics   | Segments   | Bytes   | Pipeline fingerprint                                               |
|------------------------------------|----------|--------------------|------------|---------|--------------------------------------------------------------------|
| `SUM(value)`                       | 3        | 1                  | 4          | 4351    | 7eb9595b3933179dfe5a564af65259781702e73f488044ae92e52dccd7dbee94   |
| `SUM(abs(value))` retained         | 4        | absent             | 4          | 2722    | 3d9a178c9d50543487a58b75c0a9ee331a30c0dcaa209e7a7eb5f6bfa9a8b4e7   |
| `LAST_VALUE(value ORDER BY seq)`   | 5        | 2                  | 4          | 5774    | 53f1ce81f702d835e6443ffaad4ba17ad7fef3fa3b7ea5c8cebd5918eff1a92c   |

These are terminal full captures of tiny inputs. Full/delta continuation and
atomic rejection are covered separately by the existing native tests below;
this Python capture test does not claim durable replay from `from_arrow`.

## Other verification

With the same Cargo cache and debug/build-job settings:

```bash
cargo test --locked -p calc-flow --lib -- \
  current_checkpoint_tests \
  test_ordered_last_layout_five_control_corruption_is_atomic_with_valid_control_digest \
  test_ordered_last_delta_inventory_and_generation_corruption_fail_closed \
  test_ordered_last_delta_rewrite_and_every_cold_prefix_match_uninterrupted_and_df55 \
  test_checkpointing_hash_records_disabled_mode_without_changing_the_plan \
  --nocapture
```

9 passed, 3121 filtered out, 0.63s execution after 6m58s compilation. This
includes layout-3/4 roundtrip/recapture/continuation, rejection of absent and
unsupported outer layouts before decoding, DF release identity validation,
layout-5 control/delta corruption with atomic refusal and byte-preserved
installed state, 34 ordered recovery prefixes across full/delta base rewrites
against uninterrupted execution and DF55, and the historical fixed config
hash assertion. These are current-checkout results, not historical reruns
claimed from DAL-441/442.

Ruff 0.16.0 lint and format checks passed for the three affected Python files.
`git diff --check` passed; Cargo.lock, project schema and Rust/binding sources
have no diff. Full regression, coverage, other platforms and release artifacts
remain CI responsibilities. No performance measurements were run.

The runtime brief's `.codex/references/{code-style,unit-test-style,component-boundaries}.md`
are absent from this baseline; the actual repository AGENTS.md,
`.codex/agents/cf-implementer.toml` and `.agents/skills/code-style/SKILL.md`
provided the applicable local rules. No examples or component dependencies changed.
