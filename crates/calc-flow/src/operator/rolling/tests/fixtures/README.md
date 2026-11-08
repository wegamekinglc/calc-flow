# Shared numeric Rows legacy checkpoint

`shared-rows-8d8be7d9.json` was captured by a freshly compiled core from
`8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`. Production source in the detached
legacy worktree was unchanged. The adjacent generator patch contains only a
test-module declaration and the auditable fixture helpers/capture test; its
SHA-256 and the actual old binary hash are in the provenance JSON.

The fixed scenario has two entities and same-input Rows 5/80 means. Two Arrow
RecordBatches contain 164 original rows. Watermark 160 finalizes the prefix and
leaves four pending rows at checkpoint epoch 1. Appending 68 rows and advancing
to 232 produces 72 rows per continuation. Both numerical profiles have frozen
inline metadata, exact segment bytes and checksums, logical budget charge, and
separate uninterrupted/restored output bits. The old binary itself restored
the captured snapshot before recording the restored continuation.

The candidate test requires exact equality with every frozen field. In
particular, identical writer bytes remain accepted by the old reader; the
uninterrupted and restored routes each compare to their corresponding old
route, without strengthening the baseline's cross-recovery floating-point
contract.

The fixture SHA-256 is
`b14aae9b7b9d65e1e01d0d144e481bfad7a7feafaaf3111991f4364493888825`.
This is historical evidence: do not regenerate it to accommodate an encoder
or numerical change. To reproduce capture, apply the adjacent test-only patch with `git apply --unidiff-zero`
to an isolated checkout at the pinned commit and run
`capture_pinned_baseline_shared_rows_fixture` with
`CALC_FLOW_P3_LEGACY_FIXTURE_OUTPUT` pointing to an output under `target/`.
