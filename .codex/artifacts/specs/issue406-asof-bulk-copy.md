# Issue 406: compact complete ASOF payloads by contiguous copying

The complete issue 406 objective and original eight-target table remain active.
This increment starts from reviewed parent `467a3a9b`, with ordinary product
`8d3f6933`. Earlier stopped ON attempts and comparison classifications remain.

## Evidence and direction

The fixed ordinary `05c13fd9` diagnostic records 11.3956% inclusive samples in
`compact_complete_record`. This function is byte-identical in parent product
`8d3f6933`; that evidence keeps its original product/native identity. It is a
diagnostic motivation, not a profile of the new source or a speedup claim.
For non-shareable complete records the function allocates a UInt64 identity
index for every row and invokes Arrow take per column. A complete, contiguous
non-null flat array can instead copy its value buffers in one range operation.

## Requirements

1. Preserve complete-record sharing eligibility. For eligible non-shareable
   records, create independently owned compact flat arrays through contiguous
   Arrow copies, without a per-row index vector. Preallocate variable values
   from the selected offset span. Keep schemas, metadata, offsets and values.
2. Keep canonical null compaction and unsupported shapes on the existing
   vectorized take path. Test explicit eligibility boundaries and short inputs;
   every decline remains observable through a focused work counter.
3. Keep reservation sizes/lifetimes, exact logical charges, limit/error priority,
   caller immutability, checkpoint bytes and restored continuation. Additional
   bounded setup must fit the existing workspace before allocation. Preserve
   current cancellation and worker ownership boundaries.
4. Record a focused failing complexity test before implementation. Reuse the
   existing direct-admission registration to retain the libtest table size.
   Add compact/share/take parity, buffer ownership, sliced/bit-offset arrays,
   nullable variable backing, schema metadata, allocation and funding controls.
5. Run the directly affected direct-admission, workspace and checkpoint controls,
   formatting, scoped Clippy and complexity, schema-drift and whitespace checks,
   followed by specialist review. Full CI/coverage remain separate requirements.
6. Freeze any later comparison before execution: exact ordinary parent/candidate
   releases, directly affected one/32-CPU ASOF OFF paths and a necessary small
   batch control, fixed two rounds of ten adjacent AB/BA pairs, one warmup per
   arm per round and a 600-second inclusive measurement budget. Builds are
   separate. Keep full oracles, CPU/resource evidence and owned cleanup; retain
   every inconclusive/stopped result. This subset cannot certify all eight rows.
