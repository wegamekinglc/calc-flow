# Exact Arrow schema on a one-row session push

The median per-call cost of Python input validation fell from 9.689 µs to
0.401 µs for a 32-column RecordBatch. Both rounds cleared the maintained
five-percent improvement threshold. The whole public session push measured
38.104 µs and 28.987 µs, respectively. Its paired round intervals were
[−31.30%, +2.71%] and [−26.29%, −21.50%], giving the maintained verdict
`no-confirmed-regression`. The first round leaves the size of the public
speedup uncertain.

Each observation averaged 128 calls under normal garbage collection. Use
the [paired observations and artifact identities](python-public.json) for
the scope of these percentiles. The four-column, forced-GC rolling session
uses a different workload.

## Reproduction

1. Extract the recorded release wheel into two separate import roots and
   replace `calc_flow/session.py` with the respective product revision.
   Verify both module hashes and the shared native library hash in the JSON.
2. Use one CPU (CPU 8 here), one numerical-library worker and a fresh
   process for each arm and pair. Build a one-row PyArrow RecordBatch with
   32 Int64 columns named c0 through c31 containing their integer indexes.
   Reconstruct an equivalent schema separately for the validation control.
3. Make a table input with those fields, select c0, open a table session
   with disabled watermarks and warm it with one push. Time the 128 individual
   calls to `_session_batch` with the reconstructed schema. Verify each
   returns the input object. Then time 128 individual `session.push` calls,
   checking the returned single column, row and value outside each clock.
   Verify that finish emits no rows.
4. Run two rounds, each with one warmup per arm followed by ten measured
   pairs. Alternate arm order using `pair_order(round + index + 1)` with
   zero-based rounds, index −1 for warmup, then indexes 0 through 9. Use
   the maintained paired comparison on the per-process means. The 9.51-second
   acquisition included process startup, warmups and validation.
