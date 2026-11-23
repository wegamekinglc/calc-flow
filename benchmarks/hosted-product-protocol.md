# Public single 60-second Duration SUM

PR 660 changes sum readout in production rolling state. Use one optimized
Linux ABI3 CPython 3.13 Release wheel per arm. The tested first-parent
commit and accepted candidate must use the same Linux build recipe and
runner image version. Verify GitHub workflow, job, actual checkout/tree,
archive SHA-256 and installed wheel/module bytes. Build time is separate.

Use the public .stream declaration and the immutable fixture from PR 638,
with only the single 60-second Float64 SUM. There are 100,000 time-adjacent
rows (1 ms apart), 1,000 distinct entities, 100 batches of 1,000 rows,
deterministic nulls and one entirely null entity. The fixture has SHA-256
36847ac453bd58a3a0a918d993ec726d56bbddb8c3af6e8362ca940999286ab0.
Every observed input produces an output. Verify exact schema/metadata, all
sequence and group values, every sum and null, and the open lower boundary
using a separately calculated per-key prefix-sum oracle. Check one rolling
stage with a single 60-second output, time/identity ordering, min periods 1
and checkpointing disabled.

Use the source, retained-output and cleanup clock from PR 638. Start before
declaration and stop after collecting every Arrow table through the public
.stream context and its exit. Fixture, independent oracle and validation are
outside each individual latency; they count toward the experiment's 120 s.
The separate Rust runner has its own clock. The public p50 target is 12 ms.

On one fresh hosted VM, pin one CPU, one native worker and the single-thread
BLAS settings. Keep the checked 3-second preflight, every one-second pressure
interval, and final interval. Run two independent rounds with 10 alternating
AB/BA pairs per round and four excluded warmups in total. Keep the exact
paired intervals and 5% maintained rule. Record every sample and classify
all resource, correctness, time-budget or setup stops without retries.

The stage has 120 s left after the Rust and deterministic reservations.
Reserve 15 s for external provenance and 105 s from actual hosted job start,
including dependency installation, fixture, all workers, validation and
upload. End collection by 60 s elapsed, retaining 45 s for cleanup and
evidence. The prior triple-SUM warmup took 118.973 ms and its stopped job
completed in 34 s; estimated complete single-window job is 45 to 65 s.

The workflow and isolated smoke job for each arm are fixed below in the
companion registration. Before any hosted samples, the 15 s external
provenance allowance ended while fetching the original compile log (5 s
for source selection and 10 s for the final parallel requests). Zero
warmups and zero timed pairs had started. Both builds had finished.
The sole hosted job uses the original build logs to resolve exactly one
tested upload per fixed job and verify source, ABI/release, runner image
and artifact bytes within the registered 105 s, including its first 60 s
available for preparation and collection. No subsequent job is permitted.
