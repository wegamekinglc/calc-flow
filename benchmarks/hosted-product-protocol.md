# Public single-window stream phase timings

Use the verified candidate wheel from PR 660, checkout 5989dde9, with one
60-second grouped rolling Float64 SUM through TableExpr.stream. Use the
existing hashed IPC with 100,000 rows, 1,000 keys, 100 batches and 124 null
outputs. Verify sequence, groups, every SUM and null, schema and metadata
and the one-node single-window physical plan. Compute and check the oracle
outside every individual clock. Keep the emitted tables through context exit.

Take one excluded warmup. Follow it with plain, instrumented, instrumented,
plain, plain, instrumented observations. The ordinary observations reuse
the public fresh-stream clock from PR 660. The instrumented observations
partition wall time into declaration, context enter, first output, subsequent
outputs, terminal iterator advance and context exit. Count and time scoped
callbacks for source/Arrow conversion, event-time/structure validation, plan
compilation, sink delivery, Arrow table construction and consumer waiting.
These inclusive callback sums can overlap with one another and with runtime
work. Report the two sets of medians separately to show instrumentation cost.
This diagnostic compares phases on one Python clock and one hosted machine.

Use one fresh Linux VM, one pinned CPU, one native worker and single-threaded
BLAS. Preserve the existing three-second preflight, every one-second resource
sample and the final interval. Stop on the first resource, correctness, time
or source failure, saving partial evidence. The complete stage budget is
600 seconds, reserving 120 seconds for this attribution and 480 for a future
paired public product comparison. The diagnostic reserves 15 seconds for
external registration and 105 from job start including dependencies, fixture,
warmup, all validations, cleanup and upload. End collection by 60 seconds
elapsed to retain 45 seconds for cleanup and upload. Build time is separate.
Expected time is 35 to 60 seconds based on the earlier hosted job. Verify
release workflow, smoke job, checkout/tree, build image, archive digest,
wheel/native SHA-256 and installed package bytes against the registered
receipt. Keep exact source hashes and the package lock in the registration.
