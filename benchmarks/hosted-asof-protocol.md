# Public ASOF stream with six projected columns

Exercise the production ASOF gather after a single right and a single left
Arrow batch. The 510 right rows use key 1 and timestamps 4, 8, …, 1020 µs,
with sequence 1 followed by 2 at each timestamp. Left timestamps and
sequences are 0 through 1023, with absent key 0 at 1022 µs. The public
`TableExpr.stream` runs a backward ASOF join with inclusive tolerance
2 µs. Each source yields one Arrow table and reaches EOF with a
source-provided watermark policy. Checkpointing is off. The pipeline projects
both sides' key, timestamp and sequence.

Build the input tables and declare the expressions outside the measured clock.
Start immediately before constructing the public stream and stop after it
naturally drains, closes and the returned Arrow tables have been concatenated.
This includes runtime compilation and source/ASOF/sink startup, admission,
selection, materialization, public delivery and cleanup. Outside the clock,
an independent closed-form oracle compares all six column types, nullability,
values and left output order. It verifies the latest right sequence on every
tie, the inclusive two-microsecond boundary, and all 260 unmatched outputs.
The completed job must have accepted 1024 and 510 rows, emitted 764 matched
and 260 unmatched rows, and have zero processing tasks and pending left rows.

Use one hosted Linux x86-64 VM, one pinned CPU and one Tokio worker. Run two
rounds of ten adjacent alternating baseline/candidate pairs, plus one excluded
warmup per arm each round. Every fresh interpreter verifies its inputs and
complete output. The statistics use per-job durations for the maintained
exact paired-median intervals, 5% rule and descriptive pooled p50/p95. The
public latency includes more work than the native allocation measurement.

Validate official CI artifacts, actual checkout, clean release, installed
wheel files and native module, identical build image/version, common locked
dependencies, measured machine and workload hashes. The registered baseline
is product-identical to 8ce0b7c8. The single dispatch binds the candidate
head and artifact ID before timing. Stop once on any provenance, full oracle,
resource guard or budget failure and retain partial observations.

Allow 600 seconds total, with an eight-minute hosted wall limit including both
wheel downloads, installation, qualification, every process, upload and
cleanup. Reserve 45 seconds in the hosted window for settlement, and up to
60 seconds for off-runner functional qualification. Product CI builds take
about 20–30 minutes each separately. Keep the maintained three-second
preflight and one-second CPU, memory and paging monitor.
