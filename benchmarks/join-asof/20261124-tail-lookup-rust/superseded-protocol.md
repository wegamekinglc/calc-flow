# Rust ASOF candidate timestamp reuse

Measure the maintained `balanced_512` release worker at main
`85af0b53e0a38e19cb472aff18c1ccfd84048912` and the independently
reviewed eligible-tail timestamp change based on that main. Seal the exact
candidate SHA and binary digest in `arms.json` before collection. The
archived baseline worker SHA-256 is
`07b98985100c02329829eca1bf7535129898a10e49c1813669467540ae9de201`.
Its non-fresh Cargo receipt, dependency lock, feature set, compiler, benchmark
and validator must match the current candidate. The previous ready-prefix
change is outside this comparison. Start new processes for every timing on
both arms. Estimate 8–12 minutes to compile the candidate using the same warm
Cargo cache and absolute checkout path. The baseline requires no new build.

Settle 512 pending left rows against 512 retained right rows with 32 keys.
Every left timestamp is after the latest retained right timestamp of its own
key, and inside the tolerance. All 512 candidate probes reach the eligible
last-entry path. The baseline source reads that right timestamp to select the
tail and reads the selected timestamp to test tolerance. Preserve any evidence
of compiler reuse and distinguish mechanical source-level test counts from
actual elapsed performance. This fixture includes ready-prefix selection,
candidate lookup, Arrow materialization, collector, state commit and eviction.
It covers neither earlier candidates found by binary search nor public Python.
Fixture/admission, before-and-after checkpoint captures, the separate full-row
oracle and worker launch are outside the reported direct watermark callback.
The maintained allocation counters stay active during the callback.

Each fresh process first runs a full eight-column oracle over every output row
and its order, then emits one separate sample with row, watermark, retained
state, eviction and output-budget checks. Use two rounds of ten adjacent
alternating AB/BA process pairs, each preceded by one excluded warmup per arm,
for 40 measured processes plus four warmups. Use fresh baseline and candidate
processes in both rounds. Report pooled P50 and descriptive P95 and both
maintained exact paired-round intervals. The registered target is at least
10% lower pooled native P50. Apply the repository's independent 5% two-round
rule for the comparison verdict. Keep every worker report and resource sample.

Use the unchanged three-second resource gate followed by one-second sampling
as workers run. Pin all workers to its single selected CPU, with one Tokio
worker. Coordinate local native builds before timing. The 240-second wall
budget includes binary verification, qualification, fixtures, full-value
validation, warmup, process launch and cleanup. Use a 228-second inner timeout
and 238-second process timeout with two seconds for termination. The candidate
build and prior baseline construction are recorded separately. A resource,
integrity, or deadline failure stops collection with partial evidence.
