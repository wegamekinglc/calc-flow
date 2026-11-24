# Rust ASOF candidate timestamp reuse after ready-prefix integration

Measure the maintained `balanced_512` release worker at main
`dc1fd149032c586aeb5334095f6862d935b1c8fd` and the independently
reviewed eligible-tail timestamp change based on that main. Seal the exact
candidate SHA and binary digest in `arms.json` before collection. The
archived baseline worker SHA-256 is
`127f86c2fe5620c497123e037294ed5a59a6ef7db2b37379f7b90fbd7f70f0c5`.
Its non-fresh Cargo receipt, dependency lock, feature set, compiler, benchmark
and validator must match the current candidate. The ready-prefix
change is present in both arms. The reused binary was compiled at
`a21d3eed7088795763f511d9aed2fd81920e8201`. Its complete Git tree
`d44249d5436e5bd4929e943f248a73d8b7a8bb68` matches this main merge. Start new processes for every timing on
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

The initial standalone registration on 85af0b53 was superseded before
qualification, warmups or collection when main integrated ready-prefix. Its
candidate build was stopped after approximately 150 seconds. This registration
uses the integrated main baseline and the forthcoming candidate rebased onto it.

The exact registered candidate is f34040ea135004021bbc1f9ca83b186d2edb8d0f,
tree 307532292b498c4bdb247a548cb2a003cd849dec. The baseline is the complete
dc1fd149032c586aeb5334095f6862d935b1c8fd main tree identified above.
