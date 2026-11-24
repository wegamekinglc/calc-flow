# Public static dimension Join with an admitted-row iterator

The 8ac1e41d observer uses the maintained public
`engines/1000000/calc-flow-stream-1t/join` case. Seal a 64-row dimension before
enqueuing one million fact rows. Receiving the dimension first disables
deferred-left processing. Each fact follows the production run routing path,
which traverses the admitted rows for that input. Source tables use the normal
single-record Arrow chunks. A compiled microbenchmark for many source records
in one native batch measures a separate part of this optimization.

Start the public clock after the right-side watermark has actually been
observed. Stop after receiving the final Arrow table and concatenating all
results. Each worker verifies one million ordered-by-sequence values and their
exact prices multiplied by the symbol's dimension factor. The final counters
must include 1,000,064 input rows, one million output rows, zero errors and zero
retained or evicted facts. Bound the emitted bytes to the two projected result
columns. The sealed dimension, complete output oracle and worker source come
from the corrected previous public experiment. Measure only this workload;
interval joins have a different fanout and output shape.

Use a dedicated GitHub-hosted Linux runner, one CPU, one Tokio thread and
one shared hash-locked dependency environment. The dispatch binds the
candidate SHA and release artifact before acquisition. Baseline wheel
788631a5 matches all product sources at 8ce0b7c8. Verify the actual checkout
trees, official Linux artifact and installed file hashes, same wheel build
image and version, and distinct native module hashes. Two rounds of ten
alternating pairs follow the maintained exact paired median intervals and 5%
rule, after one excluded warmup per arm and round. Report both intervals,
pooled p50 and p95, and the separate absolute public p50 goal of 30 ms.

Reserve ten minutes for this case, including 60 seconds for off-runner
qualification and nine minutes from job start through final upload. Reserve
45 seconds of the hosted window for cleanup. Product wheel builds take
roughly 20–30 minutes each, separately. Enforce the three-second CPU/sibling,
paging and available-memory preflight and one-second monitoring throughout
collection. Stop on a deadline, provenance, oracle or resource failure. Run
this registration once and retain partial results.
