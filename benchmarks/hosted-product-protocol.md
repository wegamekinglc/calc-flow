# Public static dimension Join with compact morsel routing

The 8ac1e41d observer uses the maintained public
`engines/1000000/calc-flow-stream-1t/join` case. Seal a 64-row dimension before
enqueuing one million fact rows. Receiving the dimension first disables
deferred-left processing. Each fact follows the production run routing path,
which receives a compact Arrow payload after admission and preserves the original source row IDs.

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
ba5471bb matches all product sources at 85af0b53. Verify the actual checkout
trees, official Linux artifact and installed file hashes, same wheel build
image and version, and distinct native module hashes. Two rounds of ten
alternating pairs follow the maintained exact paired median intervals and 5%
rule, after one excluded warmup per arm and round. Report both intervals,
pooled p50 and p95, and the separate absolute public p50 goal of 30 ms.

The two public cursor workloads share a ten-minute total budget. The static
job has a three-minute wall limit, the interval job five minutes, and
off-runner qualification and cleanup share at most one minute. The remaining
minute is unallocated. Keep 45 seconds within each hosted limit for cleanup.
Previous complete hosted jobs, including receipt upload, took 46 seconds and
126 seconds respectively. Product wheel builds take roughly 20–30 minutes
each separately. Enforce the three-second CPU/sibling, paging and memory
preflight and one-second monitoring throughout collection. Stop on any
deadline, provenance, oracle or resource failure. Run each registration once.

The first compact route candidate cd879969 regressed the static case in both rounds. This comparison is reserved for a new implementation that bypasses per-row staging for a contiguous admitted segment. The reference remains the verified pre-change main runtime. Register both workload arms before measuring and require a complete static comparison before clearing the earlier regression.
