# Public full snapshot GroupBy latency

Compare the Linux CI baseline and candidate wheels on one hosted Linux VM. Both
run the snapshot arm of the warm integer SUM study from 3becc56d. The source
is the sealed composite-key sparse Arrow fixture. Populate 10,000 groups,
then submit twenty 100-row batches. Start each clock immediately before
enqueue and stop after the full cumulative snapshot is received at the public
Python iterator. Validate all 21 snapshots, including column types, row order
and each aggregate against the independent Python grouped sum after natural EOF.

Use one Python process for each observation with one available CPU and one
Tokio worker. Exclude one warm process per side and round. Within each of
two rounds, alternate the order of ten adjacent baseline/candidate pairs.
Report pooled p50 and p95 for 400 append-to-Arrow observations per side and
the two exact paired median intervals over per-process medians. Apply the
maintained five-percent verdict. Improvements in full snapshot reservation
are only one component of this public latency. The 15 ms whole-task target
uses a separate fresh changelog workload.

Baseline product tree is the official Linux CI wheel of main 788631a5.
Main 8ce0b7c8 changes two Markdown files and two benchmark comparator files
from that commit. The workflow dispatch inputs bind the candidate head SHA
and Linux CI artifact ID before startup. Verify both official package and
native hashes, real checkout trees, matching build images, the dependency
lock, workload sources, fixture and row-by-row results. Stop if the CPU,
sibling, memory or paging limits in the maintained sampler are violated.

Reserve 600 seconds total, including 60 seconds for off-runner setup,
validation and cleanup, with a nine-minute wall limit from hosted job start
and 45 seconds of that reserved for artifact upload and worker settlement.
Wheel compilation runs independently in product CI, estimated 20–30 minutes
per build. This is a single registration and a single measurement dispatch.
