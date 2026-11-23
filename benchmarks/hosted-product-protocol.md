# Public single-row native Arrow export

Compare the current direct parent release to the reviewed session record
export change, with one immutable source tree per arm. Bind both Linux
CPython 3.13 ABI3 release wheel builds to the same workflow recipe, image,
wheel and native hash. Use the candidate CI checkout's actual first parent;
a product-identical wheel requires a complete input/source comparison. The
one timed workload is WarmSessionScenario with dual_sma_spread, 4,096 warm
rows and a single new row, one entity, exact projection, checkpointing off,
an already formed native Batch and explicit Watermark. Admit one worker
and one affinity CPU. Warmup, forced-GC, phase instrumentation, known output
and whole clock follow the maintained scenario and previous PR 645. Every
worker must verify its warm history digest, start row 4096, output/finite
values, zero pending rows on clean finish, plan fingerprint and phase probe.
Input/validation/process startup and GC are outside the individual sample
clock and inside the experiment budget. Each new process starts at row 4096.

Run 2 independent rounds, one excluded prewarm per side then ten adjacent
pairs per round in alternating order, maintaining the same comparison and
exact 97.8516% per-round change intervals. Apply the maintained 5% rule;
report medians, P95, total call and native phases, and whether previous
80/120 microseconds interim p50/p95 targets are met. Freeze raw timings,
host diagnostics, all first-parent and artifact proofs. Guard preflight
and per-second machine conditions as in the maintained hosted sparse case
and stop on first failure, no timing retries. No other timed workloads.
Budget 600 seconds total, including local trace and paired artifact prechecks
<=120s, host setup/Arrow fixtures/all processes and cleanup <=480s including
45s cleanup reserve. Separately report builds. Public functional tests will
also demonstrate each nonempty one-record push skips one former generic
Arrow dispatch and the returned buffers outlive producer/capsule references.

After review of the per-record dictionary order guard, the final product
head is 83618b1f2450570173fc227afda62045e123c73c. Its tested merge is
bc5b2374968c9f066bbf28bdae07e41633578b36 over first parent
b8b897610342ba5a7bf375e181a9a5d041ede8bc. Exactly two files change
from that first parent (the PyO3 exporter and the public session test);
both changed blobs match the final reviewed PR verbatim. Prefer these
two CI builds. The earlier b79bc5f9 release remains sealed as a
reference. Should the first-parent artifact be cancelled, build the
reviewed explicit commit over its actual declared parent or rebuild
the missing exact parent independently with the original Linux recipe.

First-parent main package job 137507892691 was cancelled before upload.
Its exact source is now built through dispatch 45120573404, workflow
commit 35d4c124b9c9647b6a52b4f5b22989b5270d0a86 with fixed checkout
b8b897610342ba5a7bf375e181a9a5d041ede8bc and the original tested Linux
wheel steps. Candidate Linux run 45120320420 is the PR's official build.
