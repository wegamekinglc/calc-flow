# Warm public ASOF prefix delivery

Use the maintained Python worker from the benchmark branch. One active
`TableExpr.stream` has two open sources, source-provided watermarks, and
checkpointing disabled. Preload ten left rows at 0 and 8–16 µs, with an
absent key at 12. The left watermark is 17. Right data and a watermark
of 1 first deliver the 0 µs warm-up. Before starting the clock, wait
for the published ASOF output frontier at 0 µs and verify nine pending
probes with both sources open. Build input and oracle tables outside the clock.

Start just before enqueuing the four-row right batch followed by the
right watermark at 16. Stop when eight left rows have reached the public
Arrow iterator. Verify all six output fields in order, their complete
Arrow schema/metadata, five matched rows and three unmatched. Ties at
8 and 12 resolve to sequence 2 and 4 respectively; distance two is
inclusive and distance three is expired. Verify one row remains pending
at the exact cutoff and both sides are open. Advance to 17, validate the
tail and require an empty EOF drain. Lifecycle, Arrow concatenation,
the independent row oracle, and status checks are outside the clock.

Use one fresh hosted Linux x86-64 VM with one pinned CPU and one Tokio
worker. Compare an equivalent official baseline wheel (only one plan
Markdown file differs from main 85af0b53) and the exact reviewed prefix
candidate. Each round has one excluded warmup per wheel and ten alternating
pairs. Across two rounds there are forty formal clocks and four excluded
workers, each of which also performs the same in-job untimed warm-up.
Compute maintained exact paired-median intervals and verdict; pooled
p50/p95 are descriptive microseconds. GitHub Actions CI product builds
are independent and may take 20–30 minutes per wheel.

One attempt, ten minutes total. Reserve up to sixty seconds for off-runner
functional qualification and use an eight-minute hosted timeout including
both sealed wheels, locked dependencies, resource qualification, worker
startup, warmup, validation, upload and cleanup. Keep 45 seconds of
the hosted window for settlement. Require matching official build image
and version, actual build checkout tree, wheel/native/installed package
hashes, identical worker environment, and the maintained three-second
preflight and one-second memory, paging and CPU sampling. Stop on the
first failed gate or exhausted budget and preserve partial receipts.
