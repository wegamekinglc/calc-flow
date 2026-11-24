# Public two-sided interval Join with compact routing

Compare two official Linux CI wheels on one independent hosted VM using
the unchanged `0cca84b4` single-core interval benchmark and original sorted
pair and product oracle. Both dynamic sources contain one million records.
An interval of ±5 seconds produces 10,998,080 matched rows. Measure from
ready enqueue through public Arrow output and final concatenation. Check
every pair identity in canonical sequence order, and every price product,
outside the timed boundary. Record maximum numeric error and output count.

Both normal inputs route compact morsels during production
run routing. This measurement includes the remaining processing, 10,998,080
output rows and public Python materialization. Source arrival order has a
separate Rust regression test. The benchmark target is public p50 ≤350 ms.

Use the original two rounds of ten adjacent AB/BA pairs after a warmup per
wheel and round, with maintained paired median intervals and the 5% rule.
Verify artifact, release, wheel, installed tree, wheel build image and source
identities. Measure under the original one-CPU preflight, host/sibling load
limits and one-second paging and memory checks.

The static and interval public compact-routing comparisons share 600 seconds total.
Allocate 180 seconds to the static hosted job, 300 seconds to this job and
60 seconds combined off-runner qualification and cleanup. Preserve 45 seconds
for settlement and the artifact inside each hosted job. Earlier complete jobs
took 46 and 126 seconds respectively. Builds estimated 20–30 minutes each
are separate. Stop once on any failed oracle, deadline, provenance or resource
check and preserve all completed observations.
