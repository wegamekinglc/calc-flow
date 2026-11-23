# Public interval join retained-state comparison

Register one independent product comparison before acquisition. The change
compacts evicted prefixes in the match-key bucket and expiration-index arrays.
Use the exact previous product tree at a0e0da58f015cd19066241ecde8f3d1924e57e8c
(or a verified product-identical CI tree), and the reviewed compacted product
wheel. Seal the source trees, tested CI checkouts, artifact/wheel/native hashes,
Linux image and ABI before dispatch. Inputs and clocks are those of the
single-worker EngineCase run 45110649865. The older 40 measured observations
established the 80,029,416 bytes per side motivating this change.

Feed one million rows on each side, over 64 entities, with ±5 seconds of
matching time, one 1M-row batch per side and checkpointing off. The public
clock runs from ready enqueue through the complete 10,998,080-pair Arrow
table and final concatenate. Verify all sequence pairs and values, each
watermark and stream dimension, including replay cursor, outside the clock.
After each run require 999,680 evicted rows and 320 live rows per side and
retain the actual reported bytes for each side. Compare all 40 measured state
observations and report warmups separately. State bytes are runtime accounting.

One fresh Linux hosted VM. One eligible CPU and one Tokio/Rayon/BLAS thread.
Two independent sets of workers; one excluded warmup per arm and ten
alternating AB/BA pairs per round. Keep both rounds' exact intervals and
the existing five-percent regression rule, plus descriptive median/P95.
Three-second preflight requires the selected CPU and all SMT siblings at most
20% busy. Every interval requires host mean below 50%, available >=10 GiB,
major faults below 256 and combined swap traffic below 1024 pages. During
measurement sample every second and require siblings below 50% busy,
including the final interval. Stop on the first failed correctness, resource
or time gate; retain partial rows, without a completed verdict or retries.

Total budget 600 seconds. Reserve the already-registered native accounting
at most 60 seconds, up to 150 seconds for local provenance and fixture/unit
preparation, and 350 seconds inclusive for the hosted job from GitHub's actual
job-start through archive upload; leave up to 40 seconds unallocated. At 305
seconds stop acquisition to reserve 45 seconds for worker cleanup and
upload. Report wheel compilation independently. Seal final budget and
immutable artifact selection after the candidate wheel passes its build job.

Before either artifact's collection, the product PR was rebased onto main
39963785925fe8d161c2c2c4fccba2cc16cbf46a. Use this exact first parent for
baseline; the earlier a0e0da58 release remains a reference. Candidate PR 650
has head a1a8424891c3c24f53ab762825b355a5fed3e10b and the actual CI
test-merge f495918864b09349494091523bc50c49db30ac44. Its tree
71ec3b8f9a3aa1b563a149749a064870c945bb8f matches the reviewed head. The
only three changed files from the registered first parent are join/hot.rs,
join/run_tests.rs and join/runs.rs. If either CI build is replaced, verify
the exact actual first parent and compiled tree again before dispatch.

The main-push package job 137491069036 was cancelled before uploading a
wheel. Before measurement, build the same first parent in isolated dispatch
45114543591. Its workflow commit 4dc64db05abfd309b5f72ebeea3a6a9632025ec3
uses the complete first-parent Linux wheel job; its checkout step explicitly
selects the parent SHA. All following wheel build, inspection, isolated
smoke, and upload steps are identical. Wheel builds are timed separately.
