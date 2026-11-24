# Public interval Join after compact input routing

Each side supplied one million rows to a single-CPU managed Python stream.
The ±5-second interval produced 10,998,080 matches. From ready enqueue to
the concatenated Arrow result, p50 measured 416.892 ms on the baseline
and 417.792 ms on the candidate. The candidate exceeded the 350 ms p50
target by 67.792 ms. Both rounds' paired intervals include zero and extend
beyond +5%. The maintained result was inconclusive.

| Statistic                        | Result                     |
| -------------------------------- | -------------------------- |
| Baseline / candidate p95, ms     | 428.496 / 442.471          |
| Round 1 median and 97.85% bounds | +1.81% [−4.17%, +7.49%]    |
| Round 2 median and 97.85% bounds | −1.42% [−3.80%, +5.14%]    |

All 20 measured pairs and four excluded warmups verified the complete pair
identities in canonical sequence order and all price products. Maximum
absolute error was zero. In each invocation, both inputs evicted 999,680
rows and retained 320 rows occupying 55,016 bytes per input. All 89
host resource intervals passed. No further sampling was performed. The
hosted job took 139 seconds of its 300-second allowance. Including the
independent static job's 45 seconds and charging the entire shared
60-second external qualification and cleanup allowance gives an upper bound
of 244 seconds for the two observations within the 600-second budget.
[Public job](https://github.com/wegamekinglc/calc-flow/actions/runs/45395487646/job/138450029106)

The two official release wheels used Ubuntu 26.04 image 20261116.202.1. The
baseline source tree differs from the candidate's parent only in the
performance plan. The candidate's actual Linux checkout has the reviewed
#722 tree. The archive contains the original contents of all 523 artifact
files, including all measured and partial worker requests, resource intervals,
and the two release manifests. [Registration](registration.json) · [Source and release proof](source-qualification.json) · [Result and archive digests](receipt.json) · [All raw evidence](raw-evidence.tar.gz)
