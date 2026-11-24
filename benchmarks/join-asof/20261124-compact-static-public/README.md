# Public static Join after compact input routing

The 64-row dimension was sealed before one million fact rows entered the
public Python stream. From ready fact input to the concatenated Arrow result,
the baseline p50 was 95.126 ms and the candidate p50 was 104.389 ms. The
candidate exceeded the 30 ms target by 74.389 ms. The two paired rounds
both exceeded the maintained 5% regression threshold.

| Statistic                        | Result                     |
| -------------------------------- | -------------------------- |
| Baseline / candidate p95, ms     | 96.450 / 106.442           |
| Round 1 median and 97.85% bounds | +9.45% [+8.13%, +11.02%]   |
| Round 2 median and 97.85% bounds | +9.90% [+8.86%, +11.15%]   |
| Maintained paired verdict        | regression                 |

The 20 measured pairs and four excluded warmups each produced one million
verified results with zero absolute error. The Join consumed 1,000,064 input
rows, emitted one million rows and 16,000,000 bytes, and reported zero
retained or evicted facts. All 16 host-resource intervals passed. The
hosted job took 45 seconds within its 180-second allocation.
[Public job](https://github.com/wegamekinglc/calc-flow/actions/runs/45395479694/job/138450007891)

The admitted baseline wheel's actual tree differs from the product's parent
only in the incremental-performance planning document. The candidate wheel
used the exact reviewed #722 tree. Both official release jobs used Ubuntu
26.04 image 20261116.202.1, with identical package and thread settings
during measurement. The registration fixes the observer, output oracle,
sample count and total budget. The archive preserves the original bytes of
all 523 artifact files, including every partial and completed worker request,
the two physical wheel identities, and raw resource intervals. The receipt
includes both archive digests. [Registration](registration.json) · [Source and release proof](source-qualification.json) · [Statistics and digests](receipt.json) · [All raw evidence](raw-evidence.tar.gz)
