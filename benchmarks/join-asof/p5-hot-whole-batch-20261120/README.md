# P5 whole-batch Join experiment

Candidate `186adde0` replaces the old per-morsel run/key state machine only for admitted single-batch quote schemas. Static Join uses DataFusion HashJoin; symmetric interval Join uses native BandJoin row locators and direct full-column materialization. The plan and acquisition command are recorded in `plan.json`; raw samples and native identity are in `result.json`.

The baseline is sealed candidate `705adc7f` from `../p5-key-run-cache-20261120/result.json`. Both cases used one CPU, two warmups and ten measured samples, with the maintained complete-output oracle on every sample. The release build took 724.140 seconds separately; acquisition took 47.213 seconds within the 600-second budget. No samples were added after seeing the result.

| Case | Baseline P50 | Candidate P50 | Candidate P95 | Target | Complete oracle |
|------|-------------:|--------------:|--------------:|-------:|-----------------|
| Join 1M, one CPU | 399.845 ms | 486.163 ms | 552.442 ms | ≤30 ms | 10/10; 1,000,000 rows |
| Interval 1M, one CPU | 1,076.113 ms | 1,222.780 ms | 1,266.255 ms | ≤350 ms | 10/10; 10,998,080 rows |

**Both absolute gates failed, and this is not a performance upgrade.** The descriptive changes are +21.6% Join time and +13.6% interval time, not paired significance evidence. Earlier direct native `collect` diagnostics are not stream acceptance results and did not reproduce here.

The next diagnosis must establish whether the production plan actually enters `HotJoinCore` and separate physical execution from runner/admission/materialization costs. Do not repeat the unchanged acquisition to seek a better median. Multi-batch hot-path inputs and durable hot-path restore remain unsupported and unmeasured.
