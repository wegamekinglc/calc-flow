# Convenience-stream defaults paired measurement

Date: 2026-11-16. PR: #617. Protocol: [protocol.json](protocol.json); collector:
[collect.py](collect.py); raw pairs: [case A](case-A/pairs.json) and
[case B](case-B/pairs.json); full result: [results.json](results.json).

Both arms run the maintained `test_stream_operator_lifecycle[rolling_state]`
harness from the candidate checkout through `Program.stream(inputs)` with no
config or watermark policy. Only the sealed release wheel differs. The baseline
is `b9a46e12` (product-identical to `39cab2fb`); the candidate is `3235ca37`.
Two rounds of ten alternating fresh-process AB/BA pairs follow the maintained
`release_pairs` rule. The timed scope is the complete stream lifecycle through
Arrow output, including compilation, job startup and completion; fixture
construction and validation are outside timing.

The collector was refactored for complexity limits after collection. The
`collector_sha256` in `results.json` identifies the exact collection-time
version at commit `55514073`; re-evaluating both archived `pairs.json` files
with the refactored collector reproduces every verdict and numeric result
exactly.

The two default changes are measured together: checkpoint-off convenience jobs
and per-batch generated watermarks. This harness cannot pin one while varying
the other, so the result cannot attribute between them.

## Results

Total elapsed: 297.0 s of the 600 s budget. Both cases completed, with no
errors and no unmeasured cases.

| Case | Workload                                      | Baseline p50 | Candidate p50 | Median change | Verdict      |
|------|-----------------------------------------------|-------------:|--------------:|--------------:|--------------|
| A    | 20,000 rows, 32 ordered batches (640-row max) | 18.24 ms     | 16.54 ms      | −9.33%        | inconclusive |
| B    | 960 rows, 2 ordered batches (same batch size) | 8.57 ms      | 5.21 ms       | −39.24%       | improved     |

Case A's round changes were −9.57% and −7.55%; the maintained confidence
intervals do not establish an improvement or a confirmed regression. Case B's
round changes were −41.67% and −39.69%, with both intervals entirely below
−5%, meeting the maintained `improved` rule.

Case B is a batch-count control at the same batch size, not a large-batch
control: no maintained convenience-stream harness varies batch size. These
results support the change under the maintained paired rule, but do not
establish equivalence for case A or separate the two defaults' contributions.
