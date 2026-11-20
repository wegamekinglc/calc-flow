# P4 DataFusion delegation — sequence-gate result

Candidate `5d666afa` did not activate on example30 because its actual schema has no `sequence` column. The measured P50 was **1,711.5 ms**, P95 1,773 ms, so the ≤500 ms gate was not met. The result is close to the historical 1,669 ms baseline and does not establish a performance verdict for whole-batch DataFusion aggregation itself.

Ten measured example30 processes and two warmups completed in 20.857 seconds within the 600-second budget. Every process passed the built-in `check_positions` validation. Raw samples are in `result.json`.

The next candidate uses `ROW_NUMBER()` in a single-partition subquery to derive first-seen ordering without requiring a sequence field. That is a new candidate, not a favorable retry of this one.
