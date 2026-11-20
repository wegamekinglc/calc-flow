# P4 dense SUM vectors — small improvement, gate failed

Runtime candidate `439395a5`, sealed at source HEAD `e205b0bf`, measured **1,524.5 ms P50** and **1,598 ms P95** on the 10M-row, one-CPU example30 gate. The ≤500 ms threshold failed. Relative to the archived 1,669 ms baseline, the descriptive time reduction is approximately 8.66%; this is not a paired statistical comparison.

Ten measured example30 processes and two warmups completed in 18.693 seconds within the 600-second budget. Every process passed the built-in `check_positions` validation. The example build completed separately in 265.386 seconds of task wall time.

Raw samples are in `result.json`. The candidate remains because the paired P6 evidence shows a statistically supported improvement and P4 is not a regression, not because the P4 gate passed. The next work must attribute the remaining source, expression, routing and snapshot costs before choosing another optimization.
