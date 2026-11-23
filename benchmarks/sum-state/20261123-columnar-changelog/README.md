# Columnar sparse grouped SUM changelog

On one CPU, the median time for the complete 21-batch Rust graph fell from
21.757 ms to 10.647 ms. The candidate P95 was 12.200 ms. The two rounds
had median paired changes of −50.09% and −50.76%, with 97.85% sign intervals
of [−51.73%, −48.29%] and [−60.76%, −46.33%]. The maintained five-percent
rule classified the comparison as improved. The candidate exceeded the
10 ms median target by 0.647 ms.

The positions operator's median processing time fell from 16.570 ms to
5.389 ms. Both releases produced 13,200 changelog rows. Every worker checked
all 21 reconstructed position snapshots against the same arithmetic oracle.
The measurement uses the Rust API. Python stream and session calls have
separate clocks.

## Reproduction

Use the [previous sparse SUM protocol](../20261123-sparse-changelog/README.md#reproducing-the-native-case)
with the product revisions and build identities in the [native evidence](rust-native.json).
For an exact candidate source, start with the previous stage's product and
apply the candidate revision's SQL changes under `crates/calc-flow/src/operator/sql`.
The measured source invokes the same Arrow FromIterator implementations
through the type constructor. The final lint edit uses Iterator::collect.
Use the same changelog diagnostic example in both arms. The environment and
fixture hashes are in the evidence file. Reuse the previous release worker
as the baseline. Building the candidate took 879 seconds.

The 17.10-second acquisition covered fixture verification, startup, four
excluded warmups, twenty alternating pairs, output validation and cleanup
within a 402-second budget. The sampler recorded observed peak process-tree
RSS and operator processing time separately from graph latency.
