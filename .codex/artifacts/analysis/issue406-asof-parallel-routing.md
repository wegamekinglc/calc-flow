# Issue 406: parallel ASOF capture routing

This increment follows the [parallel-routing specification](../specs/issue406-asof-parallel-routing.md)
and uses reviewed parent `edb73649`. That parent's ordinary product is exactly
`6739fa2b`: subsequent parent commits change benchmark evidence and documents,
not Rust/Python package sources. Earlier timing binaries retain their recorded
identities. The eight-target objective, prior OFF/32 uncertainty and stopped ON
qualification remain active.

## Diagnostic evidence

Artifacts are under `target/issue406-routing-parallel-profile-20261013/`.
The frozen plan SHA-256 is
`97ca749aceebfd3b35c5a64623b6a1539bfa44a5b31d2da8d94e563f506c8fd9`.
One ASOF case uses 1M rows/input, 64k batches, true OFF and 32 CPUs, with one
warmup and five sampled windows. It reuses the attested ordinary `6739fa2b`
native `a131c569...340011`, without building another product. Acquisition,
symbolization, full oracles, owned cleanup and sealing take 10.614 / 180 seconds.

All six invocations pass oracle/lifecycle/cleanup and zero-state-file checks.
Independent validation verifies all 1,650 frozen inputs and 23 result seals.
The five windows contain 659/697/669/669/660 samples, totaling 3,354; 3,075
include native frames. No owned process group survives or requires forced kill.
Sixteen lost lifetime events have unknown processing-window membership, and
WSL2 host isolation remains unverified. Profiling does not establish a timing
improvement or a parallel regression.

Output matching accounts for 33.81% inclusive / 18.78% self samples;
`ChunkData::output_view` accounts for 5.28% self samples. Admission remains
material at 39.36% inclusive. Shares overlap. Source inspection identifies a
serial preparation pass that builds a view and hashes an encoded key per row
before submitting parallel probe work. The worker already receives integer
bucket coordinates, so routing preparation can reuse the existing chunk IDs.

## Implementation and focused verification

Parallel capture walks canonical output runs. Eligible runs resolve their
dictionary keys into a 64-slot stack array, then append worker inputs by integer
ID. A local `None` caches a missing bucket. The existing global route map still
deduplicates actual buckets across independently encoded chunks. Legacy rows,
more than 64 slots and runs with fewer than four selected rows per slot use the
generic route lookup. Shard compaction remains after all capture, preserving
route coordinates throughout preparation.

The first selected row's cancellation/progress check precedes route setup; the
remaining checks use the same global position and 1,024-row yield boundaries.
Missing keys still advance global position. Worker count, input/output credits,
right-bucket lifetime funding, matching, restore order, collector acceptance and
checkpoint contracts retain their existing boundaries. The route array adds
bounded stack scratch within the existing fixed scratch allowance and no
retained heap state. `OutputRun` is re-exported only within the ASOF module.

The focused regression first records 1,037 encoded-key hashes for 1,024 selected
rows and sixteen keys, including two missing keys. The first implementation
passes its bound of 128 while preserving exact binary-search matches, status
and reservation refund. Capture-specific differential controls cover dictionary
boundaries, short cuts, partial consumption, independent dictionaries, retired
slots, legacy rows and cancellation. They are invoked by the existing budget
regression, adding no libtest registrations.

The scoped finalization run passes 27 checks and exposes one new fixture error:
the all-missing-key control initially attempted to construct a batch without
any record batches. The helper now omits that empty reference input; the exact
affected registered test passes with all capture-specific boundary scenarios
in 0.13 seconds. The unchanged 27 passing checks are not repeated. They include
owned-work abandonment, cold checkpoint recovery and rejected collector
continuation. Default core registration remains 2,033.

Scoped `cargo clippy -p calc-flow --lib --tests --all-features -- -D warnings`
passes after the final refactor in 26.41 seconds. It initially identifies the old `output_iter` helper's
last production caller as removed; that iterator is now explicitly test-only,
without suppressing a lint. The first-row refactor removes duplicate offset
branches; the exact affected regression passes again in 0.13 seconds, without
repeating the unchanged 27 passing checks. Lizard 1.24.1 recognizes all 36
functions in the complete probe/test modules, explicitly covers every changed
function, and reports maximum complexity eight. Formatting, generated-contract
drift and whitespace checks pass. Product-wide CI
and coverage are not certified by these focused local checks.

## Paired comparison scope

Before acquisition, freeze the exact ordinary parent product `6739fa2b`, the
new committed candidate, their wheel/native identities and the maintained
harness. Reuse the attested parent native; estimate the candidate release build
at 6–14 minutes, recorded separately. Both arms must match compiled dependency
features/profiles, toolchain, environment and normalized lock identity.

| Case                        | Connection to this change                                |
|-----------------------------|----------------------------------------------------------|
| ASOF 1M/input, 64k, OFF, 32 | Direct parallel probe capture and end-to-end output path |
| ASOF 1M/input, 64k, OFF, 1  | Necessary unchanged serial-path control                  |

Fix two rounds of ten adjacent AB/BA pairs and one warmup per arm per round,
with 600 seconds inclusive of preflight, fixtures, processing, full oracles,
CPU evidence, lifecycle settlement and seals. Use the same sealed Appendix A
CPU accounting at the maintained processing clock bounds in both arms. Stop on
failure, unsuitable resources or budget; do not retry, expand samples or infer
equivalence from uncertainty. A comparison with this parent isolates the new
parallel preparation change; it does not replace the earlier merged-main
comparison or certify the original eight-target set. OFF targets remain
100 ms / 30 ms at one / 32 CPUs, and source review is separate from performance.
