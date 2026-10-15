# ASOF general batch admission (#435)

Implementation base: `aaf793a3259b3e3d2e18e30eb5e13e06b1141665` (main at
task start). This increment extends admission after the existing ordered
Utf8/UInt64 fast paths; it changes no public schema or checkpoint version.

## Representation and ordering

Each accepted record has a dense selection or physical-row index column,
typed event times, record-local sequence columns, admission key IDs, and an
optional canonical sorting permutation. Empty and fully late records are
discarded before encoding. Payload references address accepted physical order,
independently of the sorting permutation. Slices are copied when their backing
would retain uncharged bytes; projected and zero-column payloads preserve row
counts.

The existing ASOF `InputKeys` interner owns keys and reuses resident encodings.
Ranks are assigned by sorting distinct **canonical ASOF bytes**. Admission IDs
and Join persistent IDs are never ordering proofs. Join's namespace leases and
its frozen key codec have different lifetime/serialization contracts and are
not installed into ASOF state. No shared identity/funding module changes.
Probe-only left keys remain admission-local and cannot create right residents.

Per-record order and a merged duplicate probe use `(time, byte rank, sequence)`.
Resident probes are skipped only after proving the complete sorted input lies
beyond the actual retained boundary. Overlapping inputs retain resident checks;
duplicate diagnostics still precede workspace/state refusal. Existing ordered
left construction is reused after its original strict order/boundary proof.
Existing parallel right key units consume a borrowed-or-columnar identity
iterator with the original row-count thresholds, shard routes, funding,
append-tail proofs, cancellation and atomic bucket replacement. The global
sorting proof is also the right installation permutation across records.

Integer sequences reach left chunks and right installation without a canonical
identity vector. String/composite sequences also use batch selection, encoding,
rank and duplicate proofs, then explicitly materialize descriptors at the
existing canonical ownership ledger/right-installer boundary. Their encoded
owners retain the original accepted-run boundaries, preserving frozen charges
and checkpoint bytes. That compatibility conversion remains linear in admitted
rows and is counted, rather than presented as descriptor-free admission.

## Costs and refusal paths

For `n` accepted rows, `k` distinct keys and `r` records, the batch descriptors
use O(n + k + r) entries. String dictionaries encode distinct values per record;
other supported keys use the existing Arrow canonical batch converter. Key
ranking compares O(k log k) canonical byte strings. Ordered records need a
linear order proof; unordered records/duplicate probes sort O(n log n) compact
references. Canonical sequence lookup additionally searches its accepted-run
directory. Resident overlap retains the existing index lookup cost per row.
Byte comparisons and Arrow copying also scale with their actual input bytes.

Test counters record accepted records/rows, ranked keys, sorted row indices,
canonical compatibility batches and materialized rows. Invalid nullable identity
schemas and unsupported key types fail through existing schema validation;
runtime nulls keep their existing validation priority. Nullable payloads remain
supported. Late/drop, empty and all-rejected inputs have explicit selections.

The unchanged identity reservation funds preparation before allocation; wide
key copies keep the existing separate reservation. Payload/worker/install
reservations precede their owned work. Left construction preserves the original
256-row unconditional and 4,096-row/128 KiB ordered inline bounds; other inputs
use the managed CPU worker and cancellation/refund lifecycle. Installation
remains behind the existing final cancellation and capacity checks, so no
fallible preparation publishes state or output. State layouts, retained-byte
formulas, checkpoint writers and restore codecs are unchanged.

## Validation

The focused baseline reproduction compiled and failed because three accepted,
unordered rows produced three canonical admission descriptors. New regressions
compare descriptors/chunk capacities, exact checkpoint metadata and segments,
restored continuation, full/projected/zero-column output, and an independent
composite-key ASOF oracle including tolerance and ties. Failure controls compare
legacy and batch duplicate priority, explicit workspace refusal, cancellation,
state/output atomicity and complete refunds. Canonical sequence controls compare
mixed-late accepted-run owners and restored checkpoint bytes.

The first complete ASOF module run passed 342 tests and exposed four parallel
admission regressions: the new columnar rows had bypassed the existing worker
route. The integration fix reuses that route through the identity iterator;
the original hook, append, refusal and cancellation assertions are preserved.
After that fix, all 68 admission tests pass with
`cargo test --locked -p calc-flow --lib operator::asof::admission:: -- --test-threads=1 --quiet`.
`cargo clippy --locked -p calc-flow --lib --tests -- -D warnings` passes;
formatting, generated-project-schema drift and whitespace checks also pass.
The final content also passes all eight property and three boundary tests with
`cargo test --locked -p calc-flow --test stream_asof_join_properties --test stream_asof_join_boundaries -- --test-threads=1 --quiet`.
The final admission rerun includes the ordered Boolean-key inline/refund control.
All Cargo checks used Rust 1.88.0, two build jobs and the existing debug cache.

No timing or throughput claim is made. Sealed performance acceptance, complete
regression, cross-platform and the Rust 90% coverage gate remain with their
existing dedicated task/CI.
