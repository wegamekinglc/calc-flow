# Streaming Join managed Local wire preload

## Scope

This tranche prepays selected, nonempty streaming Join checkpoint wire loads
through the existing configured Join runtime pool. The private reader is the
same concrete Local lineage instance used by the manifest transaction. The
ordinary managed recovery path uses it; the terminal recovery path is unchanged.
Join layout 1, semantic capability 1, fingerprints, checkpoint bytes, decoded
row state and public backend traits retain their contracts.

Custom backends, legacy/test storage parts and unproved Windows path prefixes
keep the original loader. Those branches have no new physical funding claim.
V2 encoding, generic decoded resident funding and descriptor construction need
their later design gates. The untracked future V2 specification is a reference
and is excluded from this change.

The wire-only design checkpoint is recorded in
`target/issue363-j2b-owned-work-mapping-review.md`, SHA-256
`c3945b8bccb5347b367125f10c0862c7077fb9110d63074dcbef2e69a4dc2173`.
Its subsequent source review found missing Local I/O controls; the concrete
private Local route below addresses that finding and still requires final
source approval.

## Ownership and cancellation

The runner initializes only selected, nonempty Join runtimes before forming
its immutable capability lookup. Reservation uses that runtime's existing
pool, including resources configured before launch. Reset preserves it.

The managed factory opens the lineage once. Its concrete `Arc` is cloned and
coerced for the transaction's backend trait; the private reader keeps the same
instance, root, identity and publication lock. The transaction preserves its
operation lock and `owner_settled` cancellation behavior.

The owned request is constructed after its checked reservation. `FundedLoad`
stores its boxed owned future before the final credit field. Busy/closed load
admission, unpolled future destruction, partial errors and cancelled observers
therefore destroy request data before returning that credit. The existing
`LoadOwner` loan and drain own the real task; no new cleanup service is added.

The private Local reader uses the existing blocking filesystem worker.
`PaidRead` stores paths and the cloned handle before credit. Its result is
`PaidBytes`, with the actual `Vec<u8>` before credit. The blocking task's
unobserved successful result therefore carries its own lease. Conversion into
`StateSegment::with_owner` transfers the lease with the same bytes. Segment
clones retain it until the last owner is destroyed. Draining the load service
does not promise that an escaped successful snapshot has released its wire.

The production route never uses naked `StateSegment::bytes_arc`. The tests use
weak byte references only to observe final destruction. Error results preserve
the existing diagnostics; an error with no wire payload has no claim of an
escaped payload lease. Cancellation waits for actual I/O settlement before
partial data and credit are destroyed.

## Constructor inventory

| Component                         | Bound source                                                                      |
|-----------------------------------|-----------------------------------------------------------------------------------|
| Wire buffers                      | Validated handle byte lengths; Local allocates an exact-size `Vec` and reads it.  |
| Request and snapshot controls     | Owned handle/string lengths, two metadata clones and one fresh segment map.       |
| Metadata and segment map nodes    | Rust 1.88 BTree maximum internal-node geometry; metadata includes an empty root.  |
| JSON arrays and strings           | Clone constructors use element count and string length, without caller capacity.  |
| Future, reservation and tasks     | Concrete future sizes, DataFusion registration and pinned Tokio 1.52.3 layouts.   |
| Local paths                       | Four paths plus two concurrent clone/reallocation slots, from actual root length. |
| Local hashes and formatted text   | SHA-256 widths and checked constructor growth/reallocation bounds.                |
| Windows path conversion           | Certified verbatim roots only; two `len + 1` UTF-16 temporary buffers.            |
| Blocking input and result control | Concrete `PaidRead`, `PaidBytes`, closure and task-cell geometry.                 |

All lengths are checked before request/path/worker construction. The original
registration bootstrap uses the existing Join reservation API; subsequent
owned clones, boxes, bytes and filesystem task controls are included in its
credit. This is a retained allocation bound, not a whole-process RSS ceiling.
Existing Tokio runtime/thread-pool infrastructure is outside the new per-read
constructor inventory.

Path slots cover `PathBuf` clone/growth and, after construction, Unix syscall
path-string temporaries; they are reused between these phases. Windows
canonical verbatim roots avoid the unbounded full-path conversion branch.
Unknown prefixes use the original loader instead. Explicit map insertion
avoids the temporary vector and sorting workspace of `BTreeMap::collect`.

## Observed checks

The source base is `acd956739104732baa7905cb03a226209ca29583`.
Logs are in the isolated worktree's `target/issue363-j2b-preload-v1/`.

| Check                         | Actual result and scope                                                                  |
|-------------------------------|------------------------------------------------------------------------------------------|
| Initial wire RED              | One failure: actual 4,096-byte allocation had zero Join preload credit.                  |
| Long Local root RED           | One failure: 4,096 wire + 23,745 actual backend peak exceeded the old 12,983 inventory.  |
| Paid Local focused module     | Ten passed; certified private Local read, final owner, cancellation, errors and refusal. |
| Blocking-result lease control | One passed; existing worker's result keeps wire credit until the last segment drop.      |
| Uncertified fallback control  | One passed; original loader preserves bytes/metadata and does not initialize Join.       |

The original two RED logs remain immutable. The first RED used the generic
loader; the final fixtures explicitly use the certified private Local route.
The long-root GREEN measures actual path construction allocation in that route
and separately observes the blocking worker's wire capacity. The synthetic
wire fixture is opaque data, not a claim of generic Join decoded funding.

The ordinary shared debug cache initially returned stale test/check artifacts.
Zero-test discovery and cached check results are excluded. Only the mismatched
calc-flow fingerprint markers were backed up and invalidated; dependency and
sealed release caches were preserved. Actual test compilation and Clippy logs
identify this isolated worktree. The final receipt records exact commands,
results, source hashes and necessary static checks.

No benchmark, complete coverage/workspace run or local Windows test was run.
Required cross-platform and coverage gates remain CI responsibilities.
