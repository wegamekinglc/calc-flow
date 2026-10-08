# Funded expected-schema comparison for streaming Join

## Scope and consumer

Normal managed V1 recovery optionally constructs private expected schemas on
the existing owned CPU service. The original metadata parser runs once; its
successful fixed result and the expected schemas feed the original checked
restore body. The real `StreamReader` compares its schema with a borrowed
expected schema at its existing location. The worker never captures caller
SchemaRef, FieldRef or timezone owners.

The writer, layout 1, semantic capability, fingerprint, public API, framing,
reader acceptance and terminal route retain their contracts. Generic decoded
resident ownership, V2 encoding and terminal dispatch remain separate design
and source gates. This slice provides no timing, whole IPC, task memory or RSS
certificate. It follows proposal `57877f93` and limited Design Review
`43f6e954`; the actual implementation still requires independent source review.

## Finite constructors and payment

Each side has at most 32 fields. Schema and every field metadata map must be
empty; only O(1) `is_empty` is inspected, including maps with deletion history.
Names and optional timestamp timezone text have at most 2,046 bytes. Supported
types are flat Null, Boolean, integer, float, UTF-8, binary, date and timestamp
scalars. All other shapes retain the original metadata-only route. These are
optional eligibility decisions, not new input errors.

The inventory in [schema/inventory.rs](../../../crates/calc-flow/src/operator/join/metadata_validation/schema/inventory.rs)
uses checked requested-allocation bounds. With `A(b,a)` the aligned Arc header
and payload, one side contributes:

`Input = n*sizeof(FieldPlan) + names + fresh timezone String bytes`

`Output = A(sizeof(Schema),align(Schema)) + n*sizeof(FieldRef)`
`+ A(n*sizeof(FieldRef),align(FieldRef)) + sum A(sizeof(Field),align(Field))`
`+ moved name bytes + sum A(timezone bytes,1)`.

Input additionally pays concrete SchemaConstruction, SchemaWork and
SchemaDecision sizes. Output additionally pays DescriptorFunding's Arc layout,
the concrete decision and the complete original registration/name inventory.
Both independent reservations use `new_empty` from the actual metadata
registration on the configured Join pool. They do not borrow wire-load credit.
The original metadata construction, caller and work inventories remain paid
separately. Existing admission funds native work/output boxes and cleanup
controls from actual work/output sizes and control_bytes.

Arrow 58.3 `Schema::new(Vec<FieldRef>)` and Rust 1.88 `Arc::from(Vec)` briefly
retain both pointer Vec and Arc slice. Timestamp String-to-Arc conversion
similarly overlaps, including the header for `Some("")`. Both are included.
Field names move into fresh Fields; destination maps are fresh empty maps.
Field equality retains Arrow's name/type/nullable/metadata semantics. No caller
String capacity or historical map capacity is treated as a constructor bound.

Before each field and bounded String copy, actual job cancellation is checked
and the caller yields. Each copy's source plus destination is at most 4 KiB;
two sides inspect at most 64 field headers. This describes the schema copy
boundary, not the later synchronous IPC decoder or whole restore callback.

## Ownership, refusal and recovery order

Partial plans precede their input credit and final output funding in
SchemaConstruction. Plans are installed in that carrier before yields.
SchemaWork carries data before input credit and output funding. Its successful
OwnedExpectedSchemas declares both schema owners before the funding Arc;
DescriptorFunding declares its reservation before retirement. Whole-carrier
clones retain the lease. Production consumers borrow only `&Schema`, and the
original reader constructs its own schema from wire bytes.

Submission takes the sole descriptor-funding Arc from the caller carrier.
The caller does not keep a duplicate resident owner after handing work to the
service. Escaped descriptors remain paid after work/caller refunds and keep
managed drain Pending until their last real owner drops. Existing outside-lock
attempt detach/drop breaks the temporary attempt-to-home retirement cycle.
No wait occurs while holding its own tracked WorkOutput credit: installation
consumes the output, then cleanup observes actual refund.

Original MetadataWork, Decision, Construction and SubmissionControl layouts and
their fee expressions are unchanged. Profile/input/output and pre-install
attempt fee refusals drop optional data and credit before selecting that exact
metadata-only route. Only one attempt is accepted and metadata parses once.
After an installed attempt's generation/healthy-home-close refusal, actual
cleanup precedes original checked Legacy recovery; no second worker is admitted.
Actual job cancellation/deadline retain their errors. The checked restore body
still performs its fresh stop check before the single final state assignment;
progress restoration and startup acknowledgement follow state restoration.

## Observed evidence and limits

The actual Managed startup test first failed at the original IPC comparison:
descriptor credit was absent. It now observes one native metadata parse, one
native schema constructor, two paid original comparisons and continued output.
A separate existing post-install-close control exposed an unused caller lease
of 1,234 bytes. Unique Option::take transfer fixed it without changing its pool
equality or refund assertion; the limited transfer review is `fe735beb`.

Four direct controls pass: requested constructor allocation and partial drop;
escaped output and abandoned attempt cleanup; exact metadata-only fee refusal;
original V1 bytes/errors and caller schema immutability. The existing actual
cancel control and five frozen V1 captures/continuation also pass. Requested
constructor peak is 1,254 bytes against independent input 1,572 plus output
1,610 bytes. This small allocator observation supplements the concrete bound;
it is not an all-profile measurement. Drain checks independently attribute home,
generation and attempt funding, require zero attempt funding after cleanup,
and require final pool zero after the actual remaining job owner is dropped.

Raw fixture failures are preserved: manually installed rows initially had no
pending checkpoint segments; `Some("")` retains the old IPC reader's timezone
normalization error; epoch 7 cannot be captured again after restoring epoch 7.
The final byte control compares original and managed restore at advancing
epoch 8. No production reader changes were made to accommodate those fixtures.

An initial zero-test old-cache run is not RED. An instant cached lint referring
to another worktree is not current-source lint evidence. Only the core lib/test
lint fingerprints were invalidated; dependencies and immutable release caches
were preserved. Final receipts bind actual compiler headers, dep-info, commands,
counts and file hashes. Local Lizard limits do not establish Codacy acceptance.
Full coverage, platform CI, Codacy and final specialist review remain CI/handoff
gates; no local benchmark or full matrix was run.
