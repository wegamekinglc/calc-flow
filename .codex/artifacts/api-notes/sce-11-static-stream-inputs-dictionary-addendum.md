# Immutable Static Stream Inputs - Dictionary Contract Addendum

Artifact slug: `sce-11-static-stream-inputs-dictionary-addendum`.

## Control and Scope

- Baseline: PR #208 head
  `7690ef7c76cd14996554771444d72d9ab3429fb2`.
- This addendum extends only the dictionary projection of
  `.codex/artifacts/api-notes/sce-11-static-stream-inputs.md`, whose frozen
  SHA-256 remains
  `da9a9ec1687f8e1487d6b0341b838c69d1d07a54992cf3ee4b59b77b451c85d2`.
- The byte grammar and type tags in §7 of
  `.codex/artifacts/api-notes/symbolic-computation-engine.md` remain the
  authority. This addendum makes tag `0x54` reachable through the shared
  strict `ArrowFieldSpec.data_type` vocabulary.
- Status: **frozen for the PR #208 dictionary correction**. There are no open
  dictionary design questions. Timestamp and semantic-fingerprint behavior
  are explicitly out of scope and retain their already frozen contracts.

## Audiences

- Rust users: one additive canonical `ArrowFieldSpec.data_type` spelling; no
  new exported type or function.
- Python users: the same spelling in the frozen `ArrowFieldSpec` value and
  project dictionaries; no constructor or type-stub change.
- Studio clients: the same string inside existing project-v3 JSON; no route,
  JSON field, OpenAPI shape, or generated TypeScript shape changes.

## Surface Today

Digest v1 already assigns `0x54` to dictionary descriptors and resolves cells
to logical values, but the shared strict parser and the static-input
`canonical_type_string` seam accept only non-dictionary spellings. A project
can therefore describe a dictionary only to a private digest helper, not to
the real compile/preflight path. The cell writer also reads the physical key
before honoring the dictionary array's null bitmap.

## Frozen Additive Surface

`ArrowFieldSpec.data_type` gains exactly this grammar:

```text
DICTIONARY = "dictionary<index=" INDEX ";value=" VALUE ";ordered=" ORDERED ">"

INDEX = "int8" | "int16" | "int32" | "int64"
      | "uint8" | "uint16" | "uint32" | "uint64"

VALUE = "bool"
      | INDEX
      | "float32" | "float64"
      | "string" | "large_string"
      | "date32" | "date64"
      | "time32[s]" | "time64[us]"
      | "timestamp[ms]" | "timestamp[us]" | "timestamp[us, UTC]"

ORDERED = "false" | "true"
```

Canonical examples are:

```text
dictionary<index=int32;value=string;ordered=false>
dictionary<index=uint8;value=timestamp[us, UTC];ordered=true>
```

The parser consumes the entire string. Case, key order, punctuation, and the
three key names are exact. There is no whitespace inside the wrapper; the one
space in the already frozen `timestamp[us, UTC]` token is retained. The
`ordered` component is required and has no default. Nested dictionaries,
non-integer indices, unsupported value types, unknown components, duplicate
components, aliases, and Arrow/Rust/PyArrow display strings are rejected.

All existing non-dictionary spellings and meanings are unchanged. In
particular, this addendum neither accepts a timezone-bearing millisecond
timestamp nor adds a timestamp tag.

Because `ArrowFieldSpec` is shared, the strict parser accepts this spelling in
every existing `ArrowFieldSpec` location. This addendum mandates dictionary
runtime behavior only for SCE-11 table static inputs; existing operator- and
connector-specific type restrictions still apply after parsing.

## Strict Parse and Canonical Projection

- The shared parse result must retain both the Arrow `DataType::Dictionary`
  and its required `ordered` bit. Creating an Arrow `Field` installs that bit
  as the field's dictionary-order flag; it must not be discarded after
  producing the `DataType`.
- The private `canonical_type_string` seam projects a complete Arrow `Field`
  (or an equivalent `(DataType, dictionary_ordered)` pair), not a bare
  `DataType`. For a supported dictionary it emits the exact grammar above.
- `parse(canonical_type_string(field))` must recreate the field's dictionary
  index type, value type, and ordered bit. Conversely, every accepted
  dictionary string must round-trip byte-for-byte through canonical
  projection.
- The schema `Field::dict_is_ordered()` value is the digest descriptor's
  ordered byte (`false -> 0x00`, `true -> 0x01`). A physical dictionary
  array's ID, values order, indices, unused values, or first-chunk layout must
  not replace this schema identity.

No new public parser or wrapper type is introduced. One internal strict parse
seam must serve ports, connector schemas, static declarations, and Arrow field
construction so accepted strings cannot diverge by caller.

## Preflight and Digest Requirements

For a table static input, compile validates the declaration through the shared
strict parser. Runner preflight then compares field name, parsed type,
dictionary ordered bit, and nullability against the supplied table's exact
Arrow field before any digest is computed or any source opens.

After that match:

1. Digest schema encoding writes `0x54 || TYPE(index) || TYPE(value) ||
   ordered-byte` from the validated Arrow field.
2. Cell encoding checks the outer dictionary slot's null bitmap first. A null
   key writes exactly `0x30` and never reads or resolves the hidden key value.
3. A non-null key resolves to the logical dictionary value. A null resolved
   value also writes `0x30`; otherwise the value uses its frozen non-dictionary
   `0x31 || SCALAR` rule.
4. Dictionary IDs, values order, physical indices, unused dictionary values,
   and record-batch chunk boundaries remain absent from the digest.

Invalid or unresolvable dictionary storage fails closed. Errors may identify a
field and row coordinate but must not include the physical key, dictionary
contents, or logical payload.

## Error Cases

Paths, validation codes, and failure timing below are normative; prose after
the path may vary while still naming the violated constraint.

| Input violation                                      | Required path / code or message shape                                               | Failure point                    |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------- |
| malformed or non-canonical declaration              | `static_inputs[i].schema[j].data_type`, `unsupported_arrow_type`                    | project validation / compilation |
| unsupported index, value, or nested dictionary      | `static_inputs[i].schema[j].data_type`, `unsupported_arrow_type`                    | project validation / compilation |
| actual field has no canonical digest-v1 spelling    | `static_inputs.{name}.schema[j].data_type: Arrow type ... has no strict spelling`   | runner preflight                 |
| declared and actual type or ordered bit differ       | `static_inputs.{name}.schema[j].data_type: declared ... but the value has ...`      | runner preflight                 |
| dictionary storage cannot resolve a non-null key     | `static_inputs.{name}.schema[j].data_type: dictionary value cannot be resolved ...` | digest, before source open       |

A null dictionary key is a value case, not an error. The runtime error path
must contain one `static_inputs.{name}` root and must not repeat it during
qualification.

## Golden and Equivalence Requirements

The required golden uses input name `weights` and one nullable field named
`color` with type
`dictionary<index=int8;value=string;ordered=false>`. Its logical rows are
`["red", null, "blue", "red"]`. Both of these physical layouts must produce
the exact digest below:

```text
layout A: values=["red", "blue"], keys=[0, null, 1, 0]
layout B: values=["blue", "unused", "red"], keys=[2, null, 0, 2]
sha256:   d1f3b0c589c58b966d863243efffcf5e314e558016a19f8d4dfdbf45f629072e
```

The golden must be exercised through the production table digest path, not by
duplicating the implementation under test. The acceptance set also requires:

- identical results when the same logical rows are split across different
  record-batch boundaries;
- identical results for null-key buffers carrying different hidden physical
  index bytes, proving the null bitmap is checked first;
- different digests when index type, value type, or ordered bit differs;
- a dictionary descriptor and its plain value type to remain distinct even
  when all logical cells are equal;
- table-driven strict parse/canonical round-trips for all 8 index types, all
  20 frozen non-dictionary value types, and both ordered values;
- an end-to-end compiled-project/runner test proving the canonical spelling
  passes static-input preflight, plus precise-path rejection tests for an
  alias, a nested dictionary, and an ordered-bit mismatch.

## Example

```json
{
  "kind": "table",
  "name": "weights",
  "mutability": "static",
  "schema": [
    {
      "name": "color",
      "data_type": "dictionary<index=int8;value=string;ordered=false>",
      "nullable": true
    }
  ]
}
```

## Compatibility Conclusion

This is an additive semantic extension to the shared
`ArrowFieldSpec.data_type` vocabulary. Existing signatures, scalar strings,
project-v3 documents, canonical JSON, JSON Schema/OpenAPI shapes, fingerprint
algorithm, and digest tags remain unchanged. A new document using the
dictionary spelling is intentionally not readable by an older strict parser
that predates this addendum; existing documents retain their bytes,
fingerprints, and behavior.

The original frozen SCE-11 note is not amended in place. The separate
fingerprint and timestamp findings from PR #208 review must be fixed against
their existing contracts and are not authorized or modified by this addendum.
