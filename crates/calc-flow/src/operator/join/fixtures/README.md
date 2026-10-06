# Join V1 checkpoint fixture

`checkpoint-v1.json` is captured from the pre-J1 Join implementation at
`9b1535bc`. It freezes operator inline metadata and the exact bytes of every
segment across five epochs. The scenario retains out-of-order rows, removes
a non-tail row, appends another row, then removes two more rows together before
base compaction. This preserves evidence of tombstone ordering as well as
canonical base encoding and continuation.

The fixture is historical compatibility evidence. Do not regenerate it to
accommodate a changed encoder. J2 readers must continue to restore these bytes;
new layout fixtures belong in separate files. Managed manifest migration
evidence is recorded separately from this operator snapshot fixture.
