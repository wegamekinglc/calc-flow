# Eligible ASOF right timestamp in the actual benchmark binaries

These are disassemblies of the sealed x86-64 release workers used for this
comparison. The compiler inlined candidate selection into
`StreamAsofJoinOperator::prepare_output`. The eight-column fixture has 512
matches on 32 keys, each eligible at the last right timestamp. The comparison
uses the current production lookup with the legacy benchmark switch disabled.

On the integrated baseline (binary `127f86c2…`), the right-run map traversal
begins at 0x64093d. The instruction at 0x6409ee compares the timestamp array
element against the left timestamp. Eligibility branches from 0x6409f2 to
0x640c05, which starts a second map traversal. It ultimately reads the selected
timestamp at 0x640c6e into RAX and uses it at 0x640c93 in the inclusive,
sign-extended tolerance comparison. See [baseline assembly](baseline.eligible-path.asm).

On the candidate (binary `a28683cf…`), one map traversal starts at 0x6408e7.
The load at 0x6409a6 puts the timestamp in RCX. The eligibility comparison at
0x6409aa and the tolerance comparison at 0x640c75 use this same value. The
taken branch at 0x6409af reaches 0x640c53 directly, bypassing the historical
candidate path. See [candidate assembly](candidate.eligible-path.asm).

The build therefore eliminates a second map traversal and timestamp load on
the eligible branch in this maintained worker. The callback still computes
the pending prefix, probes and materializes all matches, delivers and commits
its batch, and evicts retained state. The independent end-to-end callback
timings are in [the report](README.md). Earlier eligible candidates and
unmatched cases follow different paths. These excerpts do not measure cache
misses or time spent in individual instructions.

`inspect_code.py` extracts each binary's code-section fingerprint, function size,
and mnemonic sequence using GNU nm/objdump/objcopy. Baseline and candidate
`prepare_output` span 11,415 and 11,305 bytes. The registered protocol,
both exact Cargo identities and all measured/qualification process records
are included in this directory.
