"""Deterministic lowering of symbolic programs to strict project-v3.

The lowerer resolves each declared table output into one fused row-local
segment, renders the segment as DataFusion SQL inside strict project-v3
``expression`` nodes, and hands the document to the existing Rust graph
compiler for final port, schema, topology, and fingerprint validation. No data
object, source, sink, or runner is accepted here, and no symbolic Python runs
while a compiled plan executes.
"""

from calc_flow.symbolic.lower import planners, program, segments, strategies
from calc_flow.symbolic.lower.program import (
    compile_program_batch,
    compile_program_stream,
    lower_program_document,
)

# The package preserves the flat ``symbolic/lower.py`` namespace: every
# module-level name of the split modules stays importable from here, including
# the private helpers the lowering tests exercise.
for _module in (segments, planners, strategies, program):
    for _name in dir(_module):
        if not _name.startswith("__"):
            globals().setdefault(_name, getattr(_module, _name))
del _module, _name

__all__ = [
    "compile_program_batch",
    "compile_program_stream",
    "lower_program_document",
]
