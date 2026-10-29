"""Lower symbolic declarations to strict project-v3 native graphs.

Row-local expressions fuse into segments; SQL, temporal, join and matrix
boundaries retain their native operators. Rust performs final graph validation.
Lowering accepts declarations only and does not run while a plan executes.
"""

from calc_flow.symbolic.lower.program import (
    compile_program_batch,
    compile_program_stream,
    lower_program_document,
)

__all__ = [
    "compile_program_batch",
    "compile_program_stream",
    "lower_program_document",
]
