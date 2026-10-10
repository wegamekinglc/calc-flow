//! Paid construction of one canonical flat output chunk.

mod bounds;
mod control;
mod funding;
mod inputs;
mod process;
mod worker;

const PAIRS_PER_UNIT: usize = 4_096;

fn units(rows: usize) -> usize {
    (rows / PAIRS_PER_UNIT).min(8)
}

fn range(rows: usize, ordinal: usize) -> std::ops::Range<usize> {
    let count = units(rows);
    let width = rows / count;
    let extra = rows % count;
    let start = ordinal * width + ordinal.min(extra);
    start..start + width + usize::from(ordinal < extra)
}
