use std::{
    collections::VecDeque,
    mem::{align_of, size_of},
};

use crate::{CalcFlowError, Result};

pub(super) fn plan<K>(rows: usize, entities: usize, retains_rows: bool, node: &str) -> Result<u64> {
    let tree = tree_nodes::<K, VecDeque<i64>>(entities, node)?;
    if !retains_rows || rows == 0 {
        return Ok(tree);
    }
    // Each deque's final capacity is <= max(2*pushes,4). One old allocation
    // may overlap a doubling; it is <= max(total pushes,2).
    let slots = rows
        .checked_mul(3)
        .and_then(|slots| {
            entities
                .checked_mul(4)
                .and_then(|minimum| slots.checked_add(minimum))
        })
        .and_then(|slots| slots.checked_add(2))
        .ok_or_else(|| invalid(node))?;
    let tails = slots
        .checked_mul(size_of::<i64>())
        .ok_or_else(|| invalid(node))?;
    tree.checked_add(u64::try_from(tails).map_err(|_| invalid(node))?)
        .ok_or_else(|| invalid(node))
}
fn tree_nodes<K, V>(count: usize, node: &str) -> Result<u64> {
    // Pinned std B=6: 11 key/value slots, parent + 12 edges, two u16s.
    // One internal-sized node per entry also covers splitting and alignment.
    let slots = size_of::<K>()
        .checked_add(size_of::<V>())
        .and_then(|width| width.checked_mul(11))
        .ok_or_else(|| invalid(node))?;
    let fixed = size_of::<usize>() * 14 + size_of::<u16>() * 2;
    let padding = align_of::<K>() + align_of::<V>() + align_of::<usize>();
    let bytes = slots
        .checked_add(fixed)
        .and_then(|bytes| bytes.checked_add(padding))
        .and_then(|bytes| bytes.checked_mul(count))
        .ok_or_else(|| invalid(node))?;
    u64::try_from(bytes).map_err(|_| invalid(node))
}
fn invalid(node: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node.to_owned(),
        message: "rolling retained row prefix workspace size overflowed".into(),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

pub(super) fn tick(position: &mut usize, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    if *position == 4_096 {
        check()?;
        *position = 0;
    }
    *position += 1;
    Ok(())
}
