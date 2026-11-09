use crate::{CalcFlowError, Result};
use std::alloc::Layout;

#[derive(Default)]
struct Statistics {
    arrays: usize,
    objects: usize,
    separators: usize,
    escaped_bytes: usize,
    largest_string: usize,
    largest_escaped: usize,
}

pub(super) fn requirement(bytes: &[u8], check: &mut dyn FnMut() -> Result<()>) -> Result<u64> {
    let statistics = scan(bytes, check)?;
    let vectors = vector_bytes(&statistics)?;
    let strings = string_bytes(&statistics)?;
    let errors = error_bytes(statistics.largest_string)?;
    check()?;
    sum_u64(vectors, strings, errors)
}
fn sum_u64(first: usize, second: usize, third: usize) -> Result<u64> {
    u64::try_from(add(add(first, second)?, third)?).map_err(|_| invalid())
}
fn vector_bytes(counts: &Statistics) -> Result<usize> {
    let arrays = multiply(counts.arrays, 6)?;
    let objects = multiply(counts.objects, 12)?;
    let children = multiply(counts.separators, 3)?;
    let slots = add(add(arrays, objects)?, children)?;
    multiply(slots, content_layout()?.size())
}
fn string_bytes(counts: &Statistics) -> Result<usize> {
    add(counts.escaped_bytes, growing_bytes(counts.largest_escaped)?)
}
fn content_layout() -> Result<Layout> {
    Layout::new::<u64>()
        .extend(Layout::new::<[usize; 3]>())
        .map(|(layout, _)| layout.pad_to_align())
        .map_err(|_| invalid())
}

fn scan(bytes: &[u8], check: &mut dyn FnMut() -> Result<()>) -> Result<Statistics> {
    let mut statistics = Statistics::default();
    let mut frames = [0_usize; 128];
    let mut depth = 0;
    let mut index = 0;
    while let Some(&byte) = bytes.get(index) {
        check()?;
        let next = step(bytes, index, byte, &mut statistics, check)?;
        if !control(byte, &mut frames, &mut depth, &mut statistics)? {
            break;
        }
        index = next;
    }
    remaining(&frames[..depth])?;
    Ok(statistics)
}

fn remaining(frames: &[usize]) -> Result<()> {
    frames.iter().try_for_each(|count| allocation(*count))
}

fn step(
    bytes: &[u8],
    index: usize,
    byte: u8,
    counts: &mut Statistics,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<usize> {
    let next = add(index, 1)?;
    if byte == b'"' {
        add(string(bytes, next, counts, check)?, 1)
    } else {
        Ok(next)
    }
}

fn control(
    byte: u8,
    frames: &mut [usize; 128],
    depth: &mut usize,
    counts: &mut Statistics,
) -> Result<bool> {
    match byte {
        b'[' | b'{' => open(byte, frames, depth, counts),
        b']' | b'}' => close(frames, depth),
        b',' | b':' => {
            counts.separators = add(counts.separators, 1)?;
            separator(frames, *depth)?;
            Ok(true)
        }
        _ => Ok(true),
    }
}

fn close(frames: &[usize; 128], depth: &mut usize) -> Result<bool> {
    if *depth > 0 {
        *depth -= 1;
        allocation(frames[*depth])?;
    }
    Ok(true)
}

fn separator(frames: &mut [usize; 128], depth: usize) -> Result<()> {
    if depth > 0 {
        frames[depth - 1] = add(frames[depth - 1], 1)?;
    }
    Ok(())
}

fn open(
    byte: u8,
    frames: &mut [usize; 128],
    depth: &mut usize,
    counts: &mut Statistics,
) -> Result<bool> {
    // serde_json 1.0.150 rejects the next container at its fixed recursion limit.
    if *depth == frames.len() {
        return Ok(false);
    }
    frames[*depth] = 0;
    *depth += 1;
    if byte == b'[' {
        counts.arrays = add(counts.arrays, 1)?;
    } else {
        counts.objects = add(counts.objects, 1)?;
    }
    Ok(true)
}

fn string(
    bytes: &[u8],
    start: usize,
    counts: &mut Statistics,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<usize> {
    let mut index = start;
    let mut escaped = false;
    while let Some(&byte) = bytes.get(index) {
        check()?;
        if byte == b'"' {
            break;
        }
        escaped |= byte == b'\\';
        index = advance(index, byte)?;
    }
    let length = index.saturating_sub(start);
    counts.largest_string = counts.largest_string.max(length);
    escaped_string(counts, length, escaped)?;
    Ok(index)
}

fn advance(index: usize, byte: u8) -> Result<usize> {
    add(index, 1 + usize::from(byte == b'\\'))
}

fn escaped_string(counts: &mut Statistics, length: usize, escaped: bool) -> Result<()> {
    if escaped {
        counts.escaped_bytes = add(counts.escaped_bytes, length)?;
        counts.largest_escaped = counts.largest_escaped.max(length);
    }
    Ok(())
}
fn allocation(separators: usize) -> Result<()> {
    let capacity = add(multiply(add(separators, 1)?, 2)?, 4)?;
    content_allocation(capacity)
}
fn content_allocation(capacity: usize) -> Result<()> {
    let layout = content_layout()?;
    let pair_bytes = multiply(layout.size(), 2)?;
    let bytes = multiply(capacity, pair_bytes)?;
    Layout::from_size_align(bytes, layout.align()).map_err(|_| invalid())?;
    Ok(())
}

fn growing_bytes(length: usize) -> Result<usize> {
    if length == 0 {
        return Ok(0);
    }
    let capacity = multiply(length.max(8), 2)?;
    Layout::array::<u8>(capacity).map_err(|_| invalid())?;
    multiply(length.max(8), 3)
}

fn error_bytes(largest_string: usize) -> Result<usize> {
    let fixed = concat!(
        "invalid cumulative summary: invalid type: , expected  at line  column ",
        "unknown variant , expected one of Signed SignedTotal Unsigned UnsignedTotal Float ",
        "unknown field , expected one of valid_count value origin_bits mean_bits m2_bits has_infinity ",
        "duplicate field missing field floating point integer sequence map string boolean null ",
        "invalid value: invalid length: byte array tuple struct NumericValue CumulativeAccumulator"
    ).len();
    let message = add(
        add(fixed, multiply(usize::BITS as usize, 2)?)?,
        multiply(largest_string, 6)?,
    )?;
    let message_peak = growing_bytes(message)?;
    add(message_peak, error_layout()?)
}
fn error_layout() -> Result<usize> {
    let boxed = Layout::new::<[usize; 3]>()
        .extend(Layout::new::<u64>())
        .map_err(|_| invalid())?
        .0
        .extend(Layout::new::<[usize; 2]>())
        .map_err(|_| invalid())?
        .0
        .pad_to_align()
        .size();
    Ok(boxed)
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or_else(invalid)
}
fn multiply(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(invalid)
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling restore summary geometry is invalid".into(),
    }
}

#[cfg(test)]
#[path = "json_tests.rs"]
mod tests;
