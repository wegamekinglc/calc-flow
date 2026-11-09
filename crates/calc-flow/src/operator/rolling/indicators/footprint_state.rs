use std::sync::Arc;
use std::{alloc::Layout, mem::size_of, sync::atomic::AtomicUsize};

use datafusion::common::ScalarValue;

use super::{
    super::{
        BTreeMap, BufferedRow, EntityRollingState, KeyValue, Result, RollingStreamState,
        RowIdentity, WindowState, budget::StateCharge, internal_error,
    },
    footprint::Footprint,
};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(super) fn state(state: &RollingStreamState, node: &str) -> Result<(StateCharge, Footprint)> {
    state_checked(state, node, &mut || Ok(()))
}

pub(super) fn state_checked(
    state: &RollingStreamState,
    node: &str,
    check: Check<'_>,
) -> Result<(StateCharge, Footprint)> {
    check()?;
    validate_general(state)?;
    let (buffer, mut footprint) = buffer_checked(&state.buffer, node, check)?;
    wrapper(&mut footprint, state, node, check)?;
    let rows = history_footprint_total(&mut footprint, &state.histories, node, check)?;
    finish_charge(buffer.rows, rows, footprint, node)
}

fn validate_general(state: &RollingStreamState) -> Result<()> {
    if !state.ordered.is_empty() || state.typed_kernel_state.is_some() {
        return Err(internal_error(
            "indicator general state contains typed storage",
        ));
    }
    Ok(())
}

fn finish_charge(
    buffer: u64,
    history: u64,
    footprint: Footprint,
    node: &str,
) -> Result<(StateCharge, Footprint)> {
    Ok((
        StateCharge {
            rows: add_rows(buffer, history, node)?,
            bytes: footprint.bytes(node)?,
        },
        footprint,
    ))
}

pub(super) fn packet_allocation_bytes(node: &str) -> Result<u64> {
    let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<RollingStreamState>())
        .map_err(|_| overflow(node))?;
    size(layout.pad_to_align().size(), node)
}

pub(super) fn identity_bytes(graph: &str, node: &str) -> Result<u64> {
    size(graph.len(), node)?
        .checked_add(size(node.len(), node)?)
        .ok_or_else(|| overflow(node))
}

fn wrapper(
    footprint: &mut Footprint,
    state: &RollingStreamState,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    footprint.add_owned(packet_allocation_bytes(node)?, node)?;
    for value in [
        state.pipeline_fingerprint.as_ref(),
        state.operator_id.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        check()?;
        footprint.add_owned(size(value.capacity(), node)?, node)?;
    }
    Ok(())
}

pub(super) fn prepare_checked(
    mut state: RollingStreamState,
    node: &str,
    check: Check<'_>,
) -> Result<(Arc<RollingStreamState>, Footprint)> {
    let (charge, mut footprint) = state_checked(&state, node, check)?;
    check()?;
    state.charge = charge;
    let state = Arc::new(state);
    footprint.pin_state(Arc::clone(&state));
    check()?;
    Ok((state, footprint))
}

#[cfg(test)]
pub(super) fn packet(
    state: Arc<RollingStreamState>,
    node: &str,
) -> Result<(StateCharge, Footprint)> {
    packet_checked(state, node, &mut || Ok(()))
}

#[cfg(test)]
pub(super) fn packet_checked(
    state: Arc<RollingStreamState>,
    node: &str,
    check: Check<'_>,
) -> Result<(StateCharge, Footprint)> {
    let (charge, mut footprint) = state_checked(&state, node, check)?;
    footprint.pin_state(state);
    check()?;
    Ok((charge, footprint))
}

fn history_footprint_total(
    footprint: &mut Footprint,
    histories: &super::super::RollingHistories,
    node: &str,
    check: Check<'_>,
) -> Result<u64> {
    let mut rows = 0;
    for (key, entity) in &histories.by_entity {
        check()?;
        entity_footprint(footprint, key, entity, node, check)?;
        rows = add_rows(rows, 1, node)?;
        rows = add_rows(rows, size(entity.rows.len(), node)?, node)?;
    }
    Ok(rows)
}

fn buffer_checked(
    rows: &BTreeMap<RowIdentity, BufferedRow>,
    node: &str,
    check: Check<'_>,
) -> Result<(StateCharge, Footprint)> {
    let mut footprint = Footprint::default();
    for (key, row) in rows {
        check()?;
        row_footprint(&mut footprint, key, row, node, check)?;
    }
    Ok((
        StateCharge {
            rows: size(rows.len(), node)?,
            bytes: footprint.bytes(node)?,
        },
        footprint,
    ))
}

fn row_footprint(
    footprint: &mut Footprint,
    key: &RowIdentity,
    row: &BufferedRow,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    footprint.add_owned(256, node)?;
    identity(footprint, key, node, check)?;
    identity(footprint, &row.identity, node, check)?;
    values(footprint, &row.values, node, check)
}

fn identity(
    footprint: &mut Footprint,
    key: &RowIdentity,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    keys(
        footprint,
        key.entity.iter().filter_map(Option::as_ref),
        node,
        check,
    )?;
    keys(footprint, key.sequence.iter(), node, check)
}

fn keys<'a>(
    footprint: &mut Footprint,
    keys: impl Iterator<Item = &'a KeyValue>,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    for key in keys {
        check()?;
        if let KeyValue::String(value) = key {
            footprint.add_owned(size(value.capacity(), node)?, node)?;
        }
    }
    Ok(())
}

fn entity_footprint(
    footprint: &mut Footprint,
    key: &[Option<KeyValue>],
    entity: &EntityRollingState,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    if !entity.columnar.records.is_empty() {
        return Err(internal_error(
            "indicator general history contains columnar storage",
        ));
    }
    entity_base(footprint, key, entity, node, check)?;
    for indicator in &entity.indicators {
        check()?;
        footprint.add_owned(indicator.logical_bytes(), node)?;
    }
    history_footprint(footprint, entity, node, check)
}

fn entity_base(
    footprint: &mut Footprint,
    key: &[Option<KeyValue>],
    entity: &EntityRollingState,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    footprint.add_owned(256, node)?;
    keys(
        footprint,
        key.iter().filter_map(Option::as_ref),
        node,
        check,
    )?;
    footprint.add_owned(
        product(entity.windows.len(), size_of::<WindowState>(), node)?,
        node,
    )
}

fn history_footprint(
    footprint: &mut Footprint,
    entity: &EntityRollingState,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    for row in &entity.rows {
        check()?;
        values(footprint, row, node, check)?;
    }
    for window in &entity.windows {
        check()?;
        extrema(footprint, window, node, check)?;
    }
    Ok(())
}

fn extrema(
    footprint: &mut Footprint,
    window: &WindowState,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    let WindowState::Extrema(extrema) = window else {
        return Ok(());
    };
    for (key, value) in &extrema.queue {
        check()?;
        keys(footprint, key.sequence.iter(), node, check)?;
        footprint.scalar_checked(value, node, check)?;
    }
    Ok(())
}

fn values(
    footprint: &mut Footprint,
    values: &Vec<ScalarValue>,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    let spare = values
        .capacity()
        .checked_sub(values.len())
        .expect("Vec len fits capacity");
    footprint.add_owned(product(spare, size_of::<ScalarValue>(), node)?, node)?;
    for value in values {
        check()?;
        footprint.scalar_checked(value, node, check)?;
    }
    Ok(())
}

fn product(count: usize, width: usize, node: &str) -> Result<u64> {
    let bytes = count.checked_mul(width).ok_or_else(|| overflow(node))?;
    size(bytes, node)
}
fn size(value: usize, node: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| overflow(node))
}
fn add_rows(left: u64, right: u64, node: &str) -> Result<u64> {
    left.checked_add(right).ok_or_else(|| overflow(node))
}
fn overflow(node: &str) -> crate::CalcFlowError {
    super::super::operator_error(node, "rolling participating state charge overflowed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_accounts_for_its_real_arc_and_owned_identity_capacity_once() {
        let mut stored = None;
        let measured = allocation_counter::measure(|| {
            let mut graph = String::with_capacity(512);
            graph.push_str("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            let mut node = String::with_capacity(256);
            node.push_str("actual_node");
            stored = Some(Arc::new(RollingStreamState {
                pipeline_fingerprint: Some(graph),
                operator_id: Some(node),
                ..RollingStreamState::default()
            }));
        });
        let state = stored.unwrap();
        let (charge, mut footprint) = packet(Arc::clone(&state), "actual_node").unwrap();
        let (_, alias) = packet(Arc::clone(&state), "actual_node").unwrap();
        let retained = u64::try_from(measured.bytes_current).unwrap();
        assert!(retained > 768);
        assert_eq!(charge.rows, 0);
        assert_eq!(charge.bytes, retained);
        footprint.merge(&alias, "actual_node").unwrap();
        assert_eq!(footprint.bytes("actual_node").unwrap(), retained);
    }

    #[test]
    fn test_checked_packet_cancellation_precedes_its_arc_allocation() {
        let state = RollingStreamState::default();
        let mut cancellation = Some(crate::CalcFlowError::Cancelled {
            run_id: "packet".into(),
        });
        let mut result = None;
        let measured = allocation_counter::measure(|| {
            result = Some(prepare_checked(state, "actual_node", &mut || {
                Err(cancellation.take().unwrap())
            }));
        });
        assert!(
            matches!(result.unwrap(), Err(crate::CalcFlowError::Cancelled { run_id }) if run_id == "packet")
        );
        assert_eq!(measured.bytes_total, 0);
    }
}
