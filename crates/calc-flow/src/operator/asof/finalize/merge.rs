//! Merge normalized left positions against one latest payload slot per key.

use super::{
    OutputPlanBuilder, StreamAsofJoinOperator, check_match_progress, left_run_source, state,
};
use crate::{Result, StreamOperatorContext};
use datafusion::execution::memory_pool::MemoryReservation;

#[cfg(test)]
thread_local! {
    static PAYLOAD_RESOLUTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn take_payload_resolutions() -> usize {
    PAYLOAD_RESOLUTIONS.replace(0)
}

struct Route<'a> {
    bucket: &'a state::RightBucket,
    cursor: state::RightCursor<'a>,
    latest: Option<(state::RowRef, usize)>,
}

const _: () = assert!(size_of::<Option<Route<'static>>>() <= 192);
const _: () = assert!(
    2 * (size_of::<(
        state::Encoding,
        (&state::RightBucket, state::RightCursor<'_>)
    )>() + 1)
        + 32
        <= 320
);

fn routes_for_keys<'a>(
    keys: &[Option<state::Encoding>],
    routes: &mut [Option<Route<'a>>],
    time: i64,
    operator: &'a StreamAsofJoinOperator,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    for (id, key) in keys.iter().enumerate() {
        if id.is_multiple_of(1_024) {
            context.check_cancelled()?;
        }
        routes[id] = key
            .as_ref()
            .and_then(|key| operator.state.right.get(key))
            .map(|bucket| Route {
                bucket,
                cursor: bucket.cursor_at(time),
                latest: None,
            });
    }
    Ok(())
}

impl Route<'_> {
    fn candidate(
        &mut self,
        time: i64,
        operator: &StreamAsofJoinOperator,
        plan: &mut OutputPlanBuilder<'_>,
        workspace: &mut MemoryReservation,
    ) -> Result<Option<(usize, usize)>> {
        let Some(row) = self
            .bucket
            .candidate_monotonic(time, operator.spec.tolerance_micros(), &mut self.cursor)
            .copied()
        else {
            return Ok(None);
        };
        if self.latest.is_none_or(|(previous, _)| previous != row) {
            #[cfg(test)]
            PAYLOAD_RESOLUTIONS.set(PAYLOAD_RESOLUTIONS.get() + 1);
            let source =
                plan.right_source(operator.state.batches.view(row), workspace, &operator.name)?;
            self.latest = Some((row, source));
        }
        let (_, source) = self.latest.expect("resolved ASOF latest payload");
        Ok(Some((source, row.row as usize)))
    }
}

pub(super) fn route_slots<'a, T>(
    count: usize,
    inline: &'a mut [Option<T>; 64],
    heap: &'a mut Vec<Option<T>>,
) -> &'a mut [Option<T>] {
    if count <= inline.len() {
        &mut inline[..count]
    } else {
        heap.resize_with(count, || None);
        &mut heap[..]
    }
}

fn append_payload(
    (time, left): (i64, state::RowRef),
    route: Option<&mut Route<'_>>,
    source: usize,
    operator: &StreamAsofJoinOperator,
    plan: &mut OutputPlanBuilder<'_>,
    workspace: &mut MemoryReservation,
) -> Result<()> {
    plan.left_position(source, left.row as usize, &operator.name)?;
    let right = route
        .map(|route| route.candidate(time, operator, plan, workspace))
        .transpose()?
        .flatten();
    plan.push_right_source(right, &operator.name)
}

pub(super) async fn append_dictionary_run(
    (keys, mut rows): (
        &[Option<state::Encoding>],
        impl Iterator<Item = (i64, usize, state::RowRef)>,
    ),
    operator: &StreamAsofJoinOperator,
    context: &StreamOperatorContext<'_>,
    plan: &mut OutputPlanBuilder<'_>,
    workspace: &mut MemoryReservation,
    index: &mut usize,
) -> Result<()> {
    let (first_time, first_key, first_row) = rows.next().expect("nonempty ASOF merge run");
    let mut inline = std::array::from_fn::<_, 64, _>(|_| None);
    let mut heap = Vec::new();
    let routes = route_slots(keys.len(), &mut inline, &mut heap);
    routes_for_keys(keys, routes, first_time, operator, context)?;
    let mut source = 0;
    for (offset, (time, key, left)) in std::iter::once((first_time, first_key, first_row))
        .chain(rows)
        .enumerate()
    {
        check_match_progress(*index, context).await?;
        source = left_run_source(source, offset, left, operator, plan, workspace)?;
        append_payload(
            (time, left),
            routes[key].as_mut(),
            source,
            operator,
            plan,
            workspace,
        )?;
        *index += 1;
    }
    Ok(())
}
