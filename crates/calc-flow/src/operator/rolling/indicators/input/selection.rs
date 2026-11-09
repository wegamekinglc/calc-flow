use crate::operator::late_output::identity::{event_time, input_rows, validate_keys};
use crate::operator::rolling::{
    Batch, CompiledRollingSpec, EventTime, LateMetricDelta, LatePolicySpec, LateRowTally, Result,
    input_row_is_late, record_late_row,
};

use super::Check;

#[derive(Clone, Copy)]
pub(super) struct Row {
    pub record: usize,
    pub offset: usize,
    pub envelope: usize,
}

pub(super) struct Selection {
    pub rows: Vec<Row>,
    pub delta: LateMetricDelta,
}

#[cfg(test)]
thread_local! {
    static LIVE_SELECTION_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
impl Selection {
    pub(super) fn live_test_bytes() -> usize {
        LIVE_SELECTION_BYTES.with(std::cell::Cell::get)
    }
}

#[cfg(test)]
impl Drop for Selection {
    fn drop(&mut self) {
        let bytes = self.rows.capacity() * size_of::<Row>();
        drop(std::mem::take(&mut self.rows));
        LIVE_SELECTION_BYTES.with(|live| live.set(live.get().checked_sub(bytes).unwrap()));
    }
}

pub(super) struct Filter<'a> {
    pub compiled: &'a CompiledRollingSpec,
    pub watermark: Option<EventTime>,
    pub policy: LatePolicySpec,
    pub lateness: u64,
    pub node: &'a str,
}

fn allocate(count: usize, node: &str) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(count).map_err(|_| {
        crate::operator::rolling::operator_error(node, "state.input_selection allocation failed")
    })?;
    if rows.capacity() > count {
        return Err(crate::operator::rolling::operator_error(
            node,
            "state.input_selection exceeded the admitted capacity",
        ));
    }
    Ok(rows)
}

impl Filter<'_> {
    pub(super) fn select(
        &self,
        input: &Batch,
        count: usize,
        check: Check<'_>,
    ) -> Result<Selection> {
        self.select_matching(input, count, false, check)
    }
    pub(super) fn select_late(
        &self,
        input: &Batch,
        count: usize,
        check: Check<'_>,
    ) -> Result<Selection> {
        self.select_matching(input, count, true, check)
    }
    fn select_matching(
        &self,
        input: &Batch,
        count: usize,
        late: bool,
        check: Check<'_>,
    ) -> Result<Selection> {
        check()?;
        let mut rows = allocate(count, self.node)?;
        let mut metrics = LateRowTally::default();
        self.stage_rows(input, &mut rows, &mut metrics, late, check)?;
        check()?;
        #[cfg(test)]
        LIVE_SELECTION_BYTES.with(|live| {
            live.set(
                live.get()
                    .checked_add(rows.capacity() * size_of::<Row>())
                    .unwrap(),
            );
        });
        Ok(Selection {
            rows,
            delta: metrics.into_delta(),
        })
    }
    fn stage_rows(
        &self,
        input: &Batch,
        rows: &mut Vec<Row>,
        metrics: &mut LateRowTally,
        late: bool,
        check: Check<'_>,
    ) -> Result<()> {
        for row in input_rows(input)? {
            check()?;
            self.stage(rows, metrics, &row, late)?;
        }
        Ok(())
    }
    fn stage(
        &self,
        rows: &mut Vec<Row>,
        metrics: &mut LateRowTally,
        row: &crate::operator::late_output::identity::InputRow<'_>,
        keep_late: bool,
    ) -> Result<()> {
        let time = self.time(row.record, row.row_index)?;
        let late = self.late(time, row.envelope_index)?;
        if late {
            record_late_row(metrics, self.watermark, time, self.node)?;
        }
        if late == keep_late {
            rows.push(Row {
                record: row.record_index,
                offset: row.row_index,
                envelope: row.envelope_index,
            });
        }
        Ok(())
    }
    fn time(
        &self,
        record: &datafusion::arrow::record_batch::RecordBatch,
        row: usize,
    ) -> Result<i64> {
        let time = event_time(
            record,
            row,
            self.compiled.event_time_index,
            self.node,
            "rolling",
        )?;
        let keys = self
            .compiled
            .partition_columns
            .iter()
            .map(|column| (column.index, false))
            .chain(
                self.compiled
                    .sequence_columns
                    .iter()
                    .map(|column| (column.index, true)),
            );
        validate_keys(record, row, keys, self.node, "rolling")?;
        Ok(time)
    }
    fn late(&self, time: i64, index: usize) -> Result<bool> {
        input_row_is_late(
            self.policy,
            self.lateness,
            time,
            self.watermark,
            index,
            self.node,
        )
    }
}
