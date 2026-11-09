use std::mem::size_of;

use super::super::super::{resources, scalar_materialization_plan};
use super::super::{
    materialization_error, retained, row_descriptors,
    selection::{Filter, Row},
};
use super::{Check, Selection, Work};
use crate::operator::rolling::{Result, operator_error};

impl Work {
    pub(super) fn preflight(&mut self, check: Check<'_>) -> Result<Selection> {
        let (count, scratch) = self.selection_size()?;
        let candidate = self.credit.source().clone_bound(&self.node)?;
        self.credit
            .adjust(candidate, scratch, self.budget, &self.node)?;
        let selection = Filter {
            compiled: &self.compiled,
            watermark: self.watermark,
            policy: self.policy,
            lateness: self.lateness,
            node: &self.node,
        }
        .select(&self.input, count, check)?;
        self.admit_selected(&selection, scratch, check)?;
        Ok(selection)
    }
    fn selection_size(&self) -> Result<(usize, u64)> {
        let count =
            self.input
                .table_payload()?
                .batches()
                .iter()
                .try_fold(0_usize, |n, record| {
                    n.checked_add(record.num_rows()).ok_or_else(|| {
                        operator_error(&self.node, "rolling input row count overflowed")
                    })
                })?;
        let bytes = count
            .checked_mul(size_of::<Row>())
            .ok_or_else(|| operator_error(&self.node, "rolling input selection size overflowed"))?;
        let identity = super::footprint_state::identity_bytes(&self.fingerprint, &self.node)?;
        Ok((
            count,
            resources::add(resources::size(bytes, &self.node)?, identity, &self.node)?,
        ))
    }
    fn admit_selected(
        &mut self,
        selected: &Selection,
        selection_bytes: u64,
        check: Check<'_>,
    ) -> Result<()> {
        let input = self.input.table_payload()?;
        let mut candidate = self.credit.source().clone_bound(&self.node)?;
        let backing = retained::plan_selected(
            selected
                .rows
                .iter()
                .map(|row| (input.batches()[row.record].columns(), row.offset)),
            &self.node,
            check,
        )?;
        candidate.merge(&backing, &self.node)?;
        let scratch = self.selected_scratch(selected, selection_bytes, check)?;
        self.credit
            .adjust(candidate, scratch, self.budget, &self.node)
    }
    fn selected_scratch(
        &self,
        selected: &Selection,
        selection_bytes: u64,
        check: Check<'_>,
    ) -> Result<u64> {
        let input = self.input.table_payload()?;
        let values = scalar_materialization_plan::plan_selected(
            selected
                .rows
                .iter()
                .map(|row| (input.batches()[row.record].columns(), row.offset)),
            check,
        )
        .map_err(|error| materialization_error(error, &self.node))?;
        let slots = row_descriptors(selected.rows.len(), self.schema.fields().len(), &self.node)?;
        let keys = self.selected_keys(selected, check)?;
        resources::add(
            selection_bytes,
            resources::add(values, resources::add(slots, keys, &self.node)?, &self.node)?,
            &self.node,
        )
    }
    fn selected_keys(&self, selected: &Selection, check: Check<'_>) -> Result<u64> {
        let input = self.input.table_payload()?;
        let mut bytes = 0;
        for key in self
            .compiled
            .partition_columns
            .iter()
            .chain(&self.compiled.sequence_columns)
        {
            let payload = scalar_materialization_plan::plan_selected(
                selected.rows.iter().map(|row| {
                    (
                        &input.batches()[row.record].columns()[key.index..=key.index],
                        row.offset,
                    )
                }),
                check,
            )
            .map_err(|error| materialization_error(error, &self.node))?;
            bytes = resources::add(
                bytes,
                resources::add(payload, payload, &self.node)?,
                &self.node,
            )?;
        }
        Ok(bytes)
    }
}
