//! Partition-local transitions for physically sorted SQL windows.

use std::ops::Range;

use datafusion::arrow::compute::kernels::partition::partition;

use super::{
    ArrayRef, DerivedBuilder, Instant, RecordBatch, Result, RollingKernelMetrics,
    RollingKernelPlan, TimestampMicrosecondArray, TypedEntityState, TypedGroupInput,
    append_typed_outputs, encode_rows, nanos, operator_error, update_typed_groups,
};

/// Only the last SQL partition can continue into the next input batch.
#[derive(Clone, Debug, Default)]
pub(in crate::operator::rolling) struct SortedRollingState {
    fingerprint: Option<String>,
    last_key: Option<Vec<u8>>,
    entity: Option<TypedEntityState>,
}

pub(in crate::operator::rolling) struct SortedRollingExecution {
    pub columns: Vec<ArrayRef>,
    pub state: SortedRollingState,
    pub metrics: RollingKernelMetrics,
}

struct SortedPartitions {
    ranges: Vec<Range<usize>>,
    first_key: Option<Vec<u8>>,
    last_key: Option<Vec<u8>>,
}

struct SortedInputs<'a> {
    groups: Vec<TypedGroupInput>,
    event_times: &'a TimestampMicrosecondArray,
}

impl SortedRollingState {
    fn validate_plan(&self, plan: &RollingKernelPlan, node_id: &str) -> Result<()> {
        if self
            .fingerprint
            .as_deref()
            .is_some_and(|fingerprint| fingerprint != plan.fingerprint)
        {
            return Err(operator_error(
                node_id,
                "sorted rolling state belongs to a different kernel plan",
            ));
        }
        Ok(())
    }

    fn take_continuing_entity(
        &mut self,
        partitions: &SortedPartitions,
    ) -> Option<TypedEntityState> {
        let entity = self.entity.take();
        if partitions.first_key == self.last_key || partitions.ranges.is_empty() {
            entity
        } else {
            None
        }
    }
}

impl RollingKernelPlan {
    /// The physical planner must require globally contiguous partitions and
    /// ascending row order before selecting this transition.
    pub(in crate::operator::rolling) fn update_sorted_and_fill_owned(
        &self,
        mut state: SortedRollingState,
        input: &RecordBatch,
        node_id: &str,
    ) -> Result<SortedRollingExecution> {
        state.validate_plan(self, node_id)?;
        let input_validation_ns = self.validate_input_timed(input, node_id)?;
        let started = Instant::now();
        let partitions = self.sorted_partitions(input, node_id)?;
        let entity_encode_ns = nanos(started.elapsed());
        let inputs = self.sorted_inputs(input, node_id)?;
        let mut builders = self
            .outputs
            .iter()
            .map(|output| DerivedBuilder::new(*output, input.num_rows()))
            .collect::<Vec<_>>();
        let mut entity = state.take_continuing_entity(&partitions);
        let started = Instant::now();
        self.fill_sorted_ranges(
            &partitions.ranges,
            &inputs,
            &mut entity,
            &mut builders,
            node_id,
        )?;
        let kernel_ns = nanos(started.elapsed());
        let started = Instant::now();
        let columns = builders
            .into_iter()
            .map(DerivedBuilder::finish)
            .collect::<Result<Vec<_>>>()?;
        let output_build_ns = nanos(started.elapsed());
        let last_key = partitions.last_key.or(state.last_key);
        let metrics = RollingKernelMetrics {
            input_validation_ns,
            entity_encode_ns,
            kernel_ns,
            output_build_ns,
            input_rows: input.num_rows(),
            output_rows: input.num_rows(),
            entities: usize::from(entity.is_some()),
            state_bytes: entity.as_ref().map_or(0, TypedEntityState::estimated_bytes)
                + last_key.as_ref().map_or(0, Vec::len),
            ..RollingKernelMetrics::default()
        };
        Ok(SortedRollingExecution {
            columns,
            state: SortedRollingState {
                fingerprint: Some(self.fingerprint.clone()),
                last_key,
                entity,
            },
            metrics,
        })
    }

    fn sorted_partitions(&self, input: &RecordBatch, node_id: &str) -> Result<SortedPartitions> {
        let partition_columns = self
            .partition_columns
            .iter()
            .map(|&index| input.column(index).clone())
            .collect::<Vec<_>>();
        let ranges = partition(&partition_columns)
            .map_err(|error| operator_error(node_id, &format!("SQL partitioning failed: {error}")))?
            .ranges();
        let first_key = ranges
            .first()
            .map(|range| self.sorted_partition_key(input, range.start, node_id))
            .transpose()?;
        let last_key = if ranges.len() > 1 {
            Some(self.sorted_partition_key(input, input.num_rows() - 1, node_id)?)
        } else {
            first_key.clone()
        };
        Ok(SortedPartitions {
            ranges,
            first_key,
            last_key,
        })
    }

    fn sorted_inputs<'a>(&self, input: &'a RecordBatch, node_id: &str) -> Result<SortedInputs<'a>> {
        let groups = self.typed_inputs(input, node_id)?;
        let event_times = input
            .column(self.event_time_index)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| operator_error(node_id, "SQL rolling event-time type changed"))?;
        Ok(SortedInputs {
            groups,
            event_times,
        })
    }

    fn fill_sorted_ranges(
        &self,
        ranges: &[Range<usize>],
        inputs: &SortedInputs<'_>,
        entity: &mut Option<TypedEntityState>,
        builders: &mut [DerivedBuilder],
        node_id: &str,
    ) -> Result<()> {
        if self.decay_pool.is_some() {
            self.fill_sorted_ranges_impl::<true>(ranges, inputs, entity, builders, node_id)
        } else {
            self.fill_sorted_ranges_impl::<false>(ranges, inputs, entity, builders, node_id)
        }
    }

    fn fill_sorted_ranges_impl<const HAS_DECAY: bool>(
        &self,
        ranges: &[Range<usize>],
        inputs: &SortedInputs<'_>,
        entity: &mut Option<TypedEntityState>,
        builders: &mut [DerivedBuilder],
        node_id: &str,
    ) -> Result<()> {
        for (range_index, range) in ranges.iter().enumerate() {
            if range_index > 0 || entity.is_none() {
                *entity = Some(TypedEntityState::new(&self.groups, range.len()));
            }
            let current = entity.as_mut().expect("nonempty SQL partition has state");
            if self.shared_rows.is_some() {
                current.prepare_shared_rows(self.shared_rows);
            }
            for row_index in range.clone() {
                self.update_sorted_row::<HAS_DECAY>(current, inputs, row_index, builders, node_id)?;
            }
        }
        Ok(())
    }

    fn update_sorted_row<const HAS_DECAY: bool>(
        &self,
        current: &mut TypedEntityState,
        inputs: &SortedInputs<'_>,
        row_index: usize,
        builders: &mut [DerivedBuilder],
        node_id: &str,
    ) -> Result<()> {
        current.transition_count = current
            .transition_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling entity transition count overflowed"))?;
        update_typed_groups(
            &inputs.groups,
            inputs.event_times.value(row_index),
            current,
            row_index,
            self.numerical_profile,
            self.nan_as_value,
            node_id,
        )?;
        append_typed_outputs::<HAS_DECAY>(
            &self.outputs,
            builders,
            current,
            &inputs.groups,
            row_index,
            node_id,
        )
    }

    fn sorted_partition_key(
        &self,
        input: &RecordBatch,
        row: usize,
        node_id: &str,
    ) -> Result<Vec<u8>> {
        let row = input.slice(row, 1);
        Ok(encode_rows(&row, &self.partition_columns, node_id)?
            .row(0)
            .data()
            .to_vec())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{
        array::{DictionaryArray, Float64Array, Int8Array, StringArray, UInt64Array},
        datatypes::{DataType, Field, Int8Type, Schema, TimeUnit},
    };

    use super::*;
    use crate::operator::rolling::{
        DataFusionRollingKernel, DataFusionRollingState, DataFusionRollingWindow,
    };

    fn dictionary_input(
        keys: Vec<i8>,
        dictionary: Vec<&str>,
        time: Vec<i64>,
        prices: Vec<f64>,
    ) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new(
                "symbol",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                false,
            ),
            Field::new("price", DataType::Float64, false),
        ]));
        let sequence = time
            .iter()
            .map(|&time| u64::try_from(time).unwrap())
            .collect::<Vec<_>>();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(TimestampMicrosecondArray::from(time)),
                Arc::new(UInt64Array::from(sequence)),
                Arc::new(
                    DictionaryArray::<Int8Type>::try_new(
                        Int8Array::from(keys),
                        Arc::new(StringArray::from(dictionary)),
                    )
                    .unwrap(),
                ),
                Arc::new(Float64Array::from(prices)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn p3_sorted_dual_avg_uses_one_shared_tape() {
        let input = dictionary_input(
            vec![0; 24],
            vec!["a"],
            (1..=24).collect(),
            (1..=24).map(f64::from).collect(),
        );
        let windows = [5, 20].map(|rows| DataFusionRollingWindow {
            input_index: 3,
            output_name: format!("mean{rows}"),
            rows,
            is_count: false,
            is_sum: false,
            duration_micros: None,
        });
        let kernel =
            DataFusionRollingKernel::compile(input.schema().as_ref(), &[2], &[0, 1], &windows)
                .unwrap();
        assert!(kernel.plan.shared_rows.is_some());
        let filled = kernel
            .update_and_fill(&DataFusionRollingState::default(), &input)
            .unwrap();
        let entity = filled.state.inner.entity.as_ref().unwrap();
        assert!(
            entity.shared_rows.is_some(),
            "real multi-AVG sorted route still owns two queues"
        );
        assert!(entity.groups.iter().all(|group| matches!(group, super::super::TypedWindowState::Numeric(state) if state.samples.len() == 0)));
    }

    fn p3_dual_kernel(input: &RecordBatch) -> DataFusionRollingKernel {
        let windows = [5, 20].map(|rows| DataFusionRollingWindow {
            input_index: 3,
            output_name: format!("mean{rows}"),
            rows,
            is_count: false,
            is_sum: false,
            duration_micros: None,
        });
        DataFusionRollingKernel::compile(input.schema().as_ref(), &[2], &[0, 1], &windows).unwrap()
    }

    fn p3_sorted_owner(state: &DataFusionRollingState) -> usize {
        Arc::as_ptr(
            state
                .inner
                .entity
                .as_ref()
                .unwrap()
                .shared_rows
                .as_ref()
                .unwrap(),
        ) as usize
    }

    fn p3_special_price(time: i64) -> f64 {
        match time % 9 {
            0 => f64::NAN,
            _ => f64::from(u32::try_from(time).unwrap()),
        }
    }

    fn p3_partition_batches() -> Vec<RecordBatch> {
        let first = dictionary_input(
            vec![0; 24],
            vec!["a"],
            (1..=24).collect(),
            (1..=24).map(p3_special_price).collect(),
        );
        let second = dictionary_input(
            std::iter::repeat_n(1, 8)
                .chain(std::iter::repeat_n(0, 12))
                .collect(),
            vec!["b", "a"],
            (25..=32).chain(1..=12).collect(),
            (25..=32).chain(1..=12).map(p3_special_price).collect(),
        );
        let third = dictionary_input(
            std::iter::repeat_n(1, 12)
                .chain(std::iter::repeat_n(0, 6))
                .collect(),
            vec!["c", "b"],
            (13..=24).chain(1..=6).collect(),
            (13..=24).chain(1..=6).map(p3_special_price).collect(),
        );
        vec![first, second, third]
    }

    fn p3_assert_column_bits(actual: &[ArrayRef], expected: &[ArrayRef]) {
        for (actual, expected) in actual.iter().zip(expected) {
            let actual = actual.as_any().downcast_ref::<Float64Array>().unwrap();
            let expected = expected.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.map(f64::to_bits))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.map(f64::to_bits))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn p3_sorted_multi_avg_partition_nan_and_refusal_match_original_queue_bits() {
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

        for refused in [false, true] {
            let pool = Arc::new(GreedyMemoryPool::new(if refused { 0 } else { 64 << 20 }));
            let _scope = super::super::shared_rows_test_pool(pool.clone());
            let batches = p3_partition_batches();
            let kernel = p3_dual_kernel(&batches[0]);
            let mut reference = kernel.clone();
            reference.plan.shared_rows = None;
            let mut actual = DataFusionRollingState::default();
            let mut old = DataFusionRollingState::default();
            for batch in batches {
                let expected = reference.update_and_fill(&old, &batch).unwrap();
                let filled = kernel.update_and_fill_owned(actual, &batch).unwrap();
                p3_assert_column_bits(&filled.columns, &expected.columns);
                let entity = filled.state.inner.entity.as_ref().unwrap();
                assert_eq!(entity.shared_rows.is_none(), refused);
                let credit = entity.shared_rows.as_ref().map_or(0, |tape| tape.bytes());
                assert_eq!(pool.reserved(), credit);
                assert!(filled.metrics.state_bytes >= credit);
                actual = filled.state;
                old = expected.state;
            }
            drop(actual);
            assert_eq!(pool.reserved(), 0);
        }
    }

    #[test]
    fn p3_sorted_owned_continuation_keeps_unique_owner_and_preserves_aliased_credit() {
        use crate::state_memory_pool::OwnedStatePool;
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

        let pool = Arc::new(OwnedStatePool::new(
            Arc::new(GreedyMemoryPool::new(64 << 20)),
            "sorted test",
            "sorted quota",
        ));
        let _scope = super::super::shared_rows_test_pool(pool.clone());
        let first = dictionary_input(
            vec![0; 24],
            vec!["a"],
            (1..=24).collect(),
            (1..=24).map(f64::from).collect(),
        );
        let kernel = p3_dual_kernel(&first);
        let mut state = kernel
            .update_and_fill_owned(DataFusionRollingState::default(), &first)
            .unwrap()
            .state;
        let original_owner = p3_sorted_owner(&state);
        let credit = pool.reserved();
        assert!(credit > 0);
        let one = |time| {
            dictionary_input(
                vec![0],
                vec!["a"],
                vec![time],
                vec![f64::from(u32::try_from(time).unwrap())],
            )
        };
        state = kernel.update_and_fill_owned(state, &one(25)).unwrap().state;
        assert_eq!(p3_sorted_owner(&state), original_owner);
        assert_eq!(pool.reserved(), credit);
        let alias = state.clone();
        let before = format!("{alias:?}");
        state = kernel.update_and_fill_owned(state, &one(26)).unwrap().state;
        assert_ne!(p3_sorted_owner(&state), p3_sorted_owner(&alias));
        assert_eq!(format!("{alias:?}"), before);
        assert_eq!(pool.reserved(), 2 * credit);
        drop(alias);
        assert_eq!(pool.reserved(), credit);
        let alias = state.clone();
        pool.set_limit(Some(u64::try_from(credit).unwrap()))
            .unwrap();
        let refused = kernel.update_and_fill_owned(state, &one(27)).unwrap();
        assert!(
            refused
                .state
                .inner
                .entity
                .as_ref()
                .unwrap()
                .shared_rows
                .is_none()
        );
        assert_eq!(pool.reserved(), credit);
        let values = refused
            .columns
            .iter()
            .map(|column| {
                column
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(0)
            })
            .collect::<Vec<_>>();
        assert_eq!(values, [25.0, 17.5]);
        drop(alias);
        assert_eq!(pool.reserved(), 0);
        drop(refused);
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn sorted_sql_route_keeps_dictionary_value_identity_and_only_last_partition() {
        let first = dictionary_input(vec![1, 1], vec!["unused", "a"], vec![1, 2], vec![2.0, 4.0]);
        let kernel = DataFusionRollingKernel::compile(
            first.schema().as_ref(),
            &[2],
            &[0, 1],
            &[DataFusionRollingWindow {
                input_index: 3,
                output_name: "mean".to_owned(),
                rows: 2,
                is_count: false,
                is_sum: false,
                duration_micros: None,
            }],
        )
        .unwrap();
        let first = kernel
            .update_and_fill(&DataFusionRollingState::default(), &first)
            .unwrap();
        let next = dictionary_input(vec![0], vec!["a", "unused"], vec![3], vec![6.0]);
        let empty = kernel
            .update_and_fill(&first.state, &next.slice(0, 0))
            .unwrap();
        assert_eq!(empty.columns[0].len(), 0);
        let continued = kernel.update_and_fill(&empty.state, &next).unwrap();
        let mean = continued.columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(mean.value(0).to_bits(), 5.0_f64.to_bits());
        assert_eq!(continued.metrics.entities, 1);
        let other = dictionary_input(vec![0], vec!["b"], vec![1], vec![99.0]);
        let other = kernel.update_and_fill(&continued.state, &other).unwrap();
        let mean = other.columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(mean.value(0).to_bits(), 99.0_f64.to_bits());
        assert_eq!(other.metrics.entities, 1);
    }
}
