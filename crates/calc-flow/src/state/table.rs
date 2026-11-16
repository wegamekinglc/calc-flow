//! Rolling's owned state-table seam.
//!
//! Kernel-facing rows and rings are backed by copy-on-write pages in prepared
//! key-group partitions. The legacy Arrow snapshot remains available for P2;
//! frozen dirty lineages provide the P3 capture boundary without encoding.

mod core;
mod pages;

pub(crate) use self::core::{
    EntityId, EntityKey, KeyGroupId, PageId, Retention, RingKind, StateTable, TableFreeze,
    TableLineage, TableSnapshot, ValueKind,
};

use self::core::{KeyedValue, RingValue};
use std::sync::Arc;

use datafusion::arrow::{
    array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StructArray, UInt64Array},
    datatypes::{DataType, Field},
};

use crate::CalcFlowError;

pub(super) fn keyed_array(kind: ValueKind, values: &[KeyedValue]) -> ArrayRef {
    match kind {
        ValueKind::UInt64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    KeyedValue::UInt64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<UInt64Array>(),
        ),
        ValueKind::Int64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    KeyedValue::Int64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<Int64Array>(),
        ),
        ValueKind::Float64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    KeyedValue::Float64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<Float64Array>(),
        ),
        ValueKind::Boolean => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    KeyedValue::Boolean(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<BooleanArray>(),
        ),
    }
}

pub(super) fn ring_array(kind: RingKind, values: &[RingValue]) -> ArrayRef {
    match kind {
        RingKind::Float64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    RingValue::Float64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<Float64Array>(),
        ),
        RingKind::Int64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    RingValue::Int64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<Int64Array>(),
        ),
        RingKind::UInt64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    RingValue::UInt64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<UInt64Array>(),
        ),
        RingKind::PairFloat64 => {
            let left = values
                .iter()
                .map(|value| match value {
                    RingValue::PairFloat64(value) => value.map(|pair| pair.0),
                    _ => unreachable!(),
                })
                .collect::<Float64Array>();
            let right = values
                .iter()
                .map(|value| match value {
                    RingValue::PairFloat64(value) => value.map(|pair| pair.1),
                    _ => unreachable!(),
                })
                .collect::<Float64Array>();
            Arc::new(StructArray::from(vec![
                (
                    Arc::new(Field::new("left", DataType::Float64, true)),
                    Arc::new(left) as ArrayRef,
                ),
                (
                    Arc::new(Field::new("right", DataType::Float64, true)),
                    Arc::new(right) as ArrayRef,
                ),
            ]))
        }
    }
}

fn internal(message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{EntityKey, Retention, RingKind, StateTable, ValueKind};
    use datafusion::arrow::array::{Array, StringArray};

    #[test]
    fn rollback_restores_dictionary_keyed_rows_and_overwritten_ring_slots() {
        let mut table = StateTable::new();
        let counter = table.add_keyed_column("counter", ValueKind::UInt64);
        let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(2));
        table.begin().unwrap();
        let entity = table
            .resolve_or_append(EntityKey::new(b"a".to_vec()))
            .unwrap();
        table.set_u64(counter, entity, 1).unwrap();
        table.push_f64(history, entity, Some(1.0)).unwrap();
        table.push_f64(history, entity, Some(2.0)).unwrap();
        table.commit().unwrap();

        let before = table.snapshot();
        table.begin().unwrap();
        table.set_u64(counter, entity, 9).unwrap();
        table.push_f64(history, entity, Some(3.0)).unwrap();
        table.push_f64(history, entity, Some(4.0)).unwrap();
        let other = table
            .resolve_or_append(EntityKey::new(b"b".to_vec()))
            .unwrap();
        table.set_u64(counter, other, 5).unwrap();
        table.rollback().unwrap();

        assert_eq!(table.entity_count(), 1);
        assert_eq!(table.u64(counter, entity).unwrap(), 1);
        assert_eq!(
            table.f64_history(history, entity).unwrap(),
            vec![Some(1.0), Some(2.0)]
        );
        assert_eq!(table.logical_bytes(), before.logical_bytes());
        table.restore(before).unwrap();
        assert_eq!(table.entity_count(), 1);
    }

    #[test]
    fn rows_float64_history_is_charged_as_values_and_validity_bits() {
        let mut table = StateTable::new();
        let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(20));
        table.begin().unwrap();
        let entity = table.resolve_or_append(EntityKey::new(Vec::new())).unwrap();
        table.push_f64(history, entity, Some(1.0)).unwrap();
        table.commit().unwrap();

        assert_eq!(
            table.ring_logical_bytes(history),
            20 * 8 + 20_usize.div_ceil(8)
        );
        assert!(table.logical_bytes() <= 20 * 8 + 20_usize.div_ceil(8) + 128);
    }

    #[test]
    fn snapshot_restore_preserves_float_bits_and_ring_order() {
        let mut table = StateTable::new();
        let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(3));
        table.begin().unwrap();
        let entity = table
            .resolve_or_append(EntityKey::new(b"entity".to_vec()))
            .unwrap();
        for value in [
            Some(-0.0),
            None,
            Some(f64::from_bits(0x7ff8_0000_0000_0042)),
        ] {
            table.push_f64(history, entity, value).unwrap();
        }
        table.push_f64(history, entity, Some(4.0)).unwrap();
        table.commit().unwrap();
        let snapshot = table.snapshot();

        table.begin().unwrap();
        table.push_f64(history, entity, Some(99.0)).unwrap();
        table.commit().unwrap();
        table.restore(snapshot).unwrap();

        let restored = table.f64_history(history, entity).unwrap();
        assert_eq!(restored.len(), 3);
        assert!(restored[0].is_none());
        assert_eq!(
            restored[1].unwrap().to_bits(),
            f64::from_bits(0x7ff8_0000_0000_0042).to_bits()
        );
        assert_eq!(restored[2].unwrap().to_bits(), 4.0_f64.to_bits());
    }

    #[test]
    fn value_log_appends_are_interleaved_and_rolled_back() {
        let mut table = StateTable::new();
        let log = table.add_value_log("values");
        table.begin().unwrap();
        let first = table
            .append_strings(log, StringArray::from(vec![Some("a"), None]))
            .unwrap();
        table.commit().unwrap();
        table.begin().unwrap();
        let second = table
            .append_strings(log, StringArray::from(vec![Some("b")]))
            .unwrap();
        table.rollback().unwrap();

        assert_eq!(first, 0..2);
        assert_eq!(second, 2..3);
        assert_eq!(table.value_log_len(log), 2);
        let values = table.strings(log).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values.value(0), "a");
        assert!(values.is_null(1));
    }

    #[test]
    fn restore_rejects_column_inventory_mismatch_before_mutating_state() {
        let mut table = StateTable::new();
        table.add_keyed_column("counter", ValueKind::UInt64);
        let snapshot = table.snapshot();
        let mut other = StateTable::new();
        other.add_keyed_column("different", ValueKind::UInt64);
        assert!(other.restore(snapshot).is_err());
        assert_eq!(other.entity_count(), 0);
    }

    #[test]
    fn float64_rows_group_preserves_mean_bits_across_rollback_and_restore() {
        let mut table = StateTable::new();
        let group = table.add_numeric_rows_column("mean", 2);
        table.begin().unwrap();
        let entity = table
            .resolve_or_append(EntityKey::new(b"entity".to_vec()))
            .unwrap();
        table.push_numeric_rows(group, entity, Some(1.0)).unwrap();
        table.push_numeric_rows(group, entity, Some(3.0)).unwrap();
        table.commit().unwrap();
        assert_eq!(
            table
                .numeric_mean(group, entity)
                .unwrap()
                .unwrap()
                .to_bits(),
            2.0_f64.to_bits()
        );
        let snapshot = table.snapshot();

        table.begin().unwrap();
        table.push_numeric_rows(group, entity, Some(5.0)).unwrap();
        assert_eq!(
            table
                .numeric_mean(group, entity)
                .unwrap()
                .unwrap()
                .to_bits(),
            4.0_f64.to_bits()
        );
        table.rollback().unwrap();
        assert_eq!(
            table
                .numeric_mean(group, entity)
                .unwrap()
                .unwrap()
                .to_bits(),
            2.0_f64.to_bits()
        );
        assert_eq!(
            table.f64_history(group, entity).unwrap(),
            vec![Some(1.0), Some(3.0)]
        );

        table.restore(snapshot).unwrap();
        assert_eq!(
            table
                .numeric_mean(group, entity)
                .unwrap()
                .unwrap()
                .to_bits(),
            2.0_f64.to_bits()
        );
    }
}

#[cfg(test)]
mod page_tests;
