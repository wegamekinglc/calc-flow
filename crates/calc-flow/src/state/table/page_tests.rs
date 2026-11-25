use super::{
    EntityKey, RingKind, StateTable, ValueKind,
    core::{KeyGroupId, PageId, Retention},
};
use datafusion::arrow::array::StringArray;

fn entity(table: &mut StateTable, key: &[u8]) -> super::EntityId {
    table.begin().unwrap();
    let entity = table
        .resolve_or_append(EntityKey::new(key.to_vec()))
        .unwrap();
    table.commit().unwrap();
    entity
}

#[test]
fn cloned_table_has_independent_page_storage_and_undo() {
    let mut table = StateTable::with_checkpointing(true);
    let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(4));
    let entity = entity(&mut table, b"a");
    table.begin().unwrap();
    table.push_f64(history, entity, Some(1.0)).unwrap();
    table.commit().unwrap();
    let log = table.add_value_log("values");
    table.begin().unwrap();
    table
        .append_strings(log, StringArray::from(vec!["x"]))
        .unwrap();
    table.commit().unwrap();

    let mut cloned = table.clone();
    assert_eq!(cloned.value_log_len(log), 1);
    cloned.begin().unwrap();
    cloned.push_f64(history, entity, Some(2.0)).unwrap();
    cloned.commit().unwrap();
    assert_eq!(table.f64_history(history, entity).unwrap(), vec![Some(1.0)]);
    assert_eq!(
        cloned.f64_history(history, entity).unwrap(),
        vec![Some(1.0), Some(2.0)]
    );

    cloned.begin().unwrap();
    cloned.push_f64(history, entity, Some(3.0)).unwrap();
    cloned.rollback().unwrap();
    assert_eq!(
        cloned.f64_history(history, entity).unwrap(),
        vec![Some(1.0), Some(2.0)]
    );
    assert_eq!(table.f64_history(history, entity).unwrap(), vec![Some(1.0)]);
}

#[test]
fn checkpoint_off_has_zero_dirty_allocation_and_writes_in_place() {
    let mut table = StateTable::with_checkpointing(false);
    let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(4));
    let entity = entity(&mut table, b"a");
    table.begin().unwrap();
    table.push_f64(history, entity, Some(1.0)).unwrap();
    table.commit().unwrap();
    let before = table.ring_page_id(history, entity).unwrap();

    table.begin().unwrap();
    table.push_f64(history, entity, Some(2.0)).unwrap();
    table.commit().unwrap();

    assert_eq!(table.ring_page_id(history, entity).unwrap(), before);
    let snapshot = table.snapshot();
    table.restore(snapshot).unwrap();
    table.begin().unwrap();
    table.push_f64(history, entity, Some(3.0)).unwrap();
    table.commit().unwrap();
    assert_eq!(table.ring_page_id(history, entity).unwrap(), before);
    assert!(table.first_dirty_slots(KeyGroupId(0)).is_empty());
    assert_eq!(table.dirty_list_capacity(KeyGroupId(0)), 0);
}

#[test]
fn frozen_ring_is_immutable_and_first_post_freeze_write_cows_once() {
    let mut table = StateTable::with_checkpointing(true);
    let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(4));
    let entity = entity(&mut table, b"a");
    table.begin().unwrap();
    table.push_f64(history, entity, Some(1.0)).unwrap();
    table.push_f64(history, entity, Some(2.0)).unwrap();
    table.commit().unwrap();
    let before = table.ring_page_id(history, entity).unwrap();
    let freeze = table.freeze(1).unwrap();
    assert!(freeze.reservation_bytes() > 0);
    assert_eq!(
        table.checkpoint_reservation_bytes(),
        freeze.reservation_bytes()
    );

    table.begin().unwrap();
    table.push_f64(history, entity, Some(3.0)).unwrap();
    table.commit().unwrap();
    let cow = table.ring_page_id(history, entity).unwrap();
    assert_eq!(cow.slot, before.slot);
    assert_eq!(cow.generation, before.generation + 1);
    assert_eq!(
        freeze.ring_history(history, entity).unwrap(),
        vec![Some(1.0), Some(2.0)]
    );
    assert_eq!(
        table.f64_history(history, entity).unwrap(),
        vec![Some(1.0), Some(2.0), Some(3.0)]
    );

    table.begin().unwrap();
    table.push_f64(history, entity, Some(4.0)).unwrap();
    table.commit().unwrap();
    assert_eq!(table.ring_page_id(history, entity).unwrap(), cow);
    assert_eq!(
        freeze.ring_history(history, entity).unwrap(),
        vec![Some(1.0), Some(2.0)]
    );
}

#[test]
fn dirty_first_touch_and_undo_restore_page_and_allocator_state() {
    let mut table = StateTable::with_checkpointing(true);
    let counter = table.add_keyed_column("counter", ValueKind::UInt64);
    let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(3));
    let first = entity(&mut table, b"first");
    table.begin().unwrap();
    table.set_u64(counter, first, 1).unwrap();
    table.push_f64(history, first, Some(1.0)).unwrap();
    table.commit().unwrap();
    let first_page = table.ring_page_id(history, first).unwrap();
    let first_dirty = table.first_dirty_slots(KeyGroupId(0));
    let before_bytes = table.logical_bytes();

    table.begin().unwrap();
    table.set_u64(counter, first, 9).unwrap();
    table.push_f64(history, first, Some(2.0)).unwrap();
    let second = table
        .resolve_or_append(EntityKey::new(b"second".to_vec()))
        .unwrap();
    table.set_u64(counter, second, 2).unwrap();
    table.push_f64(history, second, Some(3.0)).unwrap();
    table.rollback().unwrap();

    assert_eq!(table.entity_count(), 1);
    assert_eq!(table.u64(counter, first).unwrap(), 1);
    assert_eq!(table.f64_history(history, first).unwrap(), vec![Some(1.0)]);
    assert_eq!(table.ring_page_id(history, first).unwrap(), first_page);
    assert_eq!(table.first_dirty_slots(KeyGroupId(0)), first_dirty);
    assert_eq!(table.logical_bytes(), before_bytes);
    assert!(table.freeze(1).is_ok());
}

#[test]
fn rollback_restores_transaction_stamp_and_first_touch_undo() {
    let mut table = StateTable::with_checkpointing(true);
    let counter = table.add_keyed_column("counter", ValueKind::UInt64);
    let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(4));
    let entity = entity(&mut table, b"a");
    table.begin().unwrap();
    table.set_u64(counter, entity, 1).unwrap();
    table.push_f64(history, entity, Some(1.0)).unwrap();
    table.commit().unwrap();

    for rejected in [2_u32, 3, 4] {
        table.begin().unwrap();
        table.set_u64(counter, entity, u64::from(rejected)).unwrap();
        table
            .push_f64(history, entity, Some(f64::from(rejected)))
            .unwrap();
        table.rollback().unwrap();
        assert_eq!(table.u64(counter, entity).unwrap(), 1);
        assert_eq!(table.f64_history(history, entity).unwrap(), vec![Some(1.0)]);
    }
    table.begin().unwrap();
    table.set_u64(counter, entity, 5).unwrap();
    table.push_f64(history, entity, Some(5.0)).unwrap();
    table.commit().unwrap();
    assert_eq!(table.u64(counter, entity).unwrap(), 5);
    assert_eq!(
        table.f64_history(history, entity).unwrap(),
        vec![Some(1.0), Some(5.0)]
    );
}

#[test]
fn freeze_rejects_open_transaction_and_overlap() {
    let mut table = StateTable::with_checkpointing(true);
    table.begin().unwrap();
    assert!(table.freeze(1).is_err());
    table.rollback().unwrap();
    let _freeze = table.freeze(1).unwrap();
    assert!(table.freeze(2).is_err());
    table.abort_freeze(1).unwrap();
    assert_eq!(table.checkpoint_reservation_bytes(), 0);
}

#[test]
fn multiple_groups_tombstones_and_free_slots_preserve_identity() {
    let mut table = StateTable::with_key_groups(true, &[KeyGroupId(0), KeyGroupId(1)]).unwrap();
    let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(2));
    table.begin().unwrap();
    let entity = table
        .resolve_or_append_in_group(EntityKey::new(b"right".to_vec()), KeyGroupId(1))
        .unwrap();
    table.push_f64(history, entity, Some(1.0)).unwrap();
    table.commit().unwrap();
    let before = table.ring_page_id(history, entity).unwrap();
    assert_eq!(before.key_group, KeyGroupId(1));

    let first = table.freeze(1).unwrap();
    assert_eq!(first.partition_count(), 2);
    assert_eq!(first.captured_pages(), 2);
    table.settle_freeze(1, first.complete()).unwrap();
    table.begin().unwrap();
    table.delete_ring(history, entity).unwrap();
    table.commit().unwrap();
    let second = table.freeze(2).unwrap();
    table.settle_freeze(2, second.complete()).unwrap();

    table.begin().unwrap();
    let replacement = table
        .resolve_or_append_in_group(EntityKey::new(b"replacement".to_vec()), KeyGroupId(1))
        .unwrap();
    table.commit().unwrap();
    let reused = table.ring_page_id(history, replacement).unwrap();
    assert_eq!(reused.slot, before.slot);
    assert!(reused.generation > before.generation);
    assert_ne!(reused, before);
}

#[test]
fn keyed_pages_split_and_duration_ring_growth_changes_generation() {
    let mut table = StateTable::with_checkpointing(true);
    let keyed = table.add_keyed_column("keyed", ValueKind::UInt64);
    let history = table.add_ring_column("history", RingKind::Float64, Retention::Duration);
    let mut previous: Option<PageId> = None;
    table.begin().unwrap();
    let duration = table
        .resolve_or_append(EntityKey::new(b"duration".to_vec()))
        .unwrap();
    for value in 0..5 {
        table
            .push_f64(history, duration, Some(f64::from(value)))
            .unwrap();
        let current = table.ring_page_id(history, duration).unwrap();
        if matches!(value, 0 | 1 | 2 | 4) {
            assert!(current.generation > previous.map_or(0, |page| page.generation));
        } else {
            assert_eq!(current, previous.unwrap());
        }
        previous = Some(current);
    }
    table.commit().unwrap();
    assert_eq!(
        table.f64_history(history, duration).unwrap(),
        (0..5)
            .map(|value| Some(f64::from(value)))
            .collect::<Vec<_>>()
    );

    table.begin().unwrap();
    for index in 0..1_025_u32 {
        let entity = table
            .resolve_or_append(EntityKey::new(index.to_le_bytes().to_vec()))
            .unwrap();
        table.set_u64(keyed, entity, u64::from(index)).unwrap();
    }
    table.commit().unwrap();
    assert_eq!(table.keyed_page_count(keyed, KeyGroupId(0)), 2);
}

#[test]
fn freeze_captures_only_dirty_pages_and_restore_starts_clean() {
    let mut table = StateTable::with_checkpointing(true);
    let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(2));
    let first = entity(&mut table, b"first");
    let first_freeze = table.freeze(1).unwrap();
    table.settle_freeze(1, first_freeze.complete()).unwrap();
    for index in 0..32_u32 {
        entity(&mut table, &index.to_le_bytes());
    }
    let second_freeze = table.freeze(2).unwrap();
    table.settle_freeze(2, second_freeze.complete()).unwrap();

    table.begin().unwrap();
    table.push_f64(history, first, Some(1.0)).unwrap();
    table.commit().unwrap();
    let freeze = table.freeze(3).unwrap();
    assert_eq!(freeze.captured_pages(), 1);
    table.settle_freeze(3, freeze.complete()).unwrap();
    let snapshot = table.snapshot();
    table.restore(snapshot).unwrap();
    assert!(table.first_dirty_slots(KeyGroupId(0)).is_empty());
}

#[test]
fn rollback_restores_keyed_page_capacity_after_growth() {
    for checkpointing in [false, true] {
        let mut table = StateTable::with_checkpointing(checkpointing);
        let counter = table.add_keyed_column("counter", ValueKind::UInt64);
        let mut first = None;
        for index in 1..=8_u64 {
            table.begin().unwrap();
            let next = table
                .resolve_or_append(EntityKey::new(format!("entity-{index}").into_bytes()))
                .unwrap();
            table.set_u64(counter, next, index).unwrap();
            table.commit().unwrap();
            first.get_or_insert(next);
        }
        let first = first.expect("the first entity was appended");
        let before = table.logical_bytes();

        table.begin().unwrap();
        let next = table
            .resolve_or_append(EntityKey::new(b"entity-9".to_vec()))
            .unwrap();
        table.set_u64(counter, next, 9).unwrap();
        assert!(table.logical_bytes() > before);
        table.rollback().unwrap();

        assert_eq!(table.entity_count(), 8, "checkpointing={checkpointing}");
        assert_eq!(table.u64(counter, first).unwrap(), 1);
        assert_eq!(
            table.logical_bytes(),
            before,
            "checkpointing={checkpointing}"
        );
    }
}

#[test]
fn test_recurrent_full_ticks_borrow_each_contiguous_page_once() {
    use datafusion::arrow::array::Float64Array;

    let mut table = StateTable::with_checkpointing(false);
    let history = table.add_numeric_rows_column("mean", 20);
    let order = [b"a", b"b", b"c", b"d"].map(|key| entity(&mut table, key));
    let locals: Vec<_> = (0..order.len()).cycle().take(4_097).collect();
    let values = Float64Array::from_iter_values(locals.iter().map(|_| 2.0));
    table.begin().unwrap();
    super::core::RECURRENT_PAGE_BORROWS.set(0);
    let mut count = 0;
    assert!(
        table
            .write_numeric_full_tick(history, &order, &locals, &values, false, |row, numeric| {
                assert_eq!(row, count);
                assert_eq!(numeric[0].mean().to_bits(), 2.0_f64.to_bits());
                count += 1;
                Ok(())
            })
            .unwrap()
    );
    let borrows = super::core::RECURRENT_PAGE_BORROWS.get();
    table.rollback().unwrap();
    assert_eq!(count, locals.len());
    assert!(
        order
            .iter()
            .all(|&id| table.f64_history(history, id).unwrap().is_empty())
    );
    assert_eq!(borrows, order.len(), "4,097 rows touched four ring pages");
}

#[test]
fn test_recurrent_tick_error_rolls_back_and_retries_restored_windows() {
    use datafusion::arrow::array::Float64Array;

    let mut table = StateTable::with_checkpointing(false);
    let history = table.add_numeric_rows_column("means", 2);
    table.add_numeric_window(history, 5);
    let [a, b, c, d] = [b"a", b"b", b"c", b"d"].map(|key| entity(&mut table, key));
    table.begin().unwrap();
    for (id, value) in [(a, 1.0), (b, 2.0), (c, 3.0), (d, 4.0)] {
        table.push_numeric_rows(history, id, Some(value)).unwrap();
    }
    table.commit().unwrap();
    let snapshot = table.snapshot();
    table.restore(snapshot).unwrap();
    for (order, rows) in [([b, c], 21), ([d, a], 21), ([a, b], 3)] {
        let mut reference = table.clone();
        let before = table.canonical_state_for_test();
        let locals: Vec<_> = (0..order.len()).cycle().take(rows).collect();
        let inputs = [Some(2.0), None, Some(f64::NAN), Some(7.0), Some(1.0)];
        let values = inputs
            .into_iter()
            .cycle()
            .take(rows)
            .collect::<Float64Array>();
        table.begin().unwrap();
        let stop_at = rows - 2;
        let mut emitted = 0;
        let error = table
            .write_numeric_full_tick(history, &order, &locals, &values, false, |row, _| {
                assert_eq!(row, emitted);
                emitted += 1;
                if row == stop_at {
                    return Err(super::internal("injected output failure"));
                }
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("injected output failure"));
        assert_eq!(emitted, stop_at + 1);
        table.rollback().unwrap();
        assert_eq!(table.canonical_state_for_test(), before);
        reference.begin().unwrap();
        table.begin().unwrap();
        assert!(
            table
                .write_numeric_full_tick(
                    history,
                    &order,
                    &locals,
                    &values,
                    false,
                    |row, observed| {
                        let id = order[locals[row]];
                        let value = inputs[row % inputs.len()].filter(|v| !v.is_nan());
                        reference.push_numeric_rows(history, id, value)?;
                        for (actual, scalar) in observed
                            .iter()
                            .zip(reference.numeric_accumulators(history, id)?)
                        {
                            assert_eq!(actual.valid_count, scalar.valid_count);
                            assert_eq!(actual.mean().to_bits(), scalar.mean().to_bits());
                        }
                        Ok(())
                    }
                )
                .unwrap()
        );
        table.commit().unwrap();
        reference.commit().unwrap();
        assert_eq!(
            table.canonical_state_for_test(),
            reference.canonical_state_for_test()
        );
    }
}

#[test]
fn numeric_tick_tail_requires_the_same_entity_prefix_before_any_write() {
    use datafusion::arrow::array::Float64Array;

    let mut table = StateTable::with_checkpointing(false);
    let history = table.add_numeric_rows_column("mean", 2);
    // The input order differs from the order in which the residents were created.
    let b = entity(&mut table, b"b");
    let a = entity(&mut table, b"a");
    let values = Float64Array::from(vec![Some(2.0), None, Some(6.0), Some(8.0), Some(10.0)]);
    table.begin().unwrap();
    let mut means = Vec::new();
    let mut emit = |row, numeric: &[crate::accumulate::NumericAccumulator]| {
        assert_eq!(row, means.len());
        let value = numeric[0];
        means.push((value.valid_count > 0).then(|| value.mean()));
        Ok(())
    };
    for (order, locals) in [
        ([a, a], &[0, 1, 0, 1, 0][..]),
        ([a, b], &[0, 1, 0, 1, 1]),
        ([a, b], &[0, 0, 1, 0, 0]),
        ([a, b], &[0]),
        ([a, b], &[0, 1, 2]),
    ] {
        assert!(
            !table
                .write_numeric_full_tick(history, &order, locals, &values, false, &mut emit)
                .unwrap()
        );
    }
    assert!(
        table
            .write_numeric_full_tick(history, &[a, b], &[0, 1, 0, 1, 0], &values, false, emit)
            .unwrap()
    );
    assert_eq!(
        means,
        vec![Some(2.0), None, Some(4.0), Some(8.0), Some(8.0)]
    );
    table.rollback().unwrap();
    assert!(table.f64_history(history, a).unwrap().is_empty());
    assert!(table.f64_history(history, b).unwrap().is_empty());
}
