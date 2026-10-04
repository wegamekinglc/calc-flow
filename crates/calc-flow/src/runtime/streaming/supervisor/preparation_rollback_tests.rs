use std::{future::pending, sync::Arc};

use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

use super::TaskSupervisor;
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeBudget, Result, StreamMessage, edge_channel,
};

#[tokio::test(flavor = "current_thread")]
async fn a13_dropping_unspawned_real_prepared_pair_rolls_back_registry_without_reusing_ids() {
    let mut supervisor = TaskSupervisor::new(CancellationToken::new());
    let registry = supervisor.registry();
    let settled = supervisor.settled.clone();
    let budget = EdgeBudget::new(1, 1 << 20).unwrap();
    let (mut first_sender, first_receiver) = edge_channel("prepared-first-input", budget).unwrap();
    let (second_sender, second_receiver) = edge_channel("prepared-second-input", budget).unwrap();
    let column: ArrayRef = Arc::new(Int64Array::from(vec![17]));
    let weak: std::sync::Weak<dyn Array> = Arc::downgrade(&column);
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )])),
        vec![column],
    )
    .unwrap();
    first_sender
        .send(StreamMessage::data(
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
        ))
        .await
        .unwrap();
    let charged_before_drop = first_sender.metrics();
    let prepared = supervisor.prepare_pair_with_failure_signals(
        "operator:a_asof",
        move |_| async move {
            let _retained = first_receiver;
            pending::<Result<()>>().await
        },
        "operator:c_select1",
        move |_| async move {
            let _retained = second_receiver;
            pending::<Result<()>>().await
        },
    );
    let first_ids = prepared.ids();
    let registered_before_drop = registry.snapshot();
    let drivers_before_drop = supervisor.physical_driver_count();
    drop(prepared);
    let registered_after_drop = registry.snapshot();
    let settled_after_drop = settled.lock().len();
    let charged_after_drop = first_sender.metrics();
    let source_released = weak.upgrade().is_none();
    for id in first_ids {
        registry.remove(id);
    }
    let next = supervisor.prepare_pair_with_failure_signals(
        "operator:next-first",
        |_| async { Ok(()) },
        "operator:next-second",
        |_| async { Ok(()) },
    );
    let next_ids = next.ids();
    drop(next);
    let next_registry_after_drop = registry.snapshot();
    for id in next_ids {
        registry.remove(id);
    }
    drop((first_sender, second_sender));
    let report = supervisor.join_all().await;
    assert!(report.errors.is_empty());
    assert!(registry.snapshot().is_empty());
    assert_eq!(registered_before_drop.len(), 2);
    assert_eq!(
        registered_before_drop[&first_ids[0]].task_name,
        "operator:a_asof"
    );
    assert_eq!(
        registered_before_drop[&first_ids[1]].task_name,
        "operator:c_select1"
    );
    assert_eq!(drivers_before_drop, 0);
    assert_eq!(settled_after_drop, 0);
    assert_eq!(charged_before_drop.queue_depth, 1);
    assert_eq!(charged_before_drop.charged_rows, 1);
    assert!(charged_before_drop.charged_bytes > 0);
    assert_eq!(
        (
            charged_after_drop.queue_depth,
            charged_after_drop.charged_rows,
            charged_after_drop.charged_bytes
        ),
        (0, 0, 0)
    );
    assert!(source_released);
    assert!(next_ids[0] > first_ids[1]);
    assert!(next_ids[1] > next_ids[0]);
    assert!(
        registered_after_drop.is_empty(),
        "dropping a bound but unspawned PreparedPair left reserved logical tasks"
    );
    assert!(next_registry_after_drop.is_empty());
}
