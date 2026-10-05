use super::*;
use crate::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CancellationToken, EdgeCollector,
    OperatorMetadata, StreamJobContext, StreamOperator,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use std::time::Duration;

fn operator(spec: Option<super::super::StreamAsofJoinSpec>) -> StreamAsofJoinOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = spec.unwrap_or_else(|| {
        super::super::StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(100, 1 << 20).unwrap(),
        )
        .unwrap()
    });
    StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap()
}

fn historical_snapshot(value: &str) -> (StreamAsofJoinOperator, OperatorStateSnapshot) {
    let value: serde_json::Value = serde_json::from_str(value).unwrap();
    let operator = operator(Some(serde_json::from_value(value["spec"].clone()).unwrap()));
    let segments = value["segments"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, segment)| {
            let encoded = segment["hex"].as_str().unwrap();
            let mut bytes = vec![0; encoded.len() / 2];
            hex::decode_to_slice(encoded, &mut bytes).unwrap();
            let decoded = StateSegment::new(bytes);
            assert_eq!(decoded.sha256(), segment["sha256"]);
            (name.clone(), decoded)
        })
        .collect();
    let snapshot = OperatorStateSnapshot {
        inline_metadata: serde_json::from_value(value["metadata"].clone()).unwrap(),
        segments,
    };
    assert_eq!(
        snapshot.inline_metadata["fingerprint"],
        operator.fingerprint
    );
    (operator, snapshot)
}

fn old_snapshots() -> [&'static str; 2] {
    [
        include_str!("../tests/fixtures/legacy-v3/populated.json"),
        include_str!("../tests/fixtures/legacy-v4/dominated-retained.json"),
    ]
}

fn is_version_mismatch(error: &CalcFlowError) -> bool {
    matches!(error, CalcFlowError::CheckpointMismatch { message }
        if message.contains("state version differs"))
}

#[test]
fn current_restore_rejects_layout_three_and_four_before_workspace() {
    let mut source = operator(None);
    let current = source.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        current.inline_metadata["layout_version"],
        serde_json::json!(10)
    );
    let results = [3, 4].map(|layout| {
        let mut snapshot = current.clone();
        snapshot
            .inline_metadata
            .insert("layout_version".into(), serde_json::json!(layout));
        snapshot
            .inline_metadata
            .insert("accounting_version".into(), serde_json::json!(layout));
        let mut target = operator(None);
        let pool = Arc::new(GreedyMemoryPool::new(0));
        target.runtime.pool = pool.clone();
        let before = target.status();
        let result = target.restore(&snapshot);
        let unchanged = target.status() == before
            && target.capture(Epoch::INITIAL).unwrap().inline_metadata == current.inline_metadata
            && pool.reserved() == 0;
        drop(target);
        assert_eq!(pool.reserved(), 0);
        (result, unchanged)
    });
    for (result, unchanged) in results {
        assert!(
            result.as_ref().is_err_and(is_version_mismatch),
            "{result:?}"
        );
        assert!(unchanged);
    }
}

#[tokio::test]
async fn historical_indexes_do_not_replace_live_current_state() {
    let mut results = Vec::new();
    for old in old_snapshots() {
        let (mut target, snapshot) = historical_snapshot(old);
        let record = RecordBatch::try_new(
            target.schemas[1].clone(),
            vec![
                Arc::new(StringArray::from(vec!["C"])),
                Arc::new(TimestampMicrosecondArray::from(vec![120]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![99])),
            ],
        )
        .unwrap();
        let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
        let job = StreamJobContext::new(
            1,
            "asof",
            crate::JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(target.output_ports().to_vec());
        target
            .process_data("right", input, &context, &mut output)
            .await
            .unwrap();
        let before = target.capture(Epoch::INITIAL).unwrap();
        assert_eq!(
            before.inline_metadata["layout_version"],
            serde_json::json!(10)
        );
        let status = target.status();
        let pool = target.runtime.pool.clone();
        let paid = pool.reserved();
        let sequence = target.next_output_sequence;
        let result = target.restore(&snapshot);
        let refunded = pool.reserved() == paid;
        let repeated = target.capture(Epoch::INITIAL).unwrap();
        let unchanged = target.status() == status
            && target.next_output_sequence == sequence
            && refunded
            && repeated.inline_metadata == before.inline_metadata
            && repeated.segments == before.segments;
        drop(target);
        drop(before);
        drop(repeated);
        assert_eq!(pool.reserved(), 0);
        results.push((result, unchanged));
    }
    for (result, unchanged) in results {
        assert!(
            result.as_ref().is_err_and(is_version_mismatch),
            "{result:?}"
        );
        assert!(unchanged);
    }
}

#[test]
fn historical_magic_is_rejected_before_payload_lookup() {
    let results = old_snapshots().map(|old| {
        let (operator, snapshot) = historical_snapshot(old);
        let layout = snapshot.inline_metadata["layout_version"].as_u64().unwrap();
        let segment = &snapshot.segments[&format!("asof-index-v{layout}")];
        let charge = index_v3::restore_charge(segment.bytes(), 100, 1 << 20);
        let decoded = index_v3::decode(
            segment,
            &BTreeMap::new(),
            100,
            1 << 20,
            operator.sequence_kinds(),
        );
        (charge, decoded.err())
    });
    for (charge, decoded) in results {
        assert!(
            matches!(charge, Err(CalcFlowError::CheckpointMismatch { message })
            if message == "ASOF index magic differs")
        );
        assert!(
            matches!(decoded, Some(CalcFlowError::CheckpointMismatch { message })
            if message == "ASOF index magic differs")
        );
    }
}
