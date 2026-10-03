use super::*;
use checkpoint::cost::{self, EncodingCost};
use datafusion::arrow::compute::concat_batches;

fn operator() -> StreamAsofJoinOperator {
    let (template, _) = fixture();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(1),
        AsofStateLimits::new(300_000, 128 << 20).unwrap(),
    )
    .unwrap();
    StreamAsofJoinOperator::new(
        "asof",
        template.schemas[0].clone(),
        template.schemas[1].clone(),
        spec,
    )
    .unwrap()
}

fn batch(operator: &StreamAsofJoinOperator, rows: &[(&str, i64, i64)]) -> Batch {
    Batch::table(
        vec![indexed_input(&operator.schemas[0], rows)],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_a12_bulk_restore_finalization_rebases_within_workspace() {
    let mut live = operator();
    let rows = (0..65_537_i64)
        .map(|row| ("A", row, row))
        .collect::<Vec<_>>();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(live.output_ports().to_vec());
    live.process_data(
        "right",
        batch(&live, &rows[..65_536]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    let base = live.capture(Epoch::INITIAL).unwrap();
    live.process_data(
        "right",
        batch(&live, &rows[65_536..]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    let changed = live.capture(Epoch::new(2).unwrap()).unwrap();
    let mut restored = operator();
    restored.restore(&changed).unwrap();
    assert_eq!(restored.status(), live.status());
    restored
        .process_data("left", batch(&restored, &rows), &context, &mut output)
        .await
        .unwrap();
    restored.on_end(&context, &mut output).await.unwrap();
    assert_eq!(restored.status.matched_rows, rows.len() as u64);
    assert_eq!(restored.status.state_rows, 0);
    assert!(restored.checkpoint_log.force_base);
    assert!(restored.checkpoint_log.journal.is_empty());
    let records = output
        .drain("output")
        .into_iter()
        .filter_map(|message| message.as_data().cloned())
        .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
        .collect::<Vec<_>>();
    let combined = concat_batches(&restored.schemas[2], &records).unwrap();
    for name in ["left__seq", "right__seq"] {
        let sequence = combined
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(sequence.values(), &(0..65_537_i64).collect::<Vec<_>>());
    }
    let terminal = restored.capture(Epoch::new(3).unwrap()).unwrap();
    assert_eq!(
        terminal.inline_metadata["checkpoint_log"]["frames"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    assert!(terminal.segments.is_empty());
    let mut final_restore = operator();
    final_restore.restore(&terminal).unwrap();
    let repeated = final_restore.capture(Epoch::new(4).unwrap()).unwrap();
    assert_eq!(
        repeated.inline_metadata["metrics"],
        terminal.inline_metadata["metrics"]
    );
    drop(base);
    job.gather_owner().close_and_drain().await;
}

#[tokio::test]
async fn test_a12_bulk_admission_rebases_and_preserves_later_mutations() {
    let mut live = operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(live.output_ports().to_vec());
    let rows = (0..9_218_i64)
        .map(|row| ("A", row, row))
        .collect::<Vec<_>>();
    live.process_data("right", batch(&live, &rows[..1024]), &context, &mut output)
        .await
        .unwrap();
    let base = live.capture(Epoch::INITIAL).unwrap();
    live.process_data(
        "right",
        batch(&live, &rows[1024..9216]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    assert!(live.checkpoint_log.force_base);
    assert!(live.checkpoint_log.journal.is_empty());
    live.process_data("right", batch(&live, &rows[9216..]), &context, &mut output)
        .await
        .unwrap();
    assert!(live.checkpoint_log.force_base);
    let current = live.capture(Epoch::new(2).unwrap()).unwrap();
    assert!(
        current.inline_metadata["checkpoint_log"]["generation"]
            .as_u64()
            .unwrap()
            > base.inline_metadata["checkpoint_log"]["generation"]
                .as_u64()
                .unwrap()
    );
    assert_eq!(
        current.inline_metadata["checkpoint_log"]["frames"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let mut restored = operator();
    restored.restore(&current).unwrap();
    assert_eq!(restored.status(), live.status());
    assert_eq!(restored.status.retained_right_rows, rows.len() as u64);
}

async fn single_hot_key_delta(rows: usize) {
    let mut operator = operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let initial = (0..rows)
        .map(|row| {
            (
                "A",
                i64::try_from(row).unwrap(),
                i64::try_from(row).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    operator
        .process_data("right", batch(&operator, &initial), &context, &mut output)
        .await
        .unwrap();
    cost::take();
    let base = operator.capture(Epoch::INITIAL).unwrap();
    let initial_cost = cost::take();
    assert_eq!(initial_cost.index_rows, rows);
    assert_eq!(initial_cost.ipc_rows, rows);
    assert!(initial_cost.index_bytes > 4_096, "{initial_cost:?}");
    assert!(initial_cost.ipc_bytes > 0, "{initial_cost:?}");
    operator
        .process_data(
            "right",
            batch(
                &operator,
                &[(
                    "A",
                    i64::try_from(rows).unwrap(),
                    i64::try_from(rows).unwrap(),
                )],
            ),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    cost::take();
    let changed = operator.capture(Epoch::new(2).unwrap()).unwrap();
    let changed_cost = cost::take();
    assert_eq!(operator.status.retained_right_rows, rows as u64 + 1);
    assert_ne!(base.inline_metadata, changed.inline_metadata);
    eprintln!("A12 retained={rows} base={initial_cost:?} changed={changed_cost:?}");
    assert_eq!(changed_cost.ipc_rows, 1);
    assert_eq!(
        changed_cost.index_rows, 1,
        "unchanged hot-key rows were encoded"
    );
    assert!(changed_cost.index_bytes <= 4_096, "{changed_cost:?}");
}

#[tokio::test(flavor = "current_thread")]
async fn test_a12_single_row_hot_key_checkpoint_4096_encodes_only_dirty_row() {
    single_hot_key_delta(4_096).await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_a12_single_row_hot_key_checkpoint_65536_encodes_only_dirty_row() {
    single_hot_key_delta(65_536).await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_a12_unchanged_checkpoint_reuses_index_and_ipc_without_encoding() {
    let mut operator = operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            batch(&operator, &[("A", 0, 1)]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let initial = operator.capture(Epoch::INITIAL).unwrap();
    cost::take();
    let unchanged = operator.capture(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(cost::take(), EncodingCost::default());
    assert_eq!(initial.segments, unchanged.segments);
    assert_eq!(
        initial.inline_metadata["metrics"],
        unchanged.inline_metadata["metrics"]
    );
    assert_eq!(unchanged.inline_metadata["epoch"], serde_json::json!(2));
}

fn expected_output(schema: SchemaRef, matched: usize) -> RecordBatch {
    let left_keys = std::iter::once("B").chain(std::iter::repeat_n("A", matched));
    let left_times =
        std::iter::once(-1).chain((0..matched).map(|row| i64::try_from(row).unwrap() * 2 + 1));
    let left_sequences =
        std::iter::once(-1).chain((0..matched).map(|row| i64::try_from(row).unwrap() + 1));
    let right_keys = std::iter::once(None).chain(std::iter::repeat_n(Some("A"), matched));
    let right_times =
        std::iter::once(None).chain((0..matched).map(|row| Some(i64::try_from(row).unwrap() * 2)));
    let right_sequences = std::iter::once(None)
        .chain((0..matched).map(|row| Some(i64::try_from(row).unwrap() + 1_000)));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(left_keys)),
            Arc::new(TimestampMicrosecondArray::from_iter_values(left_times).with_timezone("UTC")),
            Arc::new(Int64Array::from_iter_values(left_sequences)),
            Arc::new(right_keys.collect::<StringArray>()),
            Arc::new(
                right_times
                    .collect::<TimestampMicrosecondArray>()
                    .with_timezone("UTC"),
            ),
            Arc::new(right_sequences.collect::<Int64Array>()),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_a12_boundary_admission_keeps_retired_dirty_key_credit() {
    use checkpoint::index_v3::log::journal::{Change, Identity, Journal, Version};
    let mut operator = operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            batch(&operator, &[("A", 0, 0)]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    operator.capture(Epoch::INITIAL).unwrap();
    let key = state::Encoding::from_slice(&vec![b'k'; 65_536]);
    let allocation = key.allocation().unwrap().1;
    operator.checkpoint_log.journal = Journal::default()
        .prepare(
            &[Change {
                identity: Identity::Right((0, key, state::Encoding::from_slice(b"s"))),
                before: Some(Version::Right {
                    tag: 2,
                    payload: None,
                }),
                after: None,
            }],
            |_| false,
            |bytes| operator.reserve_workspace(bytes),
            "asof",
        )
        .unwrap();
    let input = admission::ValidatedInput {
        index: 1,
        watermark: None,
    };
    let next = operator
        .prepare_admission(input, &batch(&operator, &[("B", 1, 1)]), &context)
        .await
        .unwrap();
    let (_, _, owners) = operator.checked_capacity_admission(&next).unwrap();
    let journal = operator.prepare_log_admission(&next, 1, &owners).unwrap();
    assert!(
        journal.bytes() >= allocation,
        "retired key lease was lost: {} < {allocation}",
        journal.bytes()
    );
}

fn assert_delta_rejected(
    operator: &StreamAsofJoinOperator,
    baseline: &crate::OperatorStateSnapshot,
    segment: &crate::StateSegment,
    kinds: [state::SequenceKind; 2],
) {
    use checkpoint::index_v3::log;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let frame = log::chain::decode(
        baseline
            .segments
            .iter()
            .find(|(name, _)| name.starts_with("asof-log-v9-"))
            .unwrap()
            .1,
        &operator.fingerprint,
    )
    .unwrap();
    let batches = operator
        .state
        .batches
        .iter()
        .map(|(key, (batch, _))| (*key, batch.clone()))
        .collect();
    let (_, reader) = checkpoint::index_v3::decode_registered_bytes(
        frame.body,
        &batches,
        300_000,
        128 << 20,
        operator.sequence_kinds(),
    )
    .unwrap();
    let charge =
        log::restore_charge(segment.bytes(), reader.count(), kinds, 300_000, 128 << 20).unwrap();
    let reserved = operator.runtime.pool.reserved();
    let result = catch_unwind(AssertUnwindSafe(|| {
        log::decode(
            segment.bytes(),
            &reader,
            &BTreeMap::new(),
            kinds,
            &AsofStateLimits::new(300_000, 128 << 20).unwrap(),
            operator.reserve_workspace(charge).unwrap(),
            || Ok(()),
        )
    }));
    assert!(matches!(
        result,
        Ok(Err(CalcFlowError::CheckpointMismatch { .. }))
    ));
    assert_eq!(operator.runtime.pool.reserved(), reserved);
}

#[tokio::test]
async fn test_a12_boundary_replay_rejects_left_suffix_growth_without_panicking() {
    use checkpoint::index_v3::log::{
        self,
        journal::{Change, Identity, Version},
    };
    let mut operator = operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            batch(&operator, &[("A", 0, 0), ("A", 1, 1)]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let (key, data, head) = operator
        .state
        .left
        .checkpoint_chunks(&operator.state.batches)
        .next()
        .unwrap();
    let before = Version::Left {
        rows: (data.sequences.len() - head) as u64,
        capacities: data.checkpoint_capacities(),
    };
    let mut capacities = data.checkpoint_capacities();
    for index in [0, 4, 5] {
        capacities[index] = capacities[index].max(3);
    }
    let changes = [Change {
        identity: Identity::Left(key),
        before: Some(before),
        after: Some(Version::Left {
            rows: 3,
            capacities,
        }),
    }];
    let input = log::Input {
        capacities: log::model::capacities(&operator.state),
        counts: log::model::counts(&operator.state),
        kinds: operator.sequence_kinds(),
        changes: &changes,
        left: &[],
        buckets: &[],
    };
    let owners = checkpoint::index_v3::source_owners(&operator.state);
    let encoded = log::encode(
        &input,
        &owners,
        |_| true,
        |bytes| operator.reserve_workspace(bytes),
        1 << 20,
        "asof",
    )
    .unwrap();
    let baseline = operator.capture(Epoch::INITIAL).unwrap();
    assert_delta_rejected(&operator, &baseline, &encoded.segment, input.kinds);
}

#[tokio::test(flavor = "current_thread")]
async fn test_a12_every_cut_through_33_deltas_and_compaction_preserves_full_values_and_status() {
    let mut live = operator();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(live.output_ports().to_vec());
    live.process_data(
        "left",
        batch(&live, &[("B", -1, -1)]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    let mut cuts = Vec::new();
    for cut in 0..=33 {
        for (side, row) in [
            ("right", ("A", cut * 2, cut + 1_000)),
            ("left", ("A", cut * 2 + 1, cut + 1)),
        ] {
            live.process_data(side, batch(&live, &[row]), &context, &mut output)
                .await
                .unwrap();
        }
        cuts.push((
            live.capture(Epoch::new(u64::try_from(cut).unwrap() + 1).unwrap())
                .unwrap(),
            live.status(),
        ));
    }
    assert!(output.drain("output").is_empty());
    for (cut, (snapshot, before)) in cuts.iter().enumerate() {
        let mut restored = operator();
        restored.restore(snapshot).unwrap();
        assert_eq!(
            restored.status.state_bytes,
            restored
                .current_inventory(restored.prepared.as_ref())
                .unwrap()
                .bytes
        );
        assert_eq!(&restored.status(), before);
        assert_eq!(restored.next_output_sequence, 0);
        assert!(!restored.terminal);
        let mut invalid = snapshot.clone();
        invalid.inline_metadata.get_mut("metrics").unwrap()["state_bytes"] =
            serde_json::json!(before.state_bytes - 1);
        let before_rejection = restored.status();
        let reserved = restored.runtime.pool.reserved();
        assert!(restored.restore(&invalid).is_err());
        assert_eq!(restored.status(), before_rejection);
        assert_eq!(restored.runtime.pool.reserved(), reserved);
        let mut recovered_output = EdgeCollector::new(restored.output_ports().to_vec());
        restored
            .on_end(&context, &mut recovered_output)
            .await
            .unwrap();
        let messages = recovered_output.drain("output");
        let mut records = Vec::new();
        let mut sequence = 0;
        for message in &messages {
            let batch = message.as_data().unwrap();
            assert_eq!(
                batch.metadata(),
                &BatchMetadata::new("asof", sequence, JsonMap::new()).unwrap()
            );
            records.extend(batch.table_payload().unwrap().batches().iter().cloned());
            sequence += u64::try_from(batch.num_rows()).unwrap();
        }
        let actual = concat_batches(&restored.schemas[2], &records).unwrap();
        assert_eq!(
            actual,
            expected_output(restored.schemas[2].clone(), cut + 1)
        );
        assert_eq!(restored.status.matched_rows, cut as u64 + 1);
        assert_eq!(restored.status.unmatched_rows, 1);
        assert_eq!(restored.status.emitted_left_rows, cut as u64 + 2);
        assert_eq!(restored.status.state_rows, 0);
        assert_eq!(restored.status.state_bytes, 0);
        assert!(restored.terminal);
        assert_eq!(restored.next_output_sequence, sequence);
    }
}
