use super::*;

fn integer_record(schema: &SchemaRef, start: i64, times: Vec<i64>) -> RecordBatch {
    let count = times.len();
    record(
        schema,
        Arc::new(Int64Array::from(vec![7; count])),
        Arc::new(Int64Array::from_iter_values(
            start..start + i64::try_from(count).unwrap(),
        )),
        times,
    )
}

fn check_descriptor_conversion(count: usize, aligned: bool) {
    let schema = schema(DataType::Int64, DataType::Int64);
    let operator = fixture(&schema);
    let records = vec![
        integer_record(&schema, 0, vec![10; 2_048]),
        integer_record(&schema, 2_048, vec![10; count - 2_048]),
    ];
    assert!(
        records
            .iter()
            .all(|record| can_share_batch(record).unwrap())
    );
    let input = batch(records);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    for index in [0, 1] {
        let side = ValidatedInput {
            index,
            watermark: None,
        };
        let workspace = operator.identity_workspace(&input, side).unwrap();
        let identities = operator
            .admission_identities(
                input.table_payload().unwrap().batches(),
                side,
                workspace.rows,
                &context,
            )
            .unwrap();
        let descriptor_bytes = identities.rows.len() * size_of::<(LeftOrder, AdmissionRef)>();
        if aligned {
            assert_eq!(
                identities.rows.capacity() * size_of::<InputRow<'_>>()
                    % size_of::<(LeftOrder, AdmissionRef)>(),
                0,
            );
        }
        let mut encoded = None;
        let allocation = allocation_counter::measure(|| {
            encoded = Some(
                encode_rows(
                    identities.rows,
                    index,
                    23,
                    operator.payload_header_bytes[index],
                    None,
                    "asof",
                )
                .unwrap(),
            );
        });
        let (rows, payloads) = encoded.unwrap();
        if aligned {
            eprintln!("side={index}, rows={count}, extra allocations={allocation:?}");
            assert!(
                allocation.bytes_max < u64::try_from(descriptor_bytes).unwrap(),
                "a second full descriptor backing was allocated: {allocation:?}"
            );
        }
        assert_eq!(rows.len(), count);
        assert_eq!(payloads.len(), 2);
        for (position, (identity, reference)) in rows.iter().enumerate() {
            assert_eq!(identity.0, 10);
            assert_eq!(reference.batch_index, position / 2_048);
            assert_eq!(reference.row as usize, position % 2_048);
            let payload = &payloads[reference.batch_index];
            assert_eq!(
                payload.key,
                (
                    u8::try_from(index).unwrap(),
                    23 + (position / 2_048 * 2_048) as u64
                )
            );
            assert_eq!(Arc::strong_count(payload), 1);
            let sequence = payload
                .record
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(
                sequence.value(reference.row as usize),
                i64::try_from(position).unwrap()
            );
        }
    }
}

#[test]
fn encode_rows_reuses_aligned_descriptor_allocation() {
    // Avoid a layout-fixup shrink: the test allocator implements realloc by
    // allocating and copying, even when the system allocator can shrink in place.
    let width = size_of::<(LeftOrder, AdmissionRef)>();
    check_descriptor_conversion(4_096 / width * width, true);
}

#[test]
fn encode_rows_preserves_unaligned_descriptor_layout() {
    check_descriptor_conversion(4_096, false);
}

#[test]
fn encode_rows_preserves_compact_references_across_discarded_batches() {
    let schema = schema(DataType::Int64, DataType::Int64);
    let operator = fixture(&schema);
    let input = batch(vec![
        integer_record(&schema, 0, vec![9, 10, 11, 12, 9, 10, 11, 12]),
        integer_record(&schema, 8, vec![9; 8]),
        integer_record(&schema, 16, vec![10; 8]),
    ]);
    let expected = [1, 2, 3, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    for index in [0, 1] {
        let side = ValidatedInput {
            index,
            watermark: Some(10),
        };
        let workspace = operator.identity_workspace(&input, side).unwrap();
        let identities = operator
            .admission_identities(
                input.table_payload().unwrap().batches(),
                side,
                workspace.rows,
                &context,
            )
            .unwrap();
        let originals = identities
            .rows
            .iter()
            .map(|(identity, _, _, key)| (identity.clone(), *key))
            .collect::<Vec<_>>();
        let (rows, payloads) = encode_rows(
            identities.rows,
            index,
            23,
            operator.payload_header_bytes[index],
            Some(&[2]),
            "asof",
        )
        .unwrap();
        assert_eq!(rows.len(), expected.len());
        assert_eq!(payloads.len(), 2);
        for (position, ((identity, reference), expected_sequence)) in
            rows.iter().zip(expected).enumerate()
        {
            assert_eq!(identity, &originals[position].0);
            assert_eq!(reference.key_index, originals[position].1);
            let (batch_index, ordinal, start) = if position < 6 {
                (0, position, 23)
            } else {
                (1, position - 6, 29)
            };
            assert_eq!(reference.batch_index, batch_index);
            assert_eq!(reference.row as usize, ordinal);
            let payload = &payloads[batch_index];
            assert_eq!(payload.key, (u8::try_from(index).unwrap(), start));
            assert_eq!(payload.record.num_columns(), 1);
            let sequence = payload
                .record
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(sequence.value(ordinal), expected_sequence);
        }
    }
}
