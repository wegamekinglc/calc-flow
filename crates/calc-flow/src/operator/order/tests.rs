use super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};
use datafusion::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn operator(limit: u64) -> BoundedOrderOperator {
    BoundedOrderOperator::new(
        "order",
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
        vec![
            SortKey::new(
                "x",
                crate::SortDirection::Ascending,
                crate::NullPlacement::Last,
            )
            .unwrap(),
        ],
        Some(1),
        SortSpec::bounded_source(crate::StateBudget::new(4, limit).unwrap()).unwrap(),
    )
    .unwrap()
}
fn raw(op: &BoundedOrderOperator) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                op.input[0].schema().unwrap().clone(),
                vec![Arc::new(Int64Array::from(vec![3, 1]))],
            )
            .unwrap(),
        ],
        crate::BatchMetadata::default(),
    )
    .unwrap()
}
#[tokio::test]
async fn test_order_retained_candidate_container_stays_charged() {
    let mut op = operator(8 << 20);
    let declaration = op.pool.reserved();
    let job = StreamJobContext::new(1, "order", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "order", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", raw(&op), &context, &mut output)
        .await
        .unwrap();
    let state = op.stream.as_ref().unwrap();
    let committed = state.history.iter().map(|h| h.charge()).sum::<usize>();
    assert!(
        op.pool.reserved() >= declaration + 4096 + committed + 64,
        "retained history Vec/Arc storage must keep its reservation after admission: declaration={declaration}, H={committed}, actual={}",
        op.pool.reserved()
    );
}

#[tokio::test]
async fn test_order_owned_budget_boundaries() {
    // x:i64 uses declaration 25616, normalized H 9282; its certified backing
    // is 1105. Concat/take 2210 + row indices/scratch 64 + control/key roles
    // 16384 gives workspace 18658; compact output adds 9282 => peak 62838.
    for (limit, success) in [(62_838, true), (62_837, false)] {
        let mut op = operator(limit);
        let input = raw(&op);
        let run = crate::RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
        let result = BatchOperator::process(
            &mut op,
            &BTreeMap::from([("input".into(), input)]),
            &BatchOperatorContext { run: &run },
        )
        .await;
        assert_eq!(
            result.is_ok(),
            success,
            "finite peak {limit}: {:?}",
            result.err()
        );
        assert!(op.pool.take_peak() <= limit as usize);
        assert_eq!(op.pool.reserved(), 25_616);
    }
    for (limit, success) in [(432_676, true), (432_675, false)] {
        let mut op = operator(limit);
        let job = StreamJobContext::new(1, "order", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "order", None);
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        op.process_data("input", raw(&op), &context, &mut output)
            .await
            .unwrap();
        let result = op.checkpoint(crate::Epoch::INITIAL);
        assert_eq!(
            result.is_ok(),
            success,
            "capture peak {limit}: {:?}",
            result.err()
        );
        assert!(op.pool.take_peak() <= limit as usize);
        assert_eq!(op.stream.as_ref().unwrap().accepted, 2);
        assert!(!op.stream.as_ref().unwrap().closed);
    }
    // Six-digit budget identities reduce restore's prepaid control decode by
    // 32 bytes from the seven-digit probe (466322), giving this exact boundary.
    for (limit, success) in [(466_290, true), (466_289, false)] {
        let mut source = operator(limit);
        let job = StreamJobContext::new(1, "order", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "order", None);
        let mut output = EdgeCollector::new(source.output_ports().to_vec());
        source
            .process_data("input", raw(&source), &context, &mut output)
            .await
            .unwrap();
        let capture = source.checkpoint(crate::Epoch::INITIAL).unwrap();
        let mut target = operator(limit);
        let result = StreamOperator::restore(&mut target, &capture);
        assert_eq!(
            result.is_ok(),
            success,
            "restore peak {limit}: {:?}",
            result.err()
        );
        assert!(target.pool.take_peak() <= limit as usize);
        assert_eq!(
            target.stream.as_ref().unwrap().accepted,
            if success { 2 } else { 0 }
        );
        assert!(!target.stream.as_ref().unwrap().closed);
    }
}

#[test]
fn test_order_sequence_workspace_exact_boundary() {
    for (limit, success) in [(56_016, true), (56_015, false)] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("seq", DataType::Int64, false),
        ]));
        let op = BoundedOrderOperator::new(
            "order",
            schema.clone(),
            vec![
                SortKey::new(
                    "x",
                    crate::SortDirection::Ascending,
                    crate::NullPlacement::Last,
                )
                .unwrap(),
            ],
            Some(1),
            SortSpec::bounded_source(crate::StateBudget::new(4, limit).unwrap())
                .unwrap()
                .with_sequence_by(vec!["seq".into()])
                .unwrap(),
        )
        .unwrap();
        let raw = Batch::table(
            vec![
                RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(Int64Array::from(vec![3, 1])),
                        Arc::new(Int64Array::from(vec![10, 11])),
                    ],
                )
                .unwrap(),
            ],
            crate::BatchMetadata::default(),
        )
        .unwrap();
        let history = op.finite_input(&raw, &|| Ok(())).unwrap();
        let pool: Arc<dyn MemoryPool> = op.pool.clone();
        let result = op.validate_sequence(&history, &pool, &|| Ok(()));
        assert_eq!(result.is_ok(), success, "sequence peak {limit}: {result:?}");
        assert!(op.pool.take_peak() <= limit as usize);
    }
}

#[test]
fn test_order_compacts_declaration_capacity_before_retaining_it() {
    let mut sequence = String::with_capacity(4 << 20);
    sequence.push_str("seq");
    let mut columns = Vec::with_capacity(4096);
    columns.push(sequence);
    let spec = SortSpec::bounded_source(crate::StateBudget::new(4, 1 << 20).unwrap())
        .unwrap()
        .with_sequence_by(columns)
        .unwrap();
    let mut by = Vec::with_capacity(4096);
    by.push(
        SortKey::new(
            "x",
            crate::SortDirection::Ascending,
            crate::NullPlacement::Last,
        )
        .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::Int64, false),
    ]));
    let op = BoundedOrderOperator::new("order", schema, by, Some(1), spec).unwrap();
    assert!(
        op.by.capacity() < 64,
        "unused comparator capacity must not remain retained outside the declaration quota"
    );
    let SortTies::Sequence { columns } = op.spec.ties() else {
        panic!("sequence");
    };
    assert!(
        columns.capacity() < 64 && columns[0].capacity() < 64,
        "unused sequence capacity must not remain retained outside the declaration quota"
    );
}

#[test]
fn test_order_concat_string_offset_preflight_boundaries() {
    assert!(work::validate_string_capacity(&DataType::Utf8, 0).is_ok());
    assert!(work::validate_string_capacity(&DataType::Utf8, i32::MAX as usize).is_ok());
    assert!(work::validate_string_capacity(&DataType::Utf8, i32::MAX as usize + 1).is_err());
    assert!(work::validate_string_capacity(&DataType::LargeUtf8, i32::MAX as usize + 1).is_ok());
}
