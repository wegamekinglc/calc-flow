use std::{collections::BTreeMap, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, Int64Array, StringArray},
        datatypes::{DataType, Field, FieldRef, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
    execution::memory_pool::{MemoryPool, MemoryReservation},
};
use serde_json::{Value, json};

use super::super::{
    SqlOperator, decode_sql_state,
    incremental::{
        self, IncrementalSql,
        compact_state::{NativeAggregateInput, NativeStateDescriptor, PaidNativeStateRecords},
    },
    ipc, metadata, retention,
};
use super::control::{
    self, CompactControl, CompactIdentity, DecodedControl, PaidControl, QuotaLedger, SegmentDigests,
};
use crate::{
    Batch, BatchKind, BatchMetadata, CancellationToken, DataFusionConfig, DataFusionRuntime,
    EdgeCollector, JsonMap, StreamJobContext, StreamOperatorContext,
    operator::{OperatorMetadata, OperatorStateSnapshot, Port, StateBudget, StreamOperator},
};

const INTEGER: &str = "SELECT MAX(value) AS hi, key, SUM(value) AS total, COUNT(value) AS valid, MIN(value) AS lo, COUNT(*) AS rows, COUNT(1) AS repeated, SUM(value) AS again FROM events GROUP BY key";
const SCALAR_AVG: &str = "SELECT AVG(value) AS mean, SUM(value) AS total, COUNT(value) AS valid, MIN(value) AS lo, MAX(value) AS hi, COUNT(*) AS rows, COUNT(1) AS repeated FROM events";
const GROUPED_AVG: &str = "SELECT AVG(value) AS mean, key, SUM(value) AS total, COUNT(value) AS valid, MIN(value) AS lo, MAX(value) AS hi, COUNT(*) AS rows, COUNT(1) AS repeated FROM events GROUP BY key";

struct ValidFixture {
    target: SqlOperator,
    snapshot: OperatorStateSnapshot,
    next: Batch,
    expected: Batch,
    expected_states: PaidNativeStateRecords,
    rows: u64,
    bytes: u64,
    source_pool: Arc<dyn MemoryPool>,
    target_pool: Arc<dyn MemoryPool>,
    owners: FixtureOwners,
}

struct FixtureOwners {
    _source: SqlOperator,
    export: PaidNativeStateRecords,
    _state: Arc<ipc::SqlInputSegment>,
    _control: PaidControl,
    _metadata: Arc<metadata::SqlMetadata>,
    _projection: Arc<retention::SqlProjection>,
    _trusted_projection: Arc<retention::SqlProjection>,
    _decoded: DecodedControl,
    _decoded_metadata: Arc<metadata::SqlMetadata>,
    _descriptor: NativeStateDescriptor,
    _identity_credit: MemoryReservation,
    _trusted_identity_credit: MemoryReservation,
    _decode_credit: MemoryReservation,
}

#[tokio::test]
async fn test_layout3_state_only_fixture_native_controls_are_valid() {
    for (query, kind) in cases() {
        let fixture = valid_fixture(query, kind).await;
        let source_pool = fixture.source_pool.clone();
        let target_pool = fixture.target_pool.clone();
        assert!(source_pool.reserved() > 0);
        assert!(target_pool.reserved() > 0);
        drop(fixture);
        assert_eq!(source_pool.reserved(), 0);
        assert_eq!(target_pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_sql_restore_accepts_valid_layout3_state_only_checkpoint() {
    let mut fixtures = Vec::new();
    for (query, kind) in cases() {
        fixtures.push(valid_fixture(query, kind).await);
    }
    for mut fixture in fixtures {
        fixture
            .target
            .restore(&fixture.snapshot)
            .expect("validated state-only layout3 checkpoint must restore");
        let before = fixture
            .target
            .incremental
            .as_ref()
            .expect("restored actual native plan")
            .export_native_state("layout3", || Ok(()))
            .unwrap();
        assert_eq!(before.records(), fixture.owners.export.records());
        assert!(
            fixture
                .target
                .set_state_budget(StateBudget::new(fixture.rows - 1, fixture.bytes).unwrap())
                .is_err()
        );
        fixture
            .target
            .set_state_budget(
                StateBudget::new(
                    fixture.rows + fixture.next.num_rows() as u64,
                    fixture.bytes + fixture.next.estimated_bytes().unwrap() as u64 + (1 << 20),
                )
                .unwrap(),
            )
            .unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "layout3", None);
        let actual = process(&mut fixture.target, fixture.next.clone(), &context).await;
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            fixture.expected.table_payload().unwrap().schema()
        );
        assert_eq!(rows(&actual), rows(&fixture.expected));
        assert_eq!(actual.metadata(), fixture.expected.metadata());
        let after = fixture
            .target
            .incremental
            .as_ref()
            .unwrap()
            .export_native_state("layout3", || Ok(()))
            .unwrap();
        assert_eq!(after.records(), fixture.expected_states.records());
    }
}

fn cases() -> [(&'static str, DataType); 3] {
    [
        (INTEGER, DataType::Int64),
        (SCALAR_AVG, DataType::Decimal128(20, 2)),
        (GROUPED_AVG, DataType::Decimal128(20, 2)),
    ]
}

struct CapturedFixture {
    snapshot: OperatorStateSnapshot,
    declared: SchemaRef,
    prefix: Batch,
    ledger: QuotaLedger,
    latest: BatchMetadata,
    projection: Arc<retention::SqlProjection>,
    export: PaidNativeStateRecords,
    encoded_state: Arc<ipc::SqlInputSegment>,
    encoded_metadata: Arc<metadata::SqlMetadata>,
    encoded_control: PaidControl,
    identity_credit: MemoryReservation,
    source_pool: Arc<dyn MemoryPool>,
    _wire: Batch,
    _control: CompactControl,
}

struct TrustedFixture {
    projection: Arc<retention::SqlProjection>,
    descriptor: NativeStateDescriptor,
    decoded: DecodedControl,
    identity_credit: MemoryReservation,
    target_pool: Arc<dyn MemoryPool>,
    _identity: CompactIdentity,
    _altered: AlteredBinding,
}

struct AlteredBinding {
    _projection: Arc<retention::SqlProjection>,
    _plan: IncrementalSql,
    _descriptor: NativeStateDescriptor,
    _identity: CompactIdentity,
}

struct DecodedFixture {
    state: Batch,
    metadata: BatchMetadata,
    metadata_owner: Arc<metadata::SqlMetadata>,
    credit: MemoryReservation,
}

struct ContinuedFixture {
    next: Batch,
    reconstructed: Batch,
    state: PaidNativeStateRecords,
    _imported: PaidNativeStateRecords,
    _rebuilt: Batch,
    _physical_next: Batch,
    _empty: Batch,
}

async fn valid_fixture(query: &'static str, kind: DataType) -> ValidFixture {
    let config = DataFusionConfig {
        batch_size: 257,
        target_partitions: 2,
        enable_rolling_rewrite: true,
        ..DataFusionConfig::default()
    };
    let job = job();
    let context = StreamOperatorContext::new(&job, "layout3", None);
    let (mut source, mut target, captured) = capture_fixture(query, &kind, config, &context).await;
    let (fresh, trusted) = trusted_fixture(&mut target, config, &captured);
    let decoded = decode_fixture(&target, &captured, &trusted);
    let restored = fresh
        .import_native_state(
            decoded.state.table_payload().unwrap().batches(),
            trusted.decoded.value.ledger.rows,
            trusted.decoded.value.ledger.seen_input,
            || Ok(()),
            "layout3",
        )
        .unwrap();
    let continued = continue_fixture(
        restored,
        &captured,
        &trusted.projection,
        decoded.metadata,
        &kind,
        &context,
    )
    .await;
    let expected = process(&mut source, continued.next.clone(), &context).await;
    finish_fixture(
        source,
        target,
        captured,
        trusted,
        (decoded.metadata_owner, decoded.credit),
        continued,
        expected,
    )
}

async fn capture_fixture(
    query: &str,
    kind: &DataType,
    config: DataFusionConfig,
    context: &StreamOperatorContext<'_>,
) -> (SqlOperator, SqlOperator, CapturedFixture) {
    let mut source = operator(query, kind, config);
    let target = operator(query, kind, config);
    let first = input(
        kind,
        &[Some(0), Some(1), None],
        &[Some(300), None, Some(700)],
        0,
    );
    let second = input(kind, &[Some(0), Some(1), None], &[Some(500), None, None], 1);
    process(&mut source, first, context).await;
    let prefix = process(&mut source, second, context).await;
    let captured = capture_state(&source, &target, config, prefix);
    (source, target, captured)
}

fn capture_state(
    source: &SqlOperator,
    target: &SqlOperator,
    config: DataFusionConfig,
    prefix: Batch,
) -> CapturedFixture {
    let declared = target.input_ports[0].schema().unwrap().clone();
    let runtime = source.retention_runtime().unwrap();
    let source_pool = pool(runtime);
    let state = source.compact.as_ref().unwrap();
    assert!(source.retained.is_none());
    let ledger = state.ledger;
    assert_eq!(ledger.rows, 6);
    assert!(ledger.bytes > 0);
    let latest = state.metadata.clone();
    let projection = state.projection().unwrap();
    let export = source
        .incremental
        .as_ref()
        .unwrap()
        .export_native_state("layout3", || Ok(()))
        .unwrap();
    assert_eq!(
        export.descriptor.group_count,
        if source.query == SCALAR_AVG { 1 } else { 3 }
    );
    assert_eq!(
        export.descriptor.wire_schema.fields().len(),
        export.descriptor.key_fields.len()
            + export
                .descriptor
                .state_fields
                .iter()
                .map(Vec::len)
                .sum::<usize>()
    );
    let capture_identity_credit = identity_credit(runtime, &declared);
    let native = native_json(&export.descriptor);
    assert!(native.get("group_count").is_none());
    assert!(native.get("rows").is_none());
    assert!(native.get("bytes").is_none());
    let wire = Batch::table(export.records().to_vec(), BatchMetadata::default()).unwrap();
    let encoded_state =
        ipc::encode(&wire, runtime.incremental_reservation("layout3"), || Ok(())).unwrap();
    let encoded_metadata = metadata::encode(
        &latest,
        metadata::reserve(runtime, &latest, "layout3").unwrap(),
        || Ok(()),
    )
    .unwrap();
    let identity = fixture_identity(source, config, &projection, &export.descriptor, native);
    let control = CompactControl {
        coalescer: incremental::global_record::coalescer::Inventory::None,
        state_layout: 3,
        state_accounting: 3,
        native_semantics: 1,
        datafusion_version: "54.0.0".into(),
        state_policy: incremental::grouped_float::Policy::ExactNumericV1,
        identity,
        ledger,
        groups: export.descriptor.group_count as u64,
        group_log: super::log::LogDescriptor {
            version: 1,
            base_generation: 0,
            generation: 0,
            base_groups: export.descriptor.group_count as u64,
            base_ledger: ledger,
            frames: Vec::new(),
        },
        segments: SegmentDigests {
            logical_schema: projection.logical_segment.sha256().into(),
            group_state: encoded_state.segment.sha256().into(),
            batch_metadata: encoded_metadata.segment.sha256().into(),
        },
    };
    let (snapshot, encoded_control) = capture_snapshot(
        runtime,
        &control,
        &projection,
        &encoded_state,
        &encoded_metadata,
    );
    assert_fixture_inventory(&snapshot);
    CapturedFixture {
        snapshot,
        declared,
        prefix,
        ledger,
        latest,
        projection,
        export,
        encoded_state,
        encoded_metadata,
        encoded_control,
        identity_credit: capture_identity_credit,
        source_pool,
        _wire: wire,
        _control: control,
    }
}

fn assert_fixture_inventory(snapshot: &OperatorStateSnapshot) {
    assert_eq!(snapshot.segments.len(), 4);
    assert_eq!(snapshot.inline_metadata.len(), 6);
    assert_eq!(
        snapshot
            .inline_metadata
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec![
            "bytes",
            "control_sha256",
            "query_sha256",
            "rows",
            "state_accounting",
            "state_layout"
        ]
    );
}

fn capture_snapshot(
    runtime: &DataFusionRuntime,
    control: &CompactControl,
    projection: &retention::SqlProjection,
    encoded_state: &ipc::SqlInputSegment,
    encoded_metadata: &metadata::SqlMetadata,
) -> (OperatorStateSnapshot, PaidControl) {
    let encoded_control = control::encode(runtime, control, "layout3", &|| Ok(())).unwrap();
    let snapshot = OperatorStateSnapshot {
        inline_metadata: control.inline_metadata(&encoded_control.segment),
        segments: BTreeMap::from([
            ("control".into(), encoded_control.segment.clone()),
            ("group-state".into(), encoded_state.segment.clone()),
            ("logical-schema".into(), projection.logical_segment.clone()),
            ("batch-metadata".into(), encoded_metadata.segment.clone()),
        ]),
    };
    (snapshot, encoded_control)
}

fn fixture_identity(
    operator: &SqlOperator,
    config: DataFusionConfig,
    projection: &retention::SqlProjection,
    descriptor: &NativeStateDescriptor,
    native_descriptor: Value,
) -> CompactIdentity {
    CompactIdentity {
        query_sha256: operator.query_digest(),
        input_alias: "events".into(),
        runtime_config: config,
        logical_schema_sha256: projection.logical_segment.sha256().into(),
        physical_schema_sha256: schema_digest(projection.columns.physical_schema()),
        retained_ordinals: projection.columns.ordinals().to_vec(),
        state_schema_sha256: schema_digest(&descriptor.wire_schema),
        output_schema_sha256: schema_digest(&descriptor.output_schema),
        native_descriptor,
    }
}

fn trusted_fixture(
    target: &mut SqlOperator,
    config: DataFusionConfig,
    captured: &CapturedFixture,
) -> (IncrementalSql, TrustedFixture) {
    let declared = &captured.declared;
    let snapshot = &captured.snapshot;
    let ledger = captured.ledger;
    target.stream_state.runtime().unwrap();
    let trusted_runtime = target.retention_runtime().unwrap();
    let target_pool = pool(trusted_runtime);
    let trusted_projection = retention::SqlProjection::resolve(
        trusted_runtime,
        &target.validated,
        "events",
        declared.clone(),
        "layout3",
    )
    .unwrap()
    .unwrap();
    let fresh = IncrementalSql::plan_sync(
        trusted_runtime,
        &target.validated,
        "events",
        declared.clone(),
        trusted_projection.columns.physical_schema().clone(),
        "layout3",
    )
    .unwrap()
    .unwrap();
    let descriptor = fresh.native_descriptor("layout3").unwrap();
    assert_eq!(descriptor.group_count, 0);
    let trusted_identity_credit = identity_credit(trusted_runtime, declared);
    let trusted = fixture_identity(
        target,
        config,
        &trusted_projection,
        &descriptor,
        native_json(&descriptor),
    );
    let decoded = control::decode(
        trusted_runtime,
        &snapshot.segments["control"],
        "layout3",
        &|| Ok(()),
    )
    .unwrap();
    decoded.value.validate_identity(&trusted).unwrap();
    let altered = validate_altered_binding(target, config, declared, &trusted, &decoded);
    decoded
        .value
        .validate_inline(&snapshot.inline_metadata, &snapshot.segments["control"])
        .unwrap();
    decoded
        .value
        .validate_segments(
            &snapshot.segments["logical-schema"],
            &snapshot.segments["group-state"],
            &snapshot.segments["batch-metadata"],
        )
        .unwrap();
    assert_eq!(decoded.value.ledger, ledger);
    (
        fresh,
        TrustedFixture {
            projection: trusted_projection,
            descriptor,
            decoded,
            identity_credit: trusted_identity_credit,
            target_pool,
            _identity: trusted,
            _altered: altered,
        },
    )
}

fn validate_altered_binding(
    target: &SqlOperator,
    config: DataFusionConfig,
    declared: &SchemaRef,
    trusted: &CompactIdentity,
    decoded: &DecodedControl,
) -> AlteredBinding {
    let trusted_runtime = target.retention_runtime().unwrap();
    let mut altered_fields = declared
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    altered_fields[1] = altered_fields[1].clone().with_data_type(DataType::Binary);
    let altered_declared = Arc::new(Schema::new_with_metadata(
        altered_fields,
        declared.metadata().clone(),
    ));
    let altered_projection = retention::SqlProjection::resolve(
        trusted_runtime,
        &target.validated,
        "events",
        altered_declared.clone(),
        "layout3",
    )
    .unwrap()
    .unwrap();
    let altered_plan = IncrementalSql::plan_sync(
        trusted_runtime,
        &target.validated,
        "events",
        altered_declared,
        altered_projection.columns.physical_schema().clone(),
        "layout3",
    )
    .unwrap()
    .unwrap();
    let altered_descriptor = altered_plan.native_descriptor("layout3").unwrap();
    let altered_identity = fixture_identity(
        target,
        config,
        &altered_projection,
        &altered_descriptor,
        native_json(&altered_descriptor),
    );
    assert_eq!(
        altered_identity.native_descriptor,
        trusted.native_descriptor
    );
    assert_eq!(
        altered_identity.physical_schema_sha256,
        trusted.physical_schema_sha256
    );
    assert!(decoded.value.validate_identity(&altered_identity).is_err());
    AlteredBinding {
        _projection: altered_projection,
        _plan: altered_plan,
        _descriptor: altered_descriptor,
        _identity: altered_identity,
    }
}

fn decode_fixture(
    target: &SqlOperator,
    captured: &CapturedFixture,
    trusted: &TrustedFixture,
) -> DecodedFixture {
    let trusted_runtime = target.retention_runtime().unwrap();
    let snapshot = &captured.snapshot;
    let descriptor = &trusted.descriptor;
    let decoded = &trusted.decoded;
    let declared = &captured.declared;
    let export = &captured.export;
    let latest = &captured.latest;
    let decode_credit = trusted_runtime.incremental_reservation("layout3-fixture-decode");
    incremental::ensure_reservation(
        &decode_credit,
        incremental::checked_bytes(
            8192,
            [
                (snapshot.segments["logical-schema"].bytes().len(), 16),
                (snapshot.segments["group-state"].bytes().len(), 16),
            ],
            "layout3",
        )
        .unwrap(),
        "layout3",
    )
    .unwrap();
    assert_eq!(
        retention::decode_schema(&snapshot.segments["logical-schema"]).unwrap(),
        declared.clone()
    );
    let decoded_state = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
    assert_eq!(
        decoded_state.table_payload().unwrap().schema(),
        &descriptor.wire_schema
    );
    assert_eq!(decoded_state.num_rows() as u64, decoded.value.groups);
    assert_eq!(
        decoded_state.table_payload().unwrap().batches(),
        export.records()
    );
    let (decoded_metadata, metadata_owner) = metadata::decode(
        trusted_runtime,
        &snapshot.segments["batch-metadata"],
        "layout3",
    )
    .unwrap();
    assert_eq!(&decoded_metadata, latest);
    DecodedFixture {
        state: decoded_state,
        metadata: decoded_metadata,
        metadata_owner,
        credit: decode_credit,
    }
}

async fn continue_fixture(
    mut restored: IncrementalSql,
    captured: &CapturedFixture,
    trusted_projection: &retention::SqlProjection,
    decoded_metadata: BatchMetadata,
    kind: &DataType,
    context: &StreamOperatorContext<'_>,
) -> ContinuedFixture {
    let export = &captured.export;
    let prefix = &captured.prefix;
    let latest = &captured.latest;
    let imported = restored.export_native_state("layout3", || Ok(())).unwrap();
    assert_eq!(imported.records(), export.records());
    let empty = Batch::table(
        vec![RecordBatch::new_empty(
            trusted_projection.columns.physical_schema().clone(),
        )],
        latest.clone(),
    )
    .unwrap();
    let preview = restored.update(&empty, context, "layout3").await.unwrap();
    assert_eq!(
        preview
            .records
            .iter()
            .map(RecordBatch::schema)
            .collect::<Vec<_>>(),
        prefix
            .table_payload()
            .unwrap()
            .batches()
            .iter()
            .map(RecordBatch::schema)
            .collect::<Vec<_>>()
    );
    let rebuilt = Batch::table(preview.records.clone(), decoded_metadata).unwrap();
    assert_eq!(rows(&rebuilt), rows(prefix));
    assert_eq!(rebuilt.metadata(), prefix.metadata());
    restored.commit(preview);
    let next = input(
        kind,
        &[Some(0), Some(1), None, Some(2)],
        &[Some(400), Some(800), Some(1200), Some(1500)],
        2,
    );
    let physical_next = Batch::table(
        next.table_payload()
            .unwrap()
            .batches()
            .iter()
            .map(|record| trusted_projection.columns.project(record).unwrap())
            .collect(),
        next.metadata().clone(),
    )
    .unwrap();
    let continued = restored
        .update(&physical_next, context, "layout3")
        .await
        .unwrap();
    let reconstructed = Batch::table(continued.records.clone(), next.metadata().clone()).unwrap();
    restored.commit(continued);
    let continuation_state = restored.export_native_state("layout3", || Ok(())).unwrap();
    drop(restored);
    ContinuedFixture {
        next,
        reconstructed,
        state: continuation_state,
        _imported: imported,
        _rebuilt: rebuilt,
        _physical_next: physical_next,
        _empty: empty,
    }
}

fn finish_fixture(
    source: SqlOperator,
    target: SqlOperator,
    captured: CapturedFixture,
    trusted: TrustedFixture,
    decoded_owners: (Arc<metadata::SqlMetadata>, MemoryReservation),
    continued: ContinuedFixture,
    expected: Batch,
) -> ValidFixture {
    assert_eq!(
        continued.reconstructed.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(&continued.reconstructed), rows(&expected));
    assert_eq!(continued.reconstructed.metadata(), expected.metadata());
    let expected_states = source
        .incremental
        .as_ref()
        .unwrap()
        .export_native_state("layout3", || Ok(()))
        .unwrap();
    assert_eq!(continued.state.records(), expected_states.records());
    eprintln!("valid layout3 native fixture: {}", source.query);
    ValidFixture {
        target,
        snapshot: captured.snapshot,
        next: continued.next,
        expected,
        expected_states,
        rows: captured.ledger.rows,
        bytes: captured.ledger.bytes,
        source_pool: captured.source_pool,
        target_pool: trusted.target_pool,
        owners: FixtureOwners {
            _source: source,
            export: captured.export,
            _state: captured.encoded_state,
            _control: captured.encoded_control,
            _metadata: captured.encoded_metadata,
            _projection: captured.projection,
            _trusted_projection: trusted.projection,
            _decoded: trusted.decoded,
            _decoded_metadata: decoded_owners.0,
            _descriptor: trusted.descriptor,
            _identity_credit: captured.identity_credit,
            _trusted_identity_credit: trusted.identity_credit,
            _decode_credit: decoded_owners.1,
        },
    }
}

fn operator(query: &str, kind: &DataType, config: DataFusionConfig) -> SqlOperator {
    let mut operator = SqlOperator::new("layout3", query, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref("events", BatchKind::Table, true, Some(schema(kind)))
                    .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap();
    operator.set_stream_resources(config, crate::UdfRegistrySnapshot::default(), vec![]);
    operator
}

fn schema(kind: &DataType) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Int64, true).with_metadata(
                std::collections::HashMap::from([("role".into(), "key".into())]),
            ),
            Field::new("unused", DataType::Utf8, false).with_metadata(
                std::collections::HashMap::from([("role".into(), "declared-unused".into())]),
            ),
            Field::new("value", kind.clone(), true).with_metadata(std::collections::HashMap::from(
                [("unit".into(), "native".into())],
            )),
        ],
        std::collections::HashMap::from([("origin".into(), "independent-full-port".into())]),
    ))
}

fn input(kind: &DataType, keys: &[Option<i64>], values: &[Option<i64>], sequence: u64) -> Batch {
    let value = ScalarValue::iter_to_array(values.iter().map(|value| match kind {
        DataType::Int64 => ScalarValue::Int64(*value),
        DataType::Decimal128(precision, scale) => {
            ScalarValue::Decimal128(value.map(i128::from), *precision, *scale)
        }
        _ => panic!("unsupported fixture datatype"),
    }))
    .unwrap();
    let record = RecordBatch::try_new(
        schema(kind),
        vec![
            Arc::new(Int64Array::from(keys.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "independent unused payload";
                keys.len()
            ])),
            value,
        ],
    )
    .unwrap();
    Batch::table(
        vec![record],
        BatchMetadata::new(
            "layout3-source",
            sequence,
            JsonMap::from([
                ("prefix".into(), json!(sequence)),
                ("audit".into(), json!({"trusted": true})),
            ]),
        )
        .unwrap(),
    )
    .unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(1, "layout3", JsonMap::new(), None, CancellationToken::new())
}

async fn process(
    operator: &mut SqlOperator,
    input: Batch,
    context: &StreamOperatorContext<'_>,
) -> Batch {
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", input, context, &mut output)
        .await
        .unwrap();
    let records = output.drain("output");
    assert_eq!(records.len(), 1);
    records[0].as_data().unwrap().clone()
}

fn rows(batch: &Batch) -> Vec<Vec<ScalarValue>> {
    let mut rows = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows
}

fn pool(runtime: &DataFusionRuntime) -> Arc<dyn MemoryPool> {
    runtime
        .incremental_planner_context(0)
        .runtime_env()
        .memory_pool
        .clone()
}

fn identity_credit(runtime: &DataFusionRuntime, schema: &SchemaRef) -> MemoryReservation {
    let reservation = runtime.incremental_reservation("layout3-fixture-identity");
    let bound = incremental::checked_bytes(
        131_072,
        [(ipc::schema_bytes(schema).unwrap(), 64)],
        "layout3",
    )
    .unwrap();
    incremental::ensure_reservation(&reservation, bound, "layout3").unwrap();
    reservation
}

fn schema_digest(schema: &SchemaRef) -> String {
    retention::encode_schema(schema)
        .unwrap()
        .sha256()
        .to_owned()
}

fn fields_json(fields: &[FieldRef]) -> Value {
    Value::Array(
        fields
            .iter()
            .map(|field| {
                json!({"name": field.name(),
        "schema_sha256": schema_digest(&Arc::new(Schema::new(vec![field.clone()]))) })
            })
            .collect(),
    )
}

fn input_json(input: &NativeAggregateInput) -> Value {
    match input {
        NativeAggregateInput::Column { index, field } => {
            json!({"kind": "column", "index": index, "field": fields_json(std::slice::from_ref(field))})
        }
        NativeAggregateInput::Literal(value) => {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "literal",
                value.data_type(),
                value.is_null(),
            )]));
            let record =
                RecordBatch::try_new(schema.clone(), vec![value.to_array().unwrap()]).unwrap();
            let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
            let bytes = super::super::encode_sql_state(&batch).unwrap();
            let segment = crate::operator::StateSegment::new(bytes);
            json!({"kind": "literal", "schema_sha256": schema_digest(&schema), "ipc_sha256": segment.sha256()})
        }
        NativeAggregateInput::Cast { input, field, safe } => {
            json!({"kind": "cast", "input": input_json(input),
            "field": fields_json(std::slice::from_ref(field)), "safe": safe, "format_policy": "datafusion_default"})
        }
        NativeAggregateInput::TryCast { input, dtype } => {
            json!({"kind":"try_cast", "input":input_json(input),
                "field":fields_json(&[Arc::new(Field::new("try_cast", dtype.clone(), true))]),
                "format_policy":"datafusion_default"})
        }
        NativeAggregateInput::Binary {
            left,
            op,
            right,
            fail_on_overflow,
        } => {
            json!({"kind":"binary", "left":input_json(left), "operator":op.to_string(),
                "right":input_json(right), "fail_on_overflow":fail_on_overflow})
        }
        NativeAggregateInput::Unary { input, op } => {
            json!({"kind":"unary", "input":input_json(input), "operator":op})
        }
        NativeAggregateInput::Case {
            operand,
            branches,
            fallback,
        } => {
            json!({"kind":"case","operand":operand.as_deref().map(input_json),
                "branches":branches.iter().map(|(when,then)| json!({
                    "when":input_json(when),"then":input_json(then),
                })).collect::<Vec<_>>(),"fallback":fallback.as_deref().map(input_json)})
        }
    }
}

fn native_json(descriptor: &NativeStateDescriptor) -> Value {
    json!({"policy": descriptor.policy, "keys": fields_json(&descriptor.key_fields),
        "aggregates": descriptor.aggregate_names.iter().enumerate().map(|(slot, function)| json!({
            "function": function, "inputs": descriptor.aggregate_inputs[slot].iter().map(input_json).collect::<Vec<_>>(),
            "filter": descriptor.aggregate_filters[slot].as_ref().map(input_json),
            "all_rows": descriptor.count_all_rows[slot], "state_fields": fields_json(&descriptor.state_fields[slot]),
            "result_field": fields_json(std::slice::from_ref(&descriptor.result_fields[slot])),
        })).collect::<Vec<_>>(), "projection": descriptor.projection.iter().map(input_json).collect::<Vec<_>>(),
        "post_filter":descriptor.post_filter.as_ref().map(input_json),
        "post_order":descriptor.post_order.as_ref().map(|order| json!({
            "keys":order.keys.iter().map(|(input, descending, nulls_first)| json!({
                "input":input_json(input),"descending":descending,"nulls_first":nulls_first,
            })).collect::<Vec<_>>(),"skip":order.skip,"fetch":order.fetch,
        })),
        "wire_schema_sha256": schema_digest(&descriptor.wire_schema), "output_schema_sha256": schema_digest(&descriptor.output_schema)})
}
