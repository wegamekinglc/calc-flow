#![cfg(feature = "benchmark-internals")]

use calc_flow::benchmark_internals::validate_matched_fixture;
use serde_json::json;

fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

fn write_json(root: &std::path::Path, name: &str, value: &serde_json::Value) -> serde_json::Value {
    let bytes = serde_json::to_vec(value).unwrap();
    std::fs::write(root.join(name), &bytes).unwrap();
    json!({"path": name, "sha256": digest(&bytes)})
}

fn write_ipc(
    root: &std::path::Path,
    name: &str,
    batches: &[datafusion::arrow::record_batch::RecordBatch],
) -> serde_json::Value {
    let mut bytes = Vec::new();
    let mut writer =
        datafusion::arrow::ipc::writer::FileWriter::try_new(&mut bytes, &batches[0].schema())
            .unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.finish().unwrap();
    drop(writer);
    std::fs::write(root.join(name), &bytes).unwrap();
    json!({"path": name, "sha256": digest(&bytes), "rows": batches.iter().map(datafusion::arrow::record_batch::RecordBatch::num_rows).sum::<usize>()})
}

fn tiny_fixture(root: &std::path::Path) -> std::path::PathBuf {
    tiny_fixture_with_inputs(root, false)
}

fn tiny_project(with_inputs: bool) -> serde_json::Value {
    use calc_flow::{AsofJoinSide, AsofStateLimits, StreamAsofJoinSpec};
    use std::time::Duration;
    let side = |prefix| {
        AsofJoinSide::new(
            vec!["key".into()],
            "ts".into(),
            vec!["sequence".into()],
            prefix,
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left".into()),
        side("right".into()),
        Duration::ZERO,
        AsofStateLimits::new(1_000, 10_000_000).unwrap(),
    )
    .unwrap();
    let fields = json!([
        {"name":"key", "data_type":"string", "nullable":false},
        {"name":"ts", "data_type":"timestamp[us, UTC]", "nullable":false},
        {"name":"sequence", "data_type":"uint64", "nullable":false},
        {"name":"value", "data_type":"float64", "nullable":false}
    ]);
    let joined_fields = ["left", "right"]
        .into_iter()
        .flat_map(|side| {
            fields.as_array().unwrap().iter().map(move |field| {
                json!({
                    "name":format!("{side}__{}",field["name"].as_str().unwrap()),
                    "data_type":field["data_type"],"nullable":side=="right"
                })
            })
        })
        .collect::<Vec<_>>();
    let mut project = json!({
        "format_version":3,"id":"native-test","name":"Native test",
        "runtime":{"mode":"stream","options":{}},
        "graph":{"name":"matched", "nodes":[
            {"id":"asof", "operator":{"kind":"stream_asof_join","spec":spec}, "input_ports":[
                {"name":"left","kind":"table","required":true,"schema":fields},
                {"name":"right","kind":"table","required":true,"schema":fields}
            ],"output_ports":[]},
            {"id":"select", "operator":{"kind":"expression","select":["left__sequence AS sequence", "right__value AS value"]},"input_ports":[{"name":"input","kind":"table","required":true,"schema":joined_fields}],"output_ports":[]}
        ], "edges":[{"source_node":"asof","source_port":"output","target_node":"select","target_port":"input"}]}
    });
    if with_inputs {
        for (name, side) in [("quotes", "left"), ("reference", "right")] {
            project["graph"]["nodes"].as_array_mut().unwrap().push(json!({
                "id":name,"operator":{"kind":"expression","select":["key", "ts", "sequence", "value"]},
                "input_ports":[{"name":"input","kind":"table","required":true,"schema":fields}],
                "output_ports":[{"name":"output","kind":"table","required":true,"schema":fields}]
            }));
            project["graph"]["edges"].as_array_mut().unwrap().push(json!({"source_node":name,"source_port":"output","target_node":"asof","target_port":side}));
        }
    }
    project
}

fn tiny_tables() -> (
    Vec<datafusion::arrow::record_batch::RecordBatch>,
    datafusion::arrow::record_batch::RecordBatch,
) {
    use datafusion::arrow::{
        array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };
    use std::sync::Arc;
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let batches = [(0_u64, 2_usize), (2, 2), (4, 1)]
        .into_iter()
        .map(|(start, count)| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(
                        (start..start + count as u64)
                            .map(|row| format!("K{row}"))
                            .collect::<Vec<_>>(),
                    )),
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![
                            10 + i64::try_from(start).unwrap();
                            count
                        ])
                        .with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from(
                        (start..start + count as u64).collect::<Vec<_>>(),
                    )),
                    Arc::new(Float64Array::from(
                        (start..start + count as u64)
                            .map(|row| f64::from(u32::try_from(row).unwrap()) + 0.5)
                            .collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let expected_schema = Arc::new(Schema::new(vec![
        Field::new("sequence", DataType::UInt64, false),
        Field::new("value", DataType::Float64, true),
    ]));
    let expected = RecordBatch::try_new(
        expected_schema,
        vec![
            Arc::new(UInt64Array::from(vec![0, 1, 2, 3, 4])),
            Arc::new(Float64Array::from(vec![0.5, 1.5, 2.5, 3.5, 4.5])),
        ],
    )
    .unwrap();
    (batches, expected)
}

fn input_binding(
    plan: &calc_flow::StreamExecutionPlan,
    with_inputs: bool,
    node: &str,
    port: &str,
) -> String {
    plan.external_inputs()
        .iter()
        .find(|(_, endpoint)| {
            if with_inputs {
                endpoint.node_id == node
            } else {
                endpoint.port == port
            }
        })
        .unwrap()
        .0
        .clone()
}

fn tiny_fixture_with_inputs(root: &std::path::Path, with_inputs: bool) -> std::path::PathBuf {
    use calc_flow::{
        ProviderRegistry, StreamRequirements, UdfRegistry, compile_stream_project_graph,
        import_project_json,
    };
    let project = tiny_project(with_inputs);
    let project_spec = import_project_json(&serde_json::to_vec(&project).unwrap()).unwrap();
    let plan = compile_stream_project_graph(
        &project_spec,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &StreamRequirements::default(),
    )
    .unwrap();
    let left = input_binding(&plan, with_inputs, "quotes", "left");
    let right = input_binding(&plan, with_inputs, "reference", "right");
    let output = plan.external_outputs().keys().next().unwrap().clone();
    let (batches, expected) = tiny_tables();
    let inputs = json!({left.clone():write_ipc(root,"left.arrow",&batches),right.clone():write_ipc(root,"right.arrow",&batches)});
    let mut processing = Vec::new();
    let mut barriers = Vec::new();
    for chunk in 0..3 {
        let begin = processing.len();
        for input in [&right, &left] {
            processing.push(json!({"kind":"data","input":input,"batches":[chunk],"chunk_id":chunk,"event_id":format!("processing/{}",processing.len())}));
        }
        for input in [&right, &left] {
            processing.push(json!({"kind":"watermark","input":input,"micros":11+chunk*2,"chunk_id":chunk,"event_id":format!("processing/{}",processing.len())}));
        }
        barriers.push(json!({"chunk_id":chunk,"after_events":(begin..processing.len()).map(|index|format!("processing/{index}")).collect::<Vec<_>>(),"output_rows":usize::min((chunk+1)*2,5)}));
    }
    let events = json!({"preload":[],"processing":processing,"terminal":[{"kind":"end","input":right,"event_id":"terminal/0"},{"kind":"end","input":left,"event_id":"terminal/1"}],"barriers":barriers});
    let mut partial = serde_json::Map::new();
    for phase in ["preload", "processing", "terminal"] {
        let mut grouped = serde_json::Map::new();
        for event in events[phase].as_array().unwrap() {
            grouped
                .entry(event["input"].as_str().unwrap().to_owned())
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .unwrap()
                .push(event.clone());
        }
        partial.insert(phase.into(), grouped.into());
    }
    partial.insert("barriers".into(), events["barriers"].clone());
    let input_identity = inputs
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, record)| {
            (
                name.clone(),
                json!({"rows":record["rows"],"sha256":record["sha256"]}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let fingerprint =
        |value: &serde_json::Value| digest(calc_flow::canonical_json(value).unwrap().as_bytes());
    let manifest = json!({
        "schema":"calc-flow.matched-native-fixture.v1","scenario":"asof_join","rows_per_side":5,"batch_rows":2,"checkpoint_enabled":!cfg!(feature = "benchmark-checkpoint-off"),
        "project":write_json(root,"project.json",&project),"inputs":inputs,"inputs_sha256":fingerprint(&input_identity.into()),
        "events":write_json(root,"events.json",&events),"shared_partial_order_sha256":fingerprint(&partial.into()),"canonical_native_events_sha256":fingerprint(&events),
        "expected_output":write_ipc(root,"expected.arrow",&[expected]),"output_binding":output,"compiled_plan_fingerprint":plan.fingerprint()
    });
    write_json(root, "fixture.json", &manifest);
    root.join("fixture.json")
}

#[tokio::test]
async fn test_matched_fixture_executes_compiled_projection_and_all_progress_callbacks() {
    let root = tempfile::tempdir().unwrap();
    let manifest = tiny_fixture(root.path());
    let result = calc_flow::benchmark_internals::run_matched_fixture(&manifest, true)
        .await
        .unwrap();
    assert_eq!(result["output_rows"], 5);
    assert_eq!(
        result["checkpoint_enabled"],
        !cfg!(feature = "benchmark-checkpoint-off")
    );
    assert_eq!(result["validated_all_rows"], true);
    assert_eq!(result["owned_cleanup_complete"], true);
    assert_eq!(result["nonterminal_epochs"], 0);
    assert_eq!(result["processing_callback_counts"]["data"], 9);
    assert!(result["processing_callback_counts"].get("end").is_none());
    assert_eq!(result["terminal_callback_counts"]["end"], 2);
    assert_eq!(result["processing_node_callback_counts"]["asof"]["data"], 6);
    assert_eq!(result["callback_counts"]["data"], 9);
    assert_eq!(result["callback_counts"]["ingress_progress"], 12);
    assert_eq!(
        result["compiled_projection"][0]["output_fields"],
        json!(["left__sequence", "right__value"])
    );
    assert_eq!(
        result["post_processing_status"]["asof"]["pending_left_rows"],
        0
    );
    for clock in ["processing_seconds", "terminal_seconds", "cleanup_seconds"] {
        assert!(result[clock].as_f64().unwrap() > 0.0);
    }
}

#[tokio::test]
async fn test_matched_fixture_preserves_input_stages_and_message_budget() {
    let root = tempfile::tempdir().unwrap();
    let manifest = tiny_fixture_with_inputs(root.path(), true);
    let result = calc_flow::benchmark_internals::run_matched_fixture(&manifest, true)
        .await
        .unwrap();
    assert_eq!(result["rows_per_side"], 5);
    assert_eq!(result["batch_rows"], 2);
    assert_eq!(
        result["output_budget"],
        json!({"max_rows":2,"max_bytes":64<<20})
    );
    assert_eq!(result["assembled_output_rows"], 5);
    assert_eq!(result["assembled_output_batches"], 3);
    let projection = result["compiled_projection"].as_array().unwrap();
    assert_eq!(projection.len(), 4);
    for node in ["quotes", "reference", "select"] {
        assert_eq!(result["node_callback_counts"][node]["data"], 3);
    }
    assert_eq!(result["node_callback_counts"]["asof"]["data"], 6);
}

#[tokio::test]
async fn test_fixture_cli_rejects_extra_arguments_before_loading_files() {
    let root = tempfile::tempdir().unwrap();
    let fixture = tiny_fixture(root.path());
    let arguments = vec![
        "--matched-fixture".into(),
        fixture.display().to_string(),
        "--check".into(),
    ];
    let result = calc_flow::benchmark_internals::run_fixture_cli(&arguments)
        .await
        .unwrap();
    assert_eq!(result["output_rows"], 5);
    for arguments in [
        vec!["--matched-fixture"],
        vec!["--matched-fixture", "missing", "--rows", "1000000"],
    ] {
        let arguments = arguments.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let error = calc_flow::benchmark_internals::run_fixture_cli(&arguments)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("arguments"), "{error}");
    }
}

async fn dispatch_envelope(
    manifest: &serde_json::Value,
    fixture: &std::path::Path,
) -> serde_json::Value {
    let control = calc_flow::benchmark_internals::run_matched_fixture(fixture, true)
        .await
        .unwrap();
    let mut events = Vec::new();
    for ordinal in 0..3 {
        for ingress in ["right", "left"] {
            let rows = if ordinal == 2 { 1 } else { 2 };
            events.push(json!({"kind":"data","ingress":ingress,"ordinal":ordinal,"rows":rows,"first_sequence":ordinal*2,"last_sequence":ordinal*2+rows-1,"record_batch_rows":[rows]}));
            events.push(json!({"kind":"watermark","ingress":ingress,"micros":11+ordinal*2}));
        }
    }
    let counts = |batches, rows, output_batches, output_rows| json!({"input_batches":batches,"input_rows":rows,"fanned_out_batches":output_batches,"fanned_out_rows":output_rows,"errors":0,"ended":false});
    json!({
        "schema":"calc-flow.issue406-dispatch-replay.v1", "source_observation_id":"tiny-e2e-source", "source_raw_sha256":"c".repeat(64),
        "match_identity":{
            "project_sha256":manifest["project"]["sha256"], "inputs_sha256":manifest["inputs_sha256"],
            "shared_partial_order_sha256":manifest["shared_partial_order_sha256"],"output_contract_sha256":manifest["expected_output"]["sha256"],
            "compiled_plan_fingerprint":manifest["compiled_plan_fingerprint"],"rows_per_side":5,"batch_rows":2,
            "checkpoint_enabled":manifest["checkpoint_enabled"],"mode":if cfg!(feature="benchmark-checkpoint-off") {"off"} else {"low"},
            "cpu_affinity":calc_flow::benchmark_internals::process_observations().unwrap()["observed_cpu_affinity"],
            "processing_boundary":"final-output-projection-watermark-completed","output_frontier":"minimum-input-watermark-minus-one-microsecond"
        },
        "dispatch_trace":{
            "schema":"calc-flow.asof-dispatch-trace.v1","job_id":19,"compiled_plan_fingerprint":manifest["compiled_plan_fingerprint"],"node_id":"asof",
            "capacity":8192,"record_count":12,"overflowed":false,"allow_barriers":false,"terminal_excluded":true,"events":events,
            "callback_counts":{"data":6,"ingress_progress":6,"watermark":3},
            "asof_state":control["post_processing_status"]["asof"],
            "output_record_batch_rows":[[2],[2],[1]],
            "completed_frontiers":{"quotes":15,"reference":15,"asof":15,"select":14},
            "output_node_id":"select","required_output_frontier_micros":14,
            "operator_data_counts":{"quotes":counts(3,5,3,5),"reference":counts(3,5,3,5),"asof":counts(6,10,3,5),"select":counts(3,5,3,5)}
        }
    })
}

#[tokio::test]
async fn test_native_replays_exact_recorded_asof_dispatch_and_proves_source_identity() {
    let root = tempfile::tempdir().unwrap();
    let fixture = tiny_fixture_with_inputs(root.path(), true);
    let manifest = validate_matched_fixture(&fixture).unwrap();
    let envelope = dispatch_envelope(&manifest, &fixture).await;
    write_json(root.path(), "trace.json", &envelope);
    let result = calc_flow::benchmark_internals::run_fixture_cli(&[
        "--matched-fixture".into(),
        fixture.display().to_string(),
        "--dispatch-trace".into(),
        root.path().join("trace.json").display().to_string(),
        "--check".into(),
    ])
    .await
    .unwrap();
    assert_eq!(result["dispatch_trace_verified"], true);
    assert_eq!(result["dispatch_replay_state_verified"], true);
    assert_eq!(
        result["dispatch_trace"]["output_record_batch_rows"],
        envelope["dispatch_trace"]["output_record_batch_rows"]
    );
    assert_eq!(
        result["dispatch_trace"]["asof_state"],
        envelope["dispatch_trace"]["asof_state"]
    );
    assert_eq!(result["replay_source_observation_id"], "tiny-e2e-source");
    assert_eq!(result["replay_source_raw_sha256"], "c".repeat(64));
    assert_eq!(
        result["replay_trace_sha256"],
        digest(&std::fs::read(root.path().join("trace.json")).unwrap())
    );
    assert_eq!(
        result["dispatch_trace"]["events"],
        envelope["dispatch_trace"]["events"]
    );
    assert_eq!(
        result["dispatch_trace"]["callback_counts"],
        envelope["dispatch_trace"]["callback_counts"]
    );
    assert_eq!(
        result["dispatch_trace"]["operator_data_counts"],
        envelope["dispatch_trace"]["operator_data_counts"]
    );
    assert_eq!(result["output_rows"], 5);
    assert_eq!(result["owned_cleanup_complete"], true);
}

#[tokio::test]
async fn test_dispatch_replay_rejects_mismatched_identity_geometry_and_callback_work() {
    let root = tempfile::tempdir().unwrap();
    let fixture = tiny_fixture_with_inputs(root.path(), true);
    let manifest = validate_matched_fixture(&fixture).unwrap();
    let original = dispatch_envelope(&manifest, &fixture).await;
    for (pointer, replacement) in [
        ("/dispatch_trace/asof_state/retained_right_rows", json!(999)),
        ("/dispatch_trace/output_record_batch_rows/0", json!([1, 1])),
        ("/match_identity/project_sha256", json!("d".repeat(64))),
        (
            "/dispatch_trace/compiled_plan_fingerprint",
            json!("different"),
        ),
        ("/dispatch_trace/events/0/first_sequence", json!(1)),
        ("/dispatch_trace/events/0/record_batch_rows", json!([1, 1])),
        ("/dispatch_trace/events/1/micros", json!(12)),
        ("/dispatch_trace/events/1/kind", json!("barrier")),
        ("/dispatch_trace/events/4/ordinal", json!(0)),
        ("/dispatch_trace/callback_counts/ingress_progress", json!(7)),
        (
            "/dispatch_trace/operator_data_counts/asof/input_batches",
            json!(5),
        ),
        ("/dispatch_trace/completed_frontiers/select", json!(13)),
    ] {
        let mut envelope = original.clone();
        *envelope.pointer_mut(pointer).unwrap() = replacement;
        write_json(root.path(), "trace.json", &envelope);
        let result = calc_flow::benchmark_internals::run_fixture_cli(&[
            "--matched-fixture".into(),
            fixture.display().to_string(),
            "--dispatch-trace".into(),
            root.path().join("trace.json").display().to_string(),
            "--check".into(),
        ])
        .await;
        assert!(result.is_err(), "accepted invalid replay field {pointer}");
    }
}

#[tokio::test]
async fn test_ordering_proof_covers_all_two_chunk_fifo_interleavings() {
    let root = tempfile::tempdir().unwrap();
    let path = tiny_fixture_with_inputs(root.path(), true);
    let result = calc_flow::benchmark_internals::run_fixture_cli(&[
        "--prove-ordering".into(),
        path.display().to_string(),
    ])
    .await
    .unwrap();
    assert_eq!(result["schema"], "calc-flow.native-ordering-proof.v1");
    assert_eq!(result["schedule_count"], 36);
    assert_eq!(result["rendezvous_equivalent"], true);
    assert_eq!(result["all_outputs_validated"], true);
    assert_eq!(result["all_owned_cleanup_complete"], true);
    assert_eq!(result["final_partial_rows"], 1);
    let schedules = result["schedules"].as_array().unwrap();
    assert_eq!(schedules.len(), 36);
    assert!(schedules.iter().all(|sample| {
        sample["callback_trace"]
            .as_array()
            .is_some_and(|trace| !trace.is_empty())
    }));
    assert!(
        !result
            .as_object()
            .unwrap()
            .keys()
            .any(|key| key.ends_with("seconds"))
    );
}

#[cfg(target_os = "linux")]
#[test]
fn test_process_observation_reads_actual_affinity_and_thread_environment() {
    let result = calc_flow::benchmark_internals::process_observations().unwrap();
    assert!(
        !result["observed_cpu_affinity"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    for key in [
        "TOKIO_WORKER_THREADS",
        "OMP_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "RAYON_NUM_THREADS",
    ] {
        assert_eq!(
            result["observed_thread_environment"][key],
            serde_json::to_value(std::env::var(key).ok()).unwrap()
        );
    }
}

#[cfg(not(feature = "benchmark-checkpoint-off"))]
#[test]
fn test_unavailable_checkpoint_off_rejects_before_loading_inputs() {
    let root = tempfile::tempdir().unwrap();
    let manifest = root.path().join("fixture.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(&json!({
            "schema": "calc-flow.matched-native-fixture.v1",
            "checkpoint_enabled": false,
        }))
        .unwrap(),
    )
    .unwrap();
    let error = validate_matched_fixture(&manifest).unwrap_err();
    assert!(
        error.to_string().contains("checkpoint-off capability"),
        "{error}"
    );
}

#[test]
fn test_modified_project_is_rejected_before_compilation() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("project.json"), b"tampered").unwrap();
    let manifest = root.path().join("fixture.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(&json!({
            "schema": "calc-flow.matched-native-fixture.v1",
            "checkpoint_enabled": true,
            "project": {"path": "project.json", "sha256": "a".repeat(64)},
        }))
        .unwrap(),
    )
    .unwrap();
    let error = validate_matched_fixture(&manifest).unwrap_err();
    assert!(error.to_string().contains("project SHA256"), "{error}");
}

fn reseal_events(
    root: &std::path::Path,
    manifest: &mut serde_json::Value,
    events: &serde_json::Value,
) {
    let fingerprint =
        |value: &serde_json::Value| digest(calc_flow::canonical_json(value).unwrap().as_bytes());
    let mut partial = serde_json::Map::new();
    for phase in ["preload", "processing", "terminal"] {
        let mut grouped = serde_json::Map::new();
        for event in events[phase].as_array().unwrap() {
            grouped
                .entry(event["input"].as_str().unwrap().to_owned())
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .unwrap()
                .push(event.clone());
        }
        partial.insert(phase.into(), grouped.into());
    }
    partial.insert("barriers".into(), events["barriers"].clone());
    manifest["events"] = write_json(root, "events.json", events);
    manifest["canonical_native_events_sha256"] = fingerprint(events).into();
    manifest["shared_partial_order_sha256"] = fingerprint(&partial.into()).into();
    write_json(root, "fixture.json", manifest);
}

fn tiny_join_fixture(root: &std::path::Path) -> std::path::PathBuf {
    use datafusion::arrow::{
        array::StringArray, ipc::reader::FileReader, record_batch::RecordBatch,
    };
    use std::{fs::File, sync::Arc};
    let path = tiny_fixture(root);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut project: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("project.json")).unwrap()).unwrap();
    project["graph"]["nodes"][0]["operator"] = json!({"kind":"stream_join","spec":{"join_type":"inner","left_keys":["key"],"right_keys":["key"],"left_event_time":"ts","right_event_time":"ts","left_prefix":"left","right_prefix":"right","bounds":{"before_micros":10,"after_micros":0},"limits":{"max_state_rows_per_side":10,"max_state_bytes_per_side":1_000_000,"max_matches_per_input_batch":10}}});
    project["graph"]["nodes"][1]["operator"]["select"] =
        json!(["left__sequence AS sequence", "left__value AS value"]);
    for field in project["graph"]["nodes"][1]["input_ports"][0]["schema"]
        .as_array_mut()
        .unwrap()
    {
        field["nullable"] = false.into();
    }
    let plan = calc_flow::compile_stream_project_graph(
        &calc_flow::import_project_json(&serde_json::to_vec(&project).unwrap()).unwrap(),
        &calc_flow::ProviderRegistry::default(),
        &calc_flow::UdfRegistry::new().snapshot(),
        &calc_flow::StreamRequirements::default(),
    )
    .unwrap();
    let left = plan
        .external_inputs()
        .iter()
        .find(|(_, endpoint)| endpoint.port == "left")
        .unwrap()
        .0
        .clone();
    let right = plan
        .external_inputs()
        .iter()
        .find(|(_, endpoint)| endpoint.port == "right")
        .unwrap()
        .0
        .clone();
    let batches = FileReader::try_new(File::open(root.join("left.arrow")).unwrap(), None)
        .unwrap()
        .map(|batch| {
            let batch = batch.unwrap();
            let mut columns = batch.columns().to_vec();
            columns[0] = Arc::new(StringArray::from(vec!["K0"; batch.num_rows()]));
            RecordBatch::try_new(batch.schema(), columns).unwrap()
        })
        .collect::<Vec<_>>();
    let inputs = json!({left.clone():write_ipc(root,"left.arrow",&batches),right.clone():write_ipc(root,"right.arrow",&[batches[0].slice(0,1)])});
    manifest["inputs_sha256"] = digest(
        calc_flow::canonical_json(
            &inputs
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, record)| {
                    (
                        name.clone(),
                        json!({"rows":record["rows"],"sha256":record["sha256"]}),
                    )
                })
                .collect::<serde_json::Map<_, _>>()
                .into(),
        )
        .unwrap()
        .as_bytes(),
    )
    .into();
    manifest["inputs"] = inputs;
    manifest["scenario"] = "join_sealed".into();
    manifest["rows"] = 5.into();
    manifest["dimension_rows"] = 1.into();
    manifest.as_object_mut().unwrap().remove("rows_per_side");
    manifest["project"] = write_json(root, "project.json", &project);
    manifest["compiled_plan_fingerprint"] = plan.fingerprint().into();
    let mut processing = Vec::new();
    let mut barriers = Vec::new();
    for chunk in 0..3 {
        let begin = processing.len();
        processing.push(json!({"kind":"data","input":left,"batches":[chunk],"chunk_id":chunk,"event_id":format!("processing/{begin}")}));
        processing.push(json!({"kind":"watermark","input":left,"micros":11+chunk*2,"chunk_id":chunk,"event_id":format!("processing/{}",begin+1)}));
        barriers.push(json!({"chunk_id":chunk,"after_events":[format!("processing/{begin}"),format!("processing/{}",begin+1)],"output_rows":usize::min((chunk+1)*2,5)}));
    }
    let events = json!({"preload":[{"kind":"data","input":right,"batches":[0],"event_id":"preload/0"},{"kind":"watermark","input":right,"micros":100,"event_id":"preload/1"}],"processing":processing,"terminal":[{"kind":"end","input":right,"event_id":"terminal/0"},{"kind":"end","input":left,"event_id":"terminal/1"}],"barriers":barriers});
    reseal_events(root, &mut manifest, &events);
    path
}

#[tokio::test]
async fn test_join_fixture_times_actual_handlers_and_requires_no_left_retention() {
    let root = tempfile::tempdir().unwrap();
    let path = tiny_join_fixture(root.path());
    let result = calc_flow::benchmark_internals::run_matched_fixture(&path, true)
        .await
        .unwrap();
    let samples = result["handler_samples"].as_array().unwrap();
    assert_eq!(
        samples
            .iter()
            .map(|sample| sample["rows"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![2, 2, 1]
    );
    assert!(samples.iter().all(
        |sample| sample["seconds"].as_f64().unwrap() > 0.0 && sample["left_retained_rows"] == 0
    ));
    assert_eq!(result["validated_all_rows"], true);
    assert_eq!(result["output_rows"], 5);
    assert_eq!(result["owned_cleanup_complete"], true);
    assert_eq!(result["compiled_projection"][0]["kind"], "stream_join");
}

#[tokio::test]
async fn test_fixture_rejects_unmatched_schedule_shapes_before_replay() {
    for variant in ["idle", "fifo", "chunks", "geometry"] {
        let root = tempfile::tempdir().unwrap();
        let path = tiny_fixture(root.path());
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let mut events: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.path().join("events.json")).unwrap())
                .unwrap();
        match variant {
            "idle" => events["processing"][2]["kind"] = "idle".into(),
            "fifo" => {
                events["processing"].as_array_mut().unwrap().swap(0, 2);
                events["barriers"][0]["after_events"]
                    .as_array_mut()
                    .unwrap()
                    .swap(0, 2);
            }
            "chunks" => {
                events["processing"].as_array_mut().unwrap().swap(3, 4);
            }
            "geometry" => manifest["batch_rows"] = 3.into(),
            _ => unreachable!(),
        }
        reseal_events(root.path(), &mut manifest, &events);
        let error = calc_flow::benchmark_internals::run_matched_fixture(&path, true)
            .await
            .unwrap_err();
        let expected = match variant {
            "idle" => "control event",
            "fifo" => "Data then Watermark",
            "chunks" => "contiguous",
            "geometry" => "message geometry",
            _ => unreachable!(),
        };
        assert!(error.to_string().contains(expected), "{variant}: {error}");
    }
}

#[tokio::test]
async fn test_oracle_rejects_reordered_output_columns() {
    use datafusion::arrow::ipc::reader::FileReader;
    let root = tempfile::tempdir().unwrap();
    let path = tiny_fixture(root.path());
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let expected = FileReader::try_new(
        std::fs::File::open(root.path().join("expected.arrow")).unwrap(),
        None,
    )
    .unwrap()
    .next()
    .unwrap()
    .unwrap()
    .project(&[1, 0])
    .unwrap();
    manifest["expected_output"] = write_ipc(root.path(), "expected.arrow", &[expected]);
    write_json(root.path(), "fixture.json", &manifest);
    let error = calc_flow::benchmark_internals::run_matched_fixture(&path, true)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("projection order"), "{error}");
}
