use super::{
    BTreeMap, CompiledStreamOperator, DataCounts, EventKind, Fixture, FixtureEvent, Replay, Result,
    RuntimeConsumer, StreamMessage, Value, VecDeque, invalid, json,
};
use crate::benchmark_internals::{DispatchTraceOwner, process_observations};

pub(super) fn message(kind: &EventKind) -> StreamMessage {
    match kind {
        EventKind::Data(batch) => StreamMessage::data(batch.clone()),
        EventKind::Watermark(at) => StreamMessage::watermark(*at),
        EventKind::Idle => StreamMessage::idle(),
        EventKind::End => StreamMessage::end_of_input(),
    }
}

pub(super) fn prepare(replay: &Replay, fixture: &mut Fixture) -> Result<()> {
    let Some(source) = &fixture.dispatch_replay else {
        return Ok(());
    };
    validate_identity(&source.envelope, &fixture.manifest)?;
    let trace = source.envelope["dispatch_trace"].clone();
    let node = validate_trace(replay, fixture, &trace)?;
    let routes = source_ingresses(replay, &node)?;
    let mut queues = BTreeMap::<String, VecDeque<FixtureEvent>>::new();
    for event in &fixture.processing {
        queues
            .entry(routes[&event.input].clone())
            .or_default()
            .push_back(event.clone());
    }
    let events = trace["events"].as_array().expect("validated events");
    let recorder = DispatchTraceOwner::default();
    recorder.start(1, &replay.parts.fingerprint, &node, false)?;
    let mut processing = Vec::with_capacity(events.len());
    for actual in events {
        let ingress = actual["ingress"]
            .as_str()
            .ok_or_else(|| invalid("dispatch ingress is missing"))?;
        let event = queues
            .get_mut(ingress)
            .and_then(VecDeque::pop_front)
            .ok_or_else(|| invalid("dispatch event duplicates or exceeds fixture input"))?;
        recorder.record(&node, ingress, &message(&event.kind))?;
        processing.push(event);
    }
    if queues.values().any(|queue| !queue.is_empty())
        || recorder.take()?["events"] != trace["events"]
    {
        return Err(invalid(
            "dispatch events differ from sealed IPC geometry or per-input FIFO",
        ));
    }
    let barriers = rendezvous(&processing, fixture)?;
    fixture.processing = processing;
    fixture.barriers = barriers;
    Ok(())
}

fn validate_identity(envelope: &Value, manifest: &Value) -> Result<()> {
    if envelope["schema"] != "calc-flow.issue406-dispatch-replay.v1"
        || envelope["source_observation_id"]
            .as_str()
            .is_none_or(str::is_empty)
        || envelope["source_raw_sha256"].as_str().is_none_or(|sha| {
            sha.len() != 64
                || !sha
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    {
        return Err(invalid("invalid dispatch source observation identity"));
    }
    let enabled = manifest["checkpoint_enabled"]
        .as_bool()
        .expect("fixture mode");
    let identity = json!({
        "project_sha256":manifest["project"]["sha256"], "inputs_sha256":manifest["inputs_sha256"],
        "shared_partial_order_sha256":manifest["shared_partial_order_sha256"], "output_contract_sha256":manifest["expected_output"]["sha256"],
        "compiled_plan_fingerprint":manifest["compiled_plan_fingerprint"], "rows_per_side":manifest["rows_per_side"], "batch_rows":manifest["batch_rows"],
        "checkpoint_enabled":enabled, "mode":if enabled {"low"} else {"off"},
        "cpu_affinity":process_observations()?["observed_cpu_affinity"],
        "processing_boundary":"final-output-projection-watermark-completed", "output_frontier":"minimum-input-watermark-minus-one-microsecond"
    });
    if envelope["match_identity"] != identity {
        return Err(invalid(
            "dispatch match identity differs from native fixture or process",
        ));
    }
    Ok(())
}

fn validate_trace(replay: &Replay, fixture: &Fixture, trace: &Value) -> Result<String> {
    let node = replay
        .nodes
        .iter()
        .find(|(_, state)| {
            matches!(
                state.node.operator,
                CompiledStreamOperator::StreamAsofJoin(_)
            )
        })
        .map(|(node, _)| node.clone())
        .ok_or_else(|| invalid("dispatch replay requires ASOF"))?;
    let events = trace["events"]
        .as_array()
        .ok_or_else(|| invalid("dispatch events must be an array"))?;
    if trace["schema"] != "calc-flow.asof-dispatch-trace.v1"
        || trace["node_id"] != node
        || trace["compiled_plan_fingerprint"] != replay.parts.fingerprint
        || trace["capacity"] != 8192
        || trace["record_count"].as_u64() != u64::try_from(events.len()).ok()
        || events.is_empty()
        || events.len() > 8192
        || trace["overflowed"] != false
        || trace["allow_barriers"] != false
        || trace["terminal_excluded"] != true
        || trace["job_id"].as_u64().is_none_or(|id| id == 0)
        || !fixture.preload.is_empty()
    {
        return Err(invalid("invalid checkpoint-free ASOF dispatch capture"));
    }
    let output_node = &replay.parts.sink_routes[&replay.output_binding]
        .source
        .node_id;
    let mut watermarks = BTreeMap::new();
    for event in &fixture.processing {
        if let EventKind::Watermark(at) = event.kind {
            watermarks.insert(&event.input, at.as_micros());
        }
    }
    let frontier = watermarks
        .values()
        .min()
        .and_then(|at| at.checked_sub(1))
        .ok_or_else(|| invalid("dispatch fixture has no final frontier"))?;
    if trace["output_node_id"] != *output_node
        || trace["required_output_frontier_micros"].as_i64() != Some(frontier)
        || trace["completed_frontiers"][output_node]
            .as_i64()
            .is_none_or(|at| at < frontier)
    {
        return Err(invalid(
            "final output projection watermark has not completed",
        ));
    }
    Ok(node)
}

fn source_ingresses(replay: &Replay, asof: &str) -> Result<BTreeMap<String, String>> {
    replay
        .parts
        .source_routes
        .iter()
        .map(|(binding, route)| {
            let ingress = if route.target.node_id == asof {
                route.target.port.clone()
            } else {
                let node = &replay.nodes[&route.target.node_id].node;
                let edge = &replay.parts.edges[&node.output_edges["output"][0]];
                let RuntimeConsumer::Node { node_id, ingress } = &edge.consumer else {
                    return Err(invalid("source expression does not feed ASOF"));
                };
                if node_id != asof {
                    return Err(invalid("source expression targets another node"));
                }
                ingress.clone()
            };
            Ok((binding.clone(), ingress))
        })
        .collect()
}

fn rendezvous(processing: &[FixtureEvent], fixture: &Fixture) -> Result<BTreeMap<String, usize>> {
    let batch_rows = fixture.manifest["batch_rows"]
        .as_u64()
        .expect("fixture batch size");
    let rows = fixture.manifest["rows_per_side"]
        .as_u64()
        .expect("ASOF rows");
    let mut barriers = BTreeMap::new();
    for (chunk, events) in processing.chunks(4).enumerate() {
        let chunk = u64::try_from(chunk).map_err(|_| invalid("dispatch chunk overflow"))?;
        if events.len() != 4 || events.iter().any(|event| event.chunk != Some(chunk)) {
            return Err(invalid(
                "actual dispatch crosses the per-chunk output rendezvous",
            ));
        }
        let count = usize::try_from(rows.min((chunk + 1) * batch_rows))
            .map_err(|_| invalid("dispatch rows exceed platform"))?;
        barriers.insert(events[3].id.clone(), count);
    }
    Ok(barriers)
}

pub(super) fn start(replay: &mut Replay, fixture: &Fixture) -> Result<()> {
    let Some(source) = &fixture.dispatch_replay else {
        return Ok(());
    };
    let node = source.envelope["dispatch_trace"]["node_id"]
        .as_str()
        .expect("validated ASOF node");
    replay.job.dispatch_trace().start(
        replay.job.job_id(),
        &replay.parts.fingerprint,
        node,
        false,
    )?;
    replay.job.dispatch_trace().register_nodes(&replay.order)?;
    replay.audit.node_data = replay
        .order
        .iter()
        .map(|node| (node.clone(), DataCounts::default()))
        .collect();
    replay.audit.output_groups = Some(vec![0; 8192].into_boxed_slice());
    Ok(())
}

pub(super) fn finish(replay: &Replay, fixture: &Fixture) -> Result<Option<Value>> {
    let Some(source) = &fixture.dispatch_replay else {
        return Ok(None);
    };
    let expected = &source.envelope["dispatch_trace"];
    let mut trace = replay.job.dispatch_trace().take()?;
    trace["operator_data_counts"] = serde_json::to_value(&replay.audit.node_data)
        .map_err(|error| invalid(format!("dispatch data counts: {error}")))?;
    trace["output_node_id"] = expected["output_node_id"].clone();
    trace["required_output_frontier_micros"] = expected["required_output_frontier_micros"].clone();
    let node = expected["node_id"].as_str().expect("validated ASOF node");
    trace["asof_state"] = replay.status()?[node].clone();
    trace["output_record_batch_rows"] = output_geometry(replay)?;
    for field in [
        "events",
        "callback_counts",
        "completed_frontiers",
        "operator_data_counts",
        "asof_state",
        "output_record_batch_rows",
    ] {
        if trace[field] != expected[field] {
            return Err(invalid(format!(
                "native replay differs from completed runtime {field}"
            )));
        }
    }
    Ok(Some(trace))
}

fn output_geometry(replay: &Replay) -> Result<Value> {
    let groups = replay
        .audit
        .output_groups
        .as_ref()
        .expect("dispatch geometry buffer");
    let mut offset = 0;
    let geometry = groups[..replay.audit.output_group_count]
        .iter()
        .map(|count| {
            let end = offset + count;
            let records = replay
                .output
                .get(offset..end)
                .ok_or_else(|| invalid("dispatch output geometry exceeds recorded batches"))?;
            offset = end;
            Ok(records
                .iter()
                .map(datafusion::arrow::record_batch::RecordBatch::num_rows)
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    if offset != replay.output.len() {
        return Err(invalid("dispatch output geometry is incomplete"));
    }
    Ok(json!(geometry))
}
