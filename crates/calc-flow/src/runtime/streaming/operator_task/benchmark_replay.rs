use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::Instant,
};

#[path = "benchmark_replay/dispatch.rs"]
mod dispatch;

use datafusion::arrow::{
    array::{Array, Float64Array, UInt64Array},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};

use crate::benchmark_internals::{
    EventKind, Fixture, FixtureEvent, fingerprint, invalid, settle_job,
};
use crate::pipeline::{
    CompiledStreamOperator, RuntimeConsumer, RuntimeProducer, RuntimeStreamNode,
    StreamRuntimePlanParts,
};
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, EventTime,
    IngressProgressSnapshot, JsonMap, OperatorMetadata, ProviderRegistry, Result, StreamJobContext,
    StreamMessage, StreamOperatorContext, StreamRequirements, UdfRegistry,
};

use super::{AggregateInput, OperatorInputProgress, ProgressEmissionKind};

struct ReplayNode {
    node: RuntimeStreamNode,
    progress: OperatorInputProgress,
}

#[derive(Default)]
struct ReplayAudit {
    callbacks: BTreeMap<&'static str, u64>,
    node_callbacks: BTreeMap<String, BTreeMap<&'static str, u64>>,
    trace: Option<Vec<Value>>,
    handler_samples: Vec<Value>,
    collect_handlers: bool,
    node_data: BTreeMap<String, DataCounts>,
    output_groups: Option<Box<[usize]>>,
    output_group_count: usize,
}

#[derive(Default, serde::Serialize)]
struct DataCounts {
    input_batches: u64,
    input_rows: u64,
    fanned_out_batches: u64,
    fanned_out_rows: u64,
    errors: u64,
    ended: bool,
}

struct Replay {
    nodes: BTreeMap<String, ReplayNode>,
    parts: StreamRuntimePlanParts,
    job: StreamJobContext,
    output: Vec<RecordBatch>,
    output_rows: usize,
    output_binding: String,
    budget: EdgeBudget,
    order: Vec<String>,
    audit: ReplayAudit,
}

impl Replay {
    fn new(fixture: &Fixture) -> Result<Self> {
        let plan = crate::compile_stream_project_graph(
            &fixture.project,
            &ProviderRegistry::default(),
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )?;
        if fixture.manifest["compiled_plan_fingerprint"].as_str() != Some(plan.fingerprint()) {
            return Err(invalid("compiled plan fingerprint mismatch"));
        }
        let budget = EdgeBudget::new(
            usize::try_from(
                fixture.manifest["batch_rows"]
                    .as_u64()
                    .expect("validated batch size"),
            )
            .map_err(|_| invalid("batch size exceeds platform capacity"))?,
            64 << 20,
        )?;
        let mut parts = plan.into_runtime_parts(budget)?;
        validate_graph(&parts, fixture.manifest["scenario"] == "join_sealed")?;
        let output_binding = fixture.manifest["output_binding"]
            .as_str()
            .ok_or_else(|| invalid("missing output binding"))?
            .to_owned();
        if !parts.sink_routes.contains_key(&output_binding) {
            return Err(invalid("compiled output binding mismatch"));
        }
        let order = parts
            .nodes
            .iter()
            .map(|node| node.node_id.clone())
            .collect();
        let nodes = std::mem::take(&mut parts.nodes)
            .into_iter()
            .map(|node| {
                let progress = OperatorInputProgress::new(node.input_ports.keys());
                (node.node_id.clone(), ReplayNode { node, progress })
            })
            .collect();
        let job = StreamJobContext::new(
            1,
            &parts.fingerprint,
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        #[cfg(feature = "benchmark-checkpoint-off")]
        let job = job.with_checkpointing(
            fixture.manifest["checkpoint_enabled"]
                .as_bool()
                .expect("validated mode"),
        );
        Ok(Self {
            nodes,
            parts,
            job,
            output: Vec::new(),
            output_rows: 0,
            output_binding,
            budget,
            order,
            audit: ReplayAudit::default(),
        })
    }

    fn projection(&self) -> Value {
        self.order.iter().map(|name| &self.nodes[name]).map(|state| {
            let (kind, configuration) = match &state.node.operator {
                CompiledStreamOperator::StreamAsofJoin(operator) => ("stream_asof_join", operator.configuration()),
                CompiledStreamOperator::StreamJoin(operator) => ("stream_join", operator.configuration()),
                CompiledStreamOperator::Expression(operator) => ("expression", operator.configuration()),
                _ => unreachable!("preflighted graph"),
            };
            let inputs = state.node.input_ports.iter().map(|(name,port)| (name.clone(),json!(port.schema().map(|schema|schema.fields().iter().map(|field|field.name()).collect::<Vec<_>>())))).collect::<serde_json::Map<_,_>>();
            json!({"node_id":state.node.node_id,"kind":kind,"configuration":configuration,"input_fields":inputs,"output_fields":state.node.output_ports["output"].schema().map(|schema|schema.fields().iter().map(|field|field.name()).collect::<Vec<_>>())})
        }).collect::<Vec<_>>().into()
    }

    fn status(&self) -> Result<Value> {
        self.nodes
            .iter()
            .filter_map(|(name, state)| {
                let status = if let Some(status) = state.node.operator.stream_join_status() {
                    Some(serde_json::to_value(status))
                } else {
                    state
                        .node
                        .operator
                        .stream_asof_join_status()
                        .map(|mut status| {
                            status.output_watermark_micros = state.progress.output_frontier;
                            serde_json::to_value(status)
                        })
                };
                status.map(|status| {
                    status
                        .map(|status| (name.clone(), status))
                        .map_err(|error| invalid(error.to_string()))
                })
            })
            .collect::<Result<serde_json::Map<_, _>>>()
            .map(Value::Object)
    }

    fn output_rows(&self) -> usize {
        self.output_rows
    }

    async fn event(&mut self, event: &FixtureEvent) -> Result<()> {
        let route = self
            .parts
            .source_routes
            .get(&event.input)
            .ok_or_else(|| invalid("fixture input differs from compiled bindings"))?;
        let mut queue = VecDeque::from([(
            route.target.node_id.clone(),
            route.target.port.clone(),
            event.kind.clone(),
        )]);
        while let Some((node_id, ingress, kind)) = queue.pop_front() {
            let state = self
                .nodes
                .get_mut(&node_id)
                .ok_or_else(|| invalid("compiled edge names missing node"))?;
            let messages = Invocation {
                job: &self.job,
                audit: &mut self.audit,
                event,
                ingress: &ingress,
                node_id: &node_id,
                budget: self.budget,
            }
            .dispatch(state, kind)
            .await?;
            for message in messages {
                for edge in &state.node.output_edges["output"] {
                    match &self.parts.edges[edge].consumer {
                        RuntimeConsumer::Node { node_id, ingress } => {
                            queue.push_back((node_id.clone(), ingress.clone(), message.clone()));
                        }
                        RuntimeConsumer::Sink { output_id }
                            if *output_id == self.output_binding =>
                        {
                            if let EventKind::Data(batch) = &message {
                                let table = batch.table_payload()?;
                                if let Some(groups) = &mut self.audit.output_groups {
                                    *groups.get_mut(self.audit.output_group_count).ok_or_else(
                                        || invalid("dispatch output geometry capacity exceeded"),
                                    )? = table.batches().len();
                                    self.audit.output_group_count += 1;
                                }
                                self.output_rows += batch.num_rows();
                                self.output.extend(table.batches().iter().cloned());
                            }
                        }
                        RuntimeConsumer::Sink { .. } => {
                            return Err(invalid("unexpected compiled output"));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

struct Invocation<'a> {
    job: &'a StreamJobContext,
    audit: &'a mut ReplayAudit,
    event: &'a FixtureEvent,
    ingress: &'a str,
    node_id: &'a str,
    budget: EdgeBudget,
}

impl Invocation<'_> {
    async fn dispatch(
        &mut self,
        state: &mut ReplayNode,
        kind: EventKind,
    ) -> Result<Vec<EventKind>> {
        if matches!(
            state.node.operator,
            CompiledStreamOperator::StreamAsofJoin(_)
        ) {
            self.job.dispatch_trace().record(
                self.node_id,
                self.ingress,
                &dispatch::message(&kind),
            )?;
        }
        let watermark = if let EventKind::Watermark(at) = &kind {
            Some(*at)
        } else {
            None
        };
        let previous = state.progress.input_watermark();
        let aggregate = match &kind {
            EventKind::Data(_) => AggregateInput::Data,
            EventKind::Watermark(at) => AggregateInput::Watermark(*at),
            EventKind::Idle => AggregateInput::Idle,
            EventKind::End => AggregateInput::End,
        };
        let emissions = state.progress.evaluate(self.ingress, aggregate)?;
        let snapshot = state.progress.snapshot()?;
        let mut collector = EdgeCollector::new(state.node.output_ports.values().cloned().collect());
        let mut pending = Vec::new();
        let controls = if let EventKind::Data(batch) = kind {
            if let Some(counts) = self.audit.node_data.get_mut(self.node_id) {
                counts.input_batches += 1;
                counts.input_rows += u64::try_from(batch.num_rows()).expect("fixture rows fit u64");
            }
            let context = StreamOperatorContext::with_ingress_progress(
                self.job,
                self.node_id,
                previous,
                snapshot,
            )
            .with_output_budget(self.budget);
            self.data(state, batch, &context, &mut collector, &mut pending)
                .await?;
            Vec::new()
        } else {
            self.progress(
                state,
                previous,
                snapshot,
                emissions,
                &mut collector,
                &mut pending,
            )
            .await?
        };
        let messages = pending
            .into_iter()
            .chain(collector.drain("output"))
            .map(|message| {
                EventKind::Data(message.as_data().expect("collector emits data").clone())
            })
            .chain(controls)
            .collect::<Vec<_>>();
        if let Some(counts) = self.audit.node_data.get_mut(self.node_id) {
            for message in &messages {
                if let EventKind::Data(batch) = message {
                    counts.fanned_out_batches += 1;
                    counts.fanned_out_rows +=
                        u64::try_from(batch.num_rows()).expect("fixture rows fit u64");
                }
            }
        }
        if let Some(at) = watermark {
            self.job
                .dispatch_trace()
                .complete_watermark(self.node_id, at.as_micros());
        }
        Ok(messages)
    }

    async fn data(
        &mut self,
        state: &mut ReplayNode,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        collector: &mut EdgeCollector,
        pending: &mut Vec<StreamMessage>,
    ) -> Result<()> {
        let before = trace_status(&state.node.operator, self.audit.trace.is_some())?;
        let timed = self.audit.collect_handlers
            && self.ingress == "left"
            && matches!(state.node.operator, CompiledStreamOperator::StreamJoin(_));
        let rows = batch.num_rows();
        let started = timed.then(Instant::now);
        state
            .node
            .operator
            .process_data(self.ingress, batch, context, collector)
            .await?;
        if let Some(started) = started {
            let seconds = started.elapsed().as_secs_f64();
            let status = state
                .node
                .operator
                .stream_join_status()
                .expect("timed Join");
            if status.left.retained_rows != 0 {
                return Err(invalid("sealed Join retained left rows after its handler"));
            }
            self.audit.handler_samples.push(json!({"rows":rows,"seconds":seconds,"left_retained_rows":status.left.retained_rows,"right_retained_rows":status.right.retained_rows}));
        }
        self.record(state, "data", &before, collector, pending)
    }

    async fn progress(
        &mut self,
        state: &mut ReplayNode,
        previous: Option<EventTime>,
        snapshot: IngressProgressSnapshot,
        emissions: Vec<ProgressEmissionKind>,
        collector: &mut EdgeCollector,
        pending: &mut Vec<StreamMessage>,
    ) -> Result<Vec<EventKind>> {
        let context = StreamOperatorContext::with_ingress_progress(
            self.job,
            self.node_id,
            previous,
            snapshot.clone(),
        )
        .with_output_budget(self.budget);
        let before = trace_status(&state.node.operator, self.audit.trace.is_some())?;
        state
            .node
            .operator
            .on_ingress_progress(self.ingress, &context, collector)
            .await?;
        self.record(state, "ingress_progress", &before, collector, pending)?;
        let mut previous_watermark = previous;
        let mut ended = false;
        let mut idle = false;
        for emission in emissions {
            match emission {
                ProgressEmissionKind::Watermark(at) => {
                    let before = trace_status(&state.node.operator, self.audit.trace.is_some())?;
                    let context = StreamOperatorContext::with_ingress_progress(
                        self.job,
                        self.node_id,
                        previous_watermark,
                        snapshot.clone(),
                    )
                    .with_output_budget(self.budget);
                    state
                        .node
                        .operator
                        .on_watermark(at, &context, collector)
                        .await?;
                    self.record(state, "watermark", &before, collector, pending)?;
                    previous_watermark = Some(at);
                }
                ProgressEmissionKind::Idle => idle = true,
                ProgressEmissionKind::EndOfInput => ended = true,
            }
        }
        let mut controls = Vec::new();
        if let Some(at) = state
            .node
            .operator
            .output_frontier_candidate(state.progress.input_watermark(), &snapshot)?
            .filter(|at| {
                state
                    .progress
                    .output_frontier
                    .is_none_or(|previous| *at > previous)
            })
        {
            state.progress.output_frontier = Some(at);
            controls.push(EventKind::Watermark(at));
        }
        if idle && !state.node.operator.suppresses_output_idle() {
            controls.push(EventKind::Idle);
        }
        if ended {
            let before = trace_status(&state.node.operator, self.audit.trace.is_some())?;
            let context = StreamOperatorContext::with_ingress_progress(
                self.job,
                self.node_id,
                state.progress.input_watermark(),
                snapshot,
            )
            .with_output_budget(self.budget);
            state.node.operator.on_end(&context, collector).await?;
            self.record(state, "end", &before, collector, pending)?;
            controls.push(EventKind::End);
        }
        Ok(controls)
    }

    fn record(
        &mut self,
        state: &ReplayNode,
        callback: &'static str,
        before: &Value,
        collector: &mut EdgeCollector,
        pending: &mut Vec<StreamMessage>,
    ) -> Result<()> {
        if matches!(
            state.node.operator,
            CompiledStreamOperator::StreamAsofJoin(_)
        ) && callback != "end"
        {
            self.job.dispatch_trace().callback(self.node_id, callback)?;
        }
        *self.audit.callbacks.entry(callback).or_default() += 1;
        *self
            .audit
            .node_callbacks
            .entry(state.node.node_id.clone())
            .or_default()
            .entry(callback)
            .or_default() += 1;
        let Some(trace) = &mut self.audit.trace else {
            return Ok(());
        };
        let messages = collector.drain("output");
        let geometry = messages
            .iter()
            .map(|message| {
                message
                    .as_data()
                    .expect("collector data")
                    .table_payload()
                    .map(|table| {
                        table
                            .batches()
                            .iter()
                            .map(RecordBatch::num_rows)
                            .collect::<Vec<_>>()
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        trace.push(json!({"event_id":self.event.id,"chunk_id":self.event.chunk,"node_id":state.node.node_id,"ingress":self.ingress,"callback":callback,"before":before,"after":trace_status(&state.node.operator,true)?,"emitted_batch_geometry":geometry}));
        pending.extend(messages);
        Ok(())
    }
}

fn trace_status(operator: &CompiledStreamOperator, enabled: bool) -> Result<Value> {
    if !enabled {
        return Ok(Value::Null);
    }
    serde_json::to_value(operator.stream_asof_join_status())
        .map_err(|error| invalid(error.to_string()))
}

fn fifo_orders(events: &[FixtureEvent]) -> Result<Vec<Vec<FixtureEvent>>> {
    let mut groups = BTreeMap::<String, Vec<FixtureEvent>>::new();
    for event in events {
        groups
            .entry(event.input.clone())
            .or_default()
            .push(event.clone());
    }
    let groups = groups.into_values().collect::<Vec<_>>();
    if groups.len() != 2
        || groups.iter().any(|group| {
            group.len() != 2
                || !matches!(group[0].kind, EventKind::Data(_))
                || !matches!(group[1].kind, EventKind::Watermark(_))
        })
    {
        return Err(invalid(
            "ordering proof requires Data then Watermark on each source",
        ));
    }
    let mut orders = Vec::new();
    for first in 0..4 {
        for second in first + 1..4 {
            let mut next = [0, 0];
            let mut order = Vec::new();
            for position in 0..4 {
                let side = usize::from(position != first && position != second);
                order.push(groups[side][next[side]].clone());
                next[side] += 1;
            }
            orders.push(order);
        }
    }
    Ok(orders)
}

pub(crate) async fn prove_ordering(fixture: Fixture) -> Result<Value> {
    let mut chunks = BTreeMap::<u64, Vec<FixtureEvent>>::new();
    for event in &fixture.processing {
        chunks
            .entry(
                event
                    .chunk
                    .ok_or_else(|| invalid("proof event has no chunk"))?,
            )
            .or_default()
            .push(event.clone());
    }
    let chunks = chunks.into_values().collect::<Vec<_>>();
    if chunks.len() != 3 {
        return Err(invalid(
            "ordering proof requires two full chunks and one final partial chunk",
        ));
    }
    let batch_rows = usize::try_from(
        fixture.manifest["batch_rows"]
            .as_u64()
            .expect("validated batch rows"),
    )
    .map_err(|_| invalid("batch size exceeds platform capacity"))?;
    let rows = |chunk: &[FixtureEvent]| {
        chunk
            .iter()
            .find_map(|event| {
                if let EventKind::Data(batch) = &event.kind {
                    Some(batch.num_rows())
                } else {
                    None
                }
            })
            .unwrap_or(0)
    };
    if rows(&chunks[0]) != batch_rows
        || rows(&chunks[1]) != batch_rows
        || rows(&chunks[2]) == 0
        || rows(&chunks[2]) >= batch_rows
    {
        return Err(invalid(
            "ordering proof has invalid full/final chunk geometry",
        ));
    }
    let orders0 = fifo_orders(&chunks[0])?;
    let orders1 = fifo_orders(&chunks[1])?;
    let mut schedules = Vec::new();
    let mut projection = Value::Null;
    for (first, order0) in orders0.iter().enumerate() {
        for (second, order1) in orders1.iter().enumerate() {
            let mut replay = Replay::new(&fixture)?;
            projection = replay.projection();
            replay.audit.trace = Some(Vec::new());
            let result = prove_schedule(
                &mut replay,
                &fixture,
                [order0.as_slice(), order1.as_slice(), chunks[2].as_slice()],
                first,
                second,
            )
            .await;
            if result.is_err() {
                replay.job.cancellation().cancel();
            }
            replay.nodes.clear();
            let cleanup = settle_job(&replay.job).await;
            let sample = result?;
            cleanup?;
            schedules.push(sample);
        }
    }
    let expected = &schedules[0]["rendezvous"];
    let equivalent = schedules
        .iter()
        .all(|sample| &sample["rendezvous"] == expected);
    let trace_identical = schedules
        .iter()
        .all(|sample| sample["callback_trace"] == schedules[0]["callback_trace"]);
    let manifest = &fixture.manifest;
    Ok(json!({
        "schema":"calc-flow.native-ordering-proof.v1","schedule_count":36,"rendezvous_equivalent":equivalent,"callback_traces_identical":trace_identical,
        "all_outputs_validated":true,"all_owned_cleanup_complete":true,"final_partial_rows":rows(&chunks[2]),"schedules":schedules,
        "project_sha256":manifest["project"]["sha256"],"compiled_plan_fingerprint":manifest["compiled_plan_fingerprint"],"compiled_projection":projection,"compiled_projection_sha256":fingerprint(&projection)?,
        "inputs_sha256":manifest["inputs_sha256"],"shared_partial_order_sha256":manifest["shared_partial_order_sha256"],"canonical_native_events_sha256":manifest["canonical_native_events_sha256"],
        "batch_rows":batch_rows,"rows_per_side":manifest["rows_per_side"],"checkpoint_enabled":manifest["checkpoint_enabled"],
        "claim":"observable rendezvous equivalence; callback traces retained; no CPU-work equivalence claim"
    }))
}

async fn prove_schedule(
    replay: &mut Replay,
    fixture: &Fixture,
    chunks: [&[FixtureEvent]; 3],
    first: usize,
    second: usize,
) -> Result<Value> {
    let mut rendezvous = Vec::new();
    let mut expected_rows = 0;
    for chunk in chunks {
        for event in chunk {
            replay.event(event).await?;
        }
        expected_rows += chunk
            .iter()
            .find_map(|event| {
                if let EventKind::Data(batch) = &event.kind {
                    Some(batch.num_rows())
                } else {
                    None
                }
            })
            .expect("proof data event");
        if replay.output_rows() != expected_rows {
            return Err(invalid("ordering proof rendezvous output differs"));
        }
        rendezvous.push(json!({"status":replay.status()?,"output_rows":replay.output_rows(),"output_batch_rows":replay.output.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),"callback_counts":replay.audit.callbacks,"node_callback_counts":replay.audit.node_callbacks}));
    }
    validate_output(&replay.output, &fixture.expected)?;
    for event in &fixture.terminal {
        replay.event(event).await?;
    }
    validate_output(&replay.output, &fixture.expected)?;
    Ok(
        json!({"first_order":first,"second_order":second,"rendezvous":rendezvous,"callback_trace":replay.audit.trace.take().expect("proof tracing"),"output_rows":replay.output_rows(),"validated_all_rows":true}),
    )
}

fn validate_graph(parts: &StreamRuntimePlanParts, sealed_join: bool) -> Result<()> {
    let joins = parts
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.operator,
                CompiledStreamOperator::StreamAsofJoin(_) | CompiledStreamOperator::StreamJoin(_)
            )
        })
        .collect::<Vec<_>>();
    if joins.len() != 1
        || !matches!(parts.nodes.len(), 2 | 4)
        || parts.source_routes.len() != 2
        || parts.sink_routes.len() != 1
        || parts.edges.len() != parts.nodes.len() + 2
    {
        return Err(invalid(
            "matched ASOF requires two optional input expressions, one ASOF and one output expression",
        ));
    }
    let join = joins[0];
    if sealed_join != matches!(join.operator, CompiledStreamOperator::StreamJoin(_)) {
        return Err(invalid("compiled join kind differs from fixture scenario"));
    }
    if join
        .input_ports
        .keys()
        .map(String::as_str)
        .ne(["left", "right"])
        || join.output_ports.len() != 1
    {
        return Err(invalid("unexpected ASOF ports"));
    }
    let output_id = &parts
        .sink_routes
        .values()
        .next()
        .expect("one sink")
        .source
        .node_id;
    let sources = parts
        .source_routes
        .values()
        .map(|route| route.target.node_id.as_str())
        .collect::<BTreeSet<_>>();
    for node in &parts.nodes {
        if node.output_edges.len() != 1 || node.output_edges.values().any(|edges| edges.len() != 1)
        {
            return Err(invalid("matched graph forbids fanout and side outputs"));
        }
        if node.node_id == join.node_id {
            continue;
        }
        if !matches!(node.operator, CompiledStreamOperator::Expression(_))
            || node.input_ports.len() != 1
        {
            return Err(invalid("matched graph contains an unexpected operator"));
        }
        let input_edge = &parts.edges[node.ingress_edges.values().next().expect("one ingress")];
        let output_edge = &parts.edges[&node.output_edges.values().next().expect("one output")[0]];
        if &node.node_id == output_id {
            if !matches!(&input_edge.producer,RuntimeProducer::Node{node_id,..} if node_id==&join.node_id)
                || !matches!(output_edge.consumer, RuntimeConsumer::Sink { .. })
            {
                return Err(invalid("output projection must directly follow ASOF"));
            }
        } else if !sources.contains(node.node_id.as_str())
            || !matches!(input_edge.producer, RuntimeProducer::Source { .. })
            || !matches!(&output_edge.consumer,RuntimeConsumer::Node{node_id,..} if node_id==&join.node_id)
        {
            return Err(invalid("input expressions must directly feed ASOF"));
        }
    }
    Ok(())
}

pub(crate) async fn run(
    mut fixture: Fixture,
    check_only: bool,
    setup_start: Instant,
) -> Result<Value> {
    let mut replay = Replay::new(&fixture)?;
    dispatch::prepare(&replay, &mut fixture)?;
    let projection = replay.projection();
    let result = run_owned(&mut replay, &fixture, check_only, setup_start, projection).await;
    let cleanup_start = Instant::now();
    if result.is_err() {
        replay.job.cancellation().cancel();
    }
    replay.nodes.clear();
    let cleanup = settle_job(&replay.job).await;
    let cleanup_seconds = cleanup_start.elapsed().as_secs_f64();
    let mut result = result?;
    cleanup?;
    result["cleanup_seconds"] = cleanup_seconds.into();
    result["owned_cleanup_complete"] = true.into();
    Ok(result)
}

async fn run_owned(
    replay: &mut Replay,
    fixture: &Fixture,
    check_only: bool,
    setup_start: Instant,
    projection: Value,
) -> Result<Value> {
    for event in &fixture.preload {
        replay.event(event).await?;
    }
    let preload_callbacks = replay.audit.callbacks.clone();
    let preload_node_callbacks = replay.audit.node_callbacks.clone();
    dispatch::start(replay, fixture)?;
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    replay.audit.collect_handlers = fixture.manifest["scenario"] == "join_sealed";
    let processing_start = Instant::now();
    for event in &fixture.processing {
        replay.event(event).await?;
        if let Some(rows) = fixture.barriers.get(&event.id) {
            if replay.output_rows() != *rows {
                return Err(invalid("native rendezvous output count mismatch"));
            }
        }
    }
    let assembled = Batch::table(replay.output.clone(), BatchMetadata::default())?;
    let processing_seconds = processing_start.elapsed().as_secs_f64();
    let dispatch_trace = dispatch::finish(replay, fixture)?;
    replay.audit.collect_handlers = false;
    let post_processing_status = replay.status()?;
    let processing_callbacks = callback_delta(&replay.audit.callbacks, &preload_callbacks);
    let processing_node_callbacks =
        node_callback_delta(&replay.audit.node_callbacks, &preload_node_callbacks);
    let before_terminal_callbacks = replay.audit.callbacks.clone();
    let before_terminal_node_callbacks = replay.audit.node_callbacks.clone();
    let processing_rows = replay.output_rows();
    let terminal_start = Instant::now();
    for event in &fixture.terminal {
        replay.event(event).await?;
    }
    let terminal_seconds = terminal_start.elapsed().as_secs_f64();
    if replay.output_rows() != processing_rows {
        return Err(invalid("matched output required EOF to complete"));
    }
    let oracle_start = Instant::now();
    validate_output(&replay.output, &fixture.expected)?;
    let oracle_seconds = oracle_start.elapsed().as_secs_f64();
    let manifest = &fixture.manifest;
    let mut result = json!({
        "schema":"calc-flow.matched-native-observation.v1","scenario":manifest["scenario"],"check_only":check_only,
        "checkpoint_enabled":manifest["checkpoint_enabled"],"nonterminal_epochs":0,
        "rows_per_side":manifest["rows_per_side"],"rows":manifest["rows"],"dimension_rows":manifest["dimension_rows"],"batch_rows":manifest["batch_rows"],
        "handler_samples":replay.audit.handler_samples,
        "output_budget":{"max_rows":replay.budget.max_rows,"max_bytes":replay.budget.max_bytes},
        "setup_seconds":setup_seconds,"processing_seconds":processing_seconds,"terminal_seconds":terminal_seconds,"oracle_seconds":oracle_seconds,
        "output_rows":processing_rows,"validated_all_rows":true,
        "assembled_output_rows":assembled.num_rows(),"assembled_output_batches":assembled.table_payload()?.batches().len(),
        "project_sha256":manifest["project"]["sha256"],"inputs_sha256":manifest["inputs_sha256"],"events_sha256":manifest["events"]["sha256"],
        "shared_partial_order_sha256":manifest["shared_partial_order_sha256"],"canonical_native_events_sha256":manifest["canonical_native_events_sha256"],
        "output_contract_sha256":manifest["expected_output"]["sha256"],"compiled_plan_fingerprint":manifest["compiled_plan_fingerprint"],
        "compiled_projection_sha256":fingerprint(&projection)?,"compiled_projection":projection,"post_processing_status":post_processing_status,"callback_counts":replay.audit.callbacks,"node_callback_counts":replay.audit.node_callbacks,
        "preload_callback_counts":preload_callbacks,"processing_callback_counts":processing_callbacks,"processing_node_callback_counts":processing_node_callbacks,
        "terminal_callback_counts":callback_delta(&replay.audit.callbacks,&before_terminal_callbacks),"terminal_node_callback_counts":node_callback_delta(&replay.audit.node_callbacks,&before_terminal_node_callbacks),
    });
    if let Some(trace) = dispatch_trace {
        let source = fixture.dispatch_replay.as_ref().expect("dispatch source");
        result["dispatch_trace"] = trace;
        result["dispatch_trace_verified"] = true.into();
        result["dispatch_replay_state_verified"] = true.into();
        result["replay_source_observation_id"] = source.envelope["source_observation_id"].clone();
        result["replay_source_raw_sha256"] = source.envelope["source_raw_sha256"].clone();
        result["replay_trace_sha256"] = source.sha256.clone().into();
    }
    Ok(result)
}

fn callback_delta(
    after: &BTreeMap<&'static str, u64>,
    before: &BTreeMap<&'static str, u64>,
) -> BTreeMap<&'static str, u64> {
    after
        .iter()
        .filter_map(|(name, count)| {
            let difference = count - before.get(name).copied().unwrap_or_default();
            (difference > 0).then_some((*name, difference))
        })
        .collect()
}

fn node_callback_delta(
    after: &BTreeMap<String, BTreeMap<&'static str, u64>>,
    before: &BTreeMap<String, BTreeMap<&'static str, u64>>,
) -> BTreeMap<String, BTreeMap<&'static str, u64>> {
    after
        .iter()
        .map(|(node, counts)| {
            (
                node.clone(),
                callback_delta(counts, before.get(node).unwrap_or(&BTreeMap::new())),
            )
        })
        .collect()
}

fn validate_output(actual: &[RecordBatch], expected: &[RecordBatch]) -> Result<()> {
    if actual.iter().chain(expected).any(|record| {
        record
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .ne(["sequence", "value"])
    }) {
        return Err(invalid(
            "native and oracle projection order must be sequence, value",
        ));
    }
    let rows = expected.iter().map(RecordBatch::num_rows).sum::<usize>();
    let mut values = vec![None; rows];
    for record in expected {
        let sequence = record
            .column_by_name("sequence")
            .and_then(|array| array.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| invalid("oracle sequence must be UInt64"))?;
        let value = record
            .column_by_name("value")
            .and_then(|array| array.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| invalid("oracle value must be Float64"))?;
        for row in 0..record.num_rows() {
            let index = usize::try_from(sequence.value(row))
                .map_err(|_| invalid("oracle index overflow"))?;
            if sequence.is_null(row)
                || value.is_null(row)
                || index >= rows
                || values[index].replace(value.value(row)).is_some()
            {
                return Err(invalid("invalid oracle sequence"));
            }
        }
    }
    let mut seen = vec![false; rows];
    for record in actual {
        if record.num_columns() != 2 {
            return Err(invalid("native output projection has extra columns"));
        }
        let sequence = record
            .column_by_name("sequence")
            .and_then(|array| array.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| invalid("native sequence must be UInt64"))?;
        let value = record
            .column_by_name("value")
            .and_then(|array| array.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| invalid("native value must be Float64"))?;
        for row in 0..record.num_rows() {
            let index = usize::try_from(sequence.value(row))
                .map_err(|_| invalid("native index overflow"))?;
            if sequence.is_null(row)
                || value.is_null(row)
                || index >= rows
                || seen[index]
                || values[index] != Some(value.value(row))
            {
                return Err(invalid("native output differs from full oracle"));
            }
            seen[index] = true;
        }
    }
    if seen.iter().any(|seen| !*seen) {
        return Err(invalid("native output is incomplete"));
    }
    Ok(())
}
