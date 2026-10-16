use std::sync::Arc;

use datafusion::arrow::array::{Array, UInt64Array};
use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::{Result, StreamMessage, StreamMessageKind};

use super::invalid;

const CAPACITY: usize = 8192;

#[derive(Clone, Copy, Default)]
struct Record {
    kind: u8,
    side: usize,
    ordinal: u64,
    rows: usize,
    first: u64,
    last: u64,
    batch_rows: [usize; 4],
    batches: usize,
    micros: i64,
    epoch: u64,
}

struct Trace {
    job_id: u64,
    fingerprint: String,
    node: String,
    allow_barriers: bool,
    records: Box<[Record]>,
    len: usize,
    ordinals: [u64; 2],
    callbacks: [u64; 3],
    invalid: Option<&'static str>,
    frontiers: Vec<(String, Option<i64>)>,
}

#[derive(Default)]
enum State {
    #[default]
    Disabled,
    Active(Trace),
    Sealed,
}

/// Feature-only job-owned fixed-capacity ASOF dispatch recorder.
#[derive(Clone, Default)]
pub struct DispatchTraceOwner(Arc<Mutex<State>>);

impl DispatchTraceOwner {
    /// Allocates the recording storage before benchmark timing starts.
    ///
    /// # Errors
    /// Rejects reuse or empty identity fields.
    pub fn start(
        &self,
        job_id: u64,
        fingerprint: &str,
        node: &str,
        allow_barriers: bool,
    ) -> Result<()> {
        let mut state = self.0.lock();
        if !matches!(*state, State::Disabled) || fingerprint.is_empty() || node.is_empty() {
            return Err(invalid("dispatch trace cannot be started"));
        }
        *state = State::Active(Trace {
            job_id,
            fingerprint: fingerprint.into(),
            node: node.into(),
            allow_barriers,
            records: vec![Record::default(); CAPACITY].into_boxed_slice(),
            len: 0,
            ordinals: [0; 2],
            callbacks: [0; 3],
            invalid: None,
            frontiers: Vec::new(),
        });
        Ok(())
    }

    /// Installs the fixed node-completion slots before any feed is admitted.
    ///
    /// # Errors
    /// Rejects a disabled, reused, or already populated recorder.
    pub fn register_nodes(&self, nodes: &[String]) -> Result<()> {
        let mut state = self.0.lock();
        let State::Active(trace) = &mut *state else {
            return Err(invalid("dispatch trace is not active"));
        };
        if trace.len != 0
            || !trace.frontiers.is_empty()
            || nodes.is_empty()
            || nodes.iter().any(|node| node.trim().is_empty())
            || nodes
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != nodes.len()
        {
            return Err(invalid("dispatch node slots already in use"));
        }
        trace.frontiers = nodes.iter().map(|node| (node.clone(), None)).collect();
        Ok(())
    }

    /// Records one actual ASOF ingress before dispatch, without growing storage.
    ///
    /// # Errors
    /// Rejects overflow, unsupported controls, ingress names, or data geometry.
    pub fn record(&self, node: &str, ingress: &str, message: &StreamMessage) -> Result<()> {
        let mut state = self.0.lock();
        let State::Active(trace) = &mut *state else {
            return Ok(());
        };
        if trace.node != node {
            return Ok(());
        }
        let result = trace.record(ingress, message);
        if result.is_err() {
            trace.invalid = Some("dispatch trace contains an invalid event");
        }
        result
    }

    /// Counts an actually completed ASOF callback.
    ///
    /// # Errors
    /// Rejects unrecognized callback names.
    pub fn callback(&self, node: &str, callback: &str) -> Result<()> {
        let mut state = self.0.lock();
        let State::Active(trace) = &mut *state else {
            return Ok(());
        };
        if trace.node != node {
            return Ok(());
        }
        let index = match callback {
            "data" => 0,
            "ingress_progress" => 1,
            "watermark" => 2,
            _ => return Err(invalid("unsupported dispatch callback")),
        };
        trace.callbacks[index] += 1;
        Ok(())
    }

    /// Marks a complete, successful watermark dispatch after all its callbacks.
    pub fn complete_watermark(&self, node: &str, micros: i64) {
        if let State::Active(trace) = &mut *self.0.lock()
            && let Some((_, frontier)) = trace.frontiers.iter_mut().find(|(name, _)| name == node)
        {
            *frontier = Some(micros);
        }
    }

    /// Returns a node's completed watermark dispatch frontier.
    pub fn frontier(&self, node: &str) -> Option<i64> {
        let state = self.0.lock();
        let State::Active(trace) = &*state else {
            return None;
        };
        trace
            .frontiers
            .iter()
            .find(|(name, _)| name == node)
            .and_then(|(_, frontier)| *frontier)
    }

    /// Seals and serializes the recorder once, after benchmark timing stops.
    ///
    /// # Errors
    /// Rejects disabled, repeated, or invalid captures.
    pub fn take(&self) -> Result<Value> {
        let mut state = self.0.lock();
        if !matches!(*state, State::Active(_)) {
            return Err(invalid("dispatch trace is not active"));
        }
        let State::Active(trace) = std::mem::replace(&mut *state, State::Sealed) else {
            unreachable!()
        };
        drop(state);
        if let Some(reason) = trace.invalid {
            return Err(invalid(reason));
        }
        let events = trace.records[..trace.len]
            .iter()
            .map(Record::json)
            .collect::<Vec<_>>();
        Ok(json!({
            "schema":"calc-flow.asof-dispatch-trace.v1", "job_id":trace.job_id,
            "compiled_plan_fingerprint":trace.fingerprint, "node_id":trace.node,
            "capacity":CAPACITY, "record_count":trace.len, "overflowed":false,
            "allow_barriers":trace.allow_barriers, "terminal_excluded":true, "events":events,
            "callback_counts":{"data":trace.callbacks[0],"ingress_progress":trace.callbacks[1],"watermark":trace.callbacks[2]},
            "completed_frontiers":trace.frontiers.into_iter().map(|(node, at)| (node,json!(at))).collect::<serde_json::Map<_,_>>()
        }))
    }
}

impl Trace {
    fn record(&mut self, ingress: &str, message: &StreamMessage) -> Result<()> {
        if self.len == CAPACITY {
            return Err(invalid("dispatch trace capacity exceeded"));
        }
        let side = match ingress {
            "left" => 0,
            "right" => 1,
            _ => return Err(invalid("unexpected ASOF ingress")),
        };
        let mut record = Record {
            side,
            ..Record::default()
        };
        match message.kind() {
            StreamMessageKind::Data => {
                record.kind = 0;
                record.ordinal = self.ordinals[side];
                record.read_data(message)?;
                self.ordinals[side] += 1;
            }
            StreamMessageKind::Watermark => {
                record.kind = 1;
                record.micros = message.as_watermark().expect("watermark").as_micros();
            }
            StreamMessageKind::Barrier if self.allow_barriers => {
                record.kind = 2;
                record.epoch = message.as_barrier().expect("barrier").as_u64();
            }
            _ => return Err(invalid("unexpected control inside processing trace")),
        }
        self.records[self.len] = record;
        self.len += 1;
        Ok(())
    }
}

impl Record {
    fn read_data(&mut self, message: &StreamMessage) -> Result<()> {
        let batch = message.as_data().expect("data");
        let table = batch.table_payload()?;
        if table.batches().is_empty()
            || table.batches().len() > self.batch_rows.len()
            || batch.num_rows() == 0
        {
            return Err(invalid("unsupported dispatch record batch geometry"));
        }
        self.rows = batch.num_rows();
        self.batches = table.batches().len();
        for (index, batch) in table.batches().iter().enumerate() {
            self.read_record_batch(index, batch)?;
        }
        Ok(())
    }

    fn read_record_batch(
        &mut self,
        index: usize,
        batch: &datafusion::arrow::record_batch::RecordBatch,
    ) -> Result<()> {
        let sequence = batch
            .column_by_name("sequence")
            .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| invalid("dispatch sequence must be UInt64"))?;
        if sequence.is_empty() || sequence.null_count() != 0 {
            return Err(invalid("dispatch sequence must be populated"));
        }
        self.batch_rows[index] = batch.num_rows();
        if index == 0 {
            self.first = sequence.value(0);
        }
        self.last = sequence.value(sequence.len() - 1);
        Ok(())
    }

    fn json(&self) -> Value {
        let ingress = ["left", "right"][self.side];
        match self.kind {
            0 => {
                json!({"kind":"data","ingress":ingress,"ordinal":self.ordinal,"rows":self.rows,"first_sequence":self.first,"last_sequence":self.last,"record_batch_rows":self.batch_rows[..self.batches]})
            }
            1 => json!({"kind":"watermark","ingress":ingress,"micros":self.micros}),
            2 => json!({"kind":"barrier","ingress":ingress,"epoch":self.epoch}),
            _ => unreachable!("fixed record kind"),
        }
    }
}
