use super::{
    Anchor, AnchorControl, CONTROL_ID, CONTROL_MAX_BYTES, Callback, Control, Descriptor, Frame,
    Log, MAX_FRAMES, Record, SIDES, codec, mismatch,
};
use crate::{
    Batch, BatchMetadata, EventTime, IngressProgressSnapshot, OperatorStateSnapshot, Result,
    SourceEvent, StreamAsofJoinOperator, StreamCollector, StreamJobContext, StreamOperator,
    StreamOperatorContext, StreamSource,
};
use async_trait::async_trait;
use futures::FutureExt;
use std::{collections::BTreeSet, panic::AssertUnwindSafe};

struct Discard;

#[async_trait]
impl StreamCollector for Discard {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Ok(())
    }
}

impl StreamAsofJoinOperator {
    pub(in crate::operator::asof) async fn restore_source_replay(
        mut self: Box<Self>,
        snapshot: OperatorStateSnapshot,
        progress: IngressProgressSnapshot,
        _frontier: Option<EventTime>,
        job: &StreamJobContext,
    ) -> Result<Box<Self>> {
        self.stop_replay()?;
        let (control, log) = self.replay_control_log(&snapshot)?;
        self = self.restore_replay_base(&log, job).await?;
        let mut readers = self.replay_readers()?;
        self.replay_and_close(&mut readers, &log, job).await?;
        self.finish_replay(&progress, &control, log)?;
        job.check_cancelled()?;
        Ok(self)
    }

    fn replay_control_log(&self, snapshot: &OperatorStateSnapshot) -> Result<(Control, Log)> {
        let control_segment = snapshot
            .segments
            .get(CONTROL_ID)
            .ok_or_else(|| mismatch("missing control segment"))?;
        let bytes = control_segment.bytes_arc();
        if bytes.len() > CONTROL_MAX_BYTES {
            return Err(mismatch("control exceeds its size limit"));
        }
        let workspace = self.reserve_workspace((bytes.len() * 128 + 4096) as u64)?;
        let control = decode_control(&bytes)?;
        self.validate_replay_control(snapshot, &control)?;
        let log = self.decode_replay_log(snapshot, &control)?;
        drop(workspace);
        Ok((control, log))
    }

    async fn restore_replay_base(
        mut self: Box<Self>,
        log: &Log,
        job: &StreamJobContext,
    ) -> Result<Box<Self>> {
        if let Some(anchor) = &log.anchor {
            let progress = Self::replay_progress(&anchor.records[0]);
            let metrics = anchor_metrics(anchor)?;
            self = self
                .restore_native_managed(
                    anchor.snapshot.clone(),
                    progress,
                    metrics.output_watermark_micros,
                    job,
                    None,
                )
                .await?;
            self.clear_anchor_bookkeeping();
            self.status.state_bytes = self.current_inventory(None)?.bytes;
        }
        Ok(self)
    }

    fn replay_readers(&self) -> Result<[Box<dyn StreamSource>; 2]> {
        let inputs = self
            .replay_inputs
            .as_ref()
            .ok_or_else(|| mismatch("sources no longer support replay"))?;
        Ok([
            inputs.factories[0].create(inputs.histories[0].fork())?,
            inputs.factories[1].create(inputs.histories[1].fork())?,
        ])
    }

    async fn replay_and_close(
        &mut self,
        readers: &mut [Box<dyn StreamSource>; 2],
        log: &Log,
        job: &StreamJobContext,
    ) -> Result<()> {
        let result = AssertUnwindSafe(self.replay_callbacks(readers, log, job))
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err(mismatch("reader or callback panicked")));
        let mut close_error = None;
        for reader in readers {
            let closed = AssertUnwindSafe(reader.close())
                .catch_unwind()
                .await
                .unwrap_or_else(|_| Err(mismatch("reader close panicked")));
            if let Err(error) = closed {
                close_error.get_or_insert(error);
            }
        }
        result?;
        if let Some(error) = close_error {
            return Err(error);
        }
        Ok(())
    }

    fn finish_replay(
        &mut self,
        progress: &IngressProgressSnapshot,
        control: &Control,
        mut log: Log,
    ) -> Result<()> {
        self.observe(progress);
        log.cut = log.records.len();
        self.replay = Some(Box::new(log));
        self.status.state_bytes = self.current_inventory(None)?.bytes;
        self.status.output_watermark_micros = control.status.output_watermark_micros;
        self.check_rebuilt_replay(control)?;
        self.check_inventory_limits(&self.current_inventory(None)?)
    }

    fn check_rebuilt_replay(&self, control: &Control) -> Result<()> {
        if self.status != control.status
            || self.terminal != control.terminal
            || self.next_output_sequence != control.next_output_sequence
        {
            return Err(mismatch(
                "rebuilt counters, capacity or output coordinate differs",
            ));
        }
        Ok(())
    }

    fn validate_replay_control(
        &self,
        snapshot: &OperatorStateSnapshot,
        control: &Control,
    ) -> Result<()> {
        let inputs = self
            .replay_inputs
            .as_ref()
            .ok_or_else(|| mismatch("sources no longer support replay"))?;
        if !valid_replay_metadata(snapshot)
            || !valid_replay_identity(snapshot, control, &self.fingerprint, &inputs.bindings)
            || !self.valid_replay_limits(control)
        {
            return Err(mismatch("control version, identity or limits differ"));
        }
        Ok(())
    }

    fn valid_replay_limits(&self, control: &Control) -> bool {
        control.status.state_bytes <= self.spec.limits().max_state_bytes()
            && control.status.state_rows <= self.spec.limits().max_state_rows()
            && control.record_capacity as u64
                <= self.spec.limits().max_state_bytes() / size_of::<Record>() as u64
    }

    fn decode_replay_log(
        &self,
        snapshot: &OperatorStateSnapshot,
        control: &Control,
    ) -> Result<Log> {
        let mut log = self.new_replay_log()?;
        log.anchor = self.decode_replay_anchor(snapshot, &control.anchor)?;
        allocate_replay_records(&mut log, control.record_capacity)?;
        let mut seen = replay_segments(&log);
        for descriptor in &control.frames {
            self.decode_replay_frame(
                snapshot,
                descriptor,
                control.record_capacity,
                &mut seen,
                &mut log,
            )?;
        }
        log.generation = control.generation;
        Ok(log)
    }

    fn decode_replay_frame(
        &self,
        snapshot: &OperatorStateSnapshot,
        descriptor: &Descriptor,
        capacity: usize,
        seen: &mut BTreeSet<String>,
        log: &mut Log,
    ) -> Result<()> {
        validate_replay_frame(descriptor, capacity, log.records.len(), seen)?;
        let segment = snapshot
            .segments
            .get(&descriptor.id)
            .ok_or_else(|| mismatch("missing callback frame"))?;
        codec::decode_into(
            &segment.bytes_arc(),
            descriptor.count,
            &mut log.records,
            &mut log.credit,
            &|bytes| self.reserve_workspace(bytes),
        )?;
        log.frames.push(Frame {
            segment: segment.clone(),
            descriptor: descriptor.clone(),
        });
        Ok(())
    }

    async fn replay_callbacks(
        &mut self,
        readers: &mut [Box<dyn StreamSource>; 2],
        log: &Log,
        job: &StreamJobContext,
    ) -> Result<()> {
        let (mut next, positions) = replay_positions(log)?;
        for (reader, position) in readers.iter_mut().zip(positions) {
            reader.open(position).await?;
        }
        let name = self.name.clone();
        for record in &log.records {
            job.check_cancelled()?;
            let context = StreamOperatorContext::with_ingress_progress(
                job,
                &name,
                record.input_watermark,
                Self::replay_progress(record),
            )
            .with_output_budget(crate::EdgeBudget {
                max_rows: record.max_rows,
                max_bytes: record.max_bytes,
            });
            self.replay_callback(readers, &mut next, record, &context, job)
                .await?;
        }
        Ok(())
    }

    async fn replay_callback(
        &mut self,
        readers: &mut [Box<dyn StreamSource>; 2],
        next: &mut [u64; 2],
        record: &Record,
        context: &StreamOperatorContext<'_>,
        job: &StreamJobContext,
    ) -> Result<()> {
        match record.callback {
            Callback::Data { side, sequence } => {
                let side = usize::from(side);
                self.replay_data(
                    &mut *readers[side],
                    &mut next[side],
                    (side, sequence),
                    record,
                    context,
                    job,
                )
                .await
            }
            Callback::Progress => {
                self.on_ingress_progress_with_output("", context, &mut Discard)
                    .await
            }
            Callback::End => self.on_end(context, &mut Discard).await,
        }
    }

    async fn replay_data(
        &mut self,
        reader: &mut dyn StreamSource,
        next: &mut u64,
        callback: (usize, u64),
        record: &Record,
        context: &StreamOperatorContext<'_>,
        job: &StreamJobContext,
    ) -> Result<()> {
        let (side, sequence) = callback;
        if *next != sequence {
            return Err(mismatch("source batch sequence differs"));
        }
        let (batch, cursor) = next_data(reader, job).await?;
        let batch = self.replay_batch(&batch, cursor, record, side, sequence)?;
        self.process_data(SIDES[side], batch, context, &mut Discard)
            .await?;
        *next = sequence
            .checked_add(1)
            .ok_or_else(|| mismatch("source sequence exhausted"))?;
        Ok(())
    }

    fn replay_batch(
        &self,
        batch: &Batch,
        cursor: crate::Cursor,
        record: &Record,
        side: usize,
        sequence: u64,
    ) -> Result<Batch> {
        let binding = &self
            .replay_inputs
            .as_ref()
            .expect("inputs were validated")
            .bindings[side];
        let saved = record
            .cursor
            .as_ref()
            .ok_or_else(|| mismatch("data callback has no cursor"))?;
        checked_cursor(saved, cursor, binding)?;
        let metadata =
            BatchMetadata::new(binding, sequence, batch.metadata().attributes().clone())?;
        Ok(batch
            .with_metadata(metadata)
            .with_source_cursor(Some(saved.clone())))
    }
}

fn anchor_metrics(anchor: &Anchor) -> Result<crate::StreamAsofJoinStatus> {
    serde_json::from_value(
        anchor
            .snapshot
            .inline_metadata
            .get("metrics")
            .ok_or_else(|| mismatch("anchor has no counters"))?
            .clone(),
    )
    .map_err(|error| mismatch(&error.to_string()))
}

fn valid_replay_metadata(snapshot: &OperatorStateSnapshot) -> bool {
    let metadata = &snapshot.inline_metadata;
    metadata.len() == 3
        && metadata.get("kind").and_then(serde_json::Value::as_str) == Some("stream_asof_join")
        && metadata
            .get("state_version")
            .and_then(serde_json::Value::as_u64)
            == Some(3)
        && metadata
            .get("source_replay")
            .and_then(serde_json::Value::as_u64)
            == Some(3)
}

fn valid_replay_identity(
    snapshot: &OperatorStateSnapshot,
    control: &Control,
    fingerprint: &str,
    bindings: &[String; 2],
) -> bool {
    control.version == 3
        && control.fingerprint == fingerprint
        && &control.bindings == bindings
        && control.frames.len() <= MAX_FRAMES
        && snapshot.segments.len()
            == control.frames.len() + 1 + anchor_segment_count(&control.anchor)
}

fn anchor_segment_count(anchor: &AnchorControl) -> usize {
    match anchor {
        AnchorControl::FromStart => 0,
        AnchorControl::Native { segments, .. } => segments.len() + 1,
    }
}

fn allocate_replay_records(log: &mut Log, capacity: usize) -> Result<()> {
    let bytes = capacity
        .checked_mul(size_of::<Record>())
        .ok_or_else(|| mismatch("record capacity overflowed"))?;
    log.credit
        .try_grow(bytes)
        .map_err(|_| mismatch("record workspace exceeds current limits"))?;
    log.records
        .try_reserve_exact(capacity)
        .map_err(|_| mismatch("record allocation failed"))
}

fn replay_segments(log: &Log) -> BTreeSet<String> {
    let mut seen = BTreeSet::from([CONTROL_ID.to_string()]);
    if let Some(anchor) = &log.anchor {
        seen.insert(anchor.start_id.clone());
        seen.extend(anchor.snapshot.segments.keys().cloned());
    }
    seen
}

fn validate_replay_frame(
    descriptor: &Descriptor,
    capacity: usize,
    length: usize,
    seen: &mut BTreeSet<String>,
) -> Result<()> {
    if descriptor.first != length as u64
        || descriptor.count == 0
        || !seen.insert(descriptor.id.clone())
        || descriptor.count > (capacity - length) as u64
    {
        return Err(mismatch("frame ranges or identities differ"));
    }
    Ok(())
}

type ReplayPositions = ([u64; 2], [Option<crate::Cursor>; 2]);

fn replay_positions(log: &Log) -> Result<ReplayPositions> {
    let mut next = [0_u64; 2];
    let mut positions = [None, None];
    if let Some(anchor) = &log.anchor {
        for record in &anchor.records[1..] {
            let Callback::Data { side, sequence } = record.callback else {
                return Err(mismatch("anchor position is not data"));
            };
            let side = usize::from(side);
            next[side] = sequence
                .checked_add(1)
                .ok_or_else(|| mismatch("source sequence exhausted"))?;
            positions[side] = record.cursor.as_deref().cloned();
        }
    }
    Ok((next, positions))
}

pub(super) fn decode_control(bytes: &[u8]) -> Result<Control> {
    let value = crate::json::parse_json_value(bytes, "ASOF replay control")
        .map_err(|error| mismatch(&error.to_string()))?;
    serde_json::from_value(value).map_err(|error| mismatch(&error.to_string()))
}

pub(super) fn checked_cursor(
    saved: &crate::Cursor,
    cursor: crate::Cursor,
    binding: &str,
) -> Result<()> {
    let cursor = cursor
        .bind_to(binding)
        .map_err(|_| mismatch("reader cursor owner differs"))?;
    if &cursor != saved {
        return Err(mismatch("reader source cursor differs"));
    }
    Ok(())
}

async fn next_data(
    reader: &mut dyn StreamSource,
    job: &StreamJobContext,
) -> Result<(Batch, crate::Cursor)> {
    loop {
        let event = tokio::select! {
            result = reader.next() => result?,
            () = job.cancellation().cancelled() => return Err(crate::CalcFlowError::Cancelled { run_id: "asof-source-replay".into() }),
        };
        match event {
            Some(SourceEvent::Data { batch, cursor }) => return Ok((batch, cursor)),
            Some(SourceEvent::Watermark(_) | SourceEvent::Idle) => job.check_cancelled()?,
            None => return Err(mismatch("history ended before its recorded callback")),
        }
    }
}
