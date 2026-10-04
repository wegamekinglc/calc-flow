use super::{CONTROL_ID, Callback, Control, Frame, Log, MAX_FRAMES, SIDES, codec, mismatch};
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
        let control_segment = snapshot
            .segments
            .get(CONTROL_ID)
            .ok_or_else(|| mismatch("missing control segment"))?;
        let bytes = control_segment.bytes_arc();
        if bytes.len() > 65536 {
            return Err(mismatch("control exceeds its size limit"));
        }
        let workspace = self.reserve_workspace((bytes.len() * 8 + 4096) as u64)?;
        let control: Control =
            serde_json::from_slice(&bytes).map_err(|error| mismatch(&error.to_string()))?;
        self.validate_replay_control(&snapshot, &control)?;
        let mut log = self.decode_replay_log(&snapshot, &control)?;
        drop(workspace);
        let inputs = self
            .replay_inputs
            .as_ref()
            .ok_or_else(|| mismatch("sources no longer support replay"))?;
        let mut readers = [
            inputs.factories[0].create(inputs.histories[0].fork())?,
            inputs.factories[1].create(inputs.histories[1].fork())?,
        ];
        let result = AssertUnwindSafe(self.replay_callbacks(&mut readers, &log, job))
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err(mismatch("reader or callback panicked")));
        let mut close_error = None;
        for reader in &mut readers {
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
        self.observe(&progress);
        log.cut = log.records.len();
        self.replay = Some(Box::new(log));
        self.status.state_bytes = self.current_inventory(None)?.bytes;
        self.status.output_watermark_micros = control.status.output_watermark_micros;
        if self.status != control.status
            || self.terminal != control.terminal
            || self.next_output_sequence != control.next_output_sequence
        {
            return Err(mismatch(
                "rebuilt counters, capacity or output coordinate differs",
            ));
        }
        self.check_inventory_limits(&self.current_inventory(None)?)?;
        job.check_cancelled()?;
        Ok(self)
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
        if snapshot.inline_metadata.len() != 3
            || snapshot
                .inline_metadata
                .get("kind")
                .and_then(serde_json::Value::as_str)
                != Some("stream_asof_join")
            || snapshot
                .inline_metadata
                .get("state_version")
                .and_then(serde_json::Value::as_u64)
                != Some(3)
            || snapshot
                .inline_metadata
                .get("source_replay")
                .and_then(serde_json::Value::as_u64)
                != Some(2)
            || control.version != 2
            || control.fingerprint != self.fingerprint
            || control.bindings != inputs.bindings
            || control.frames.len() > MAX_FRAMES
            || snapshot.segments.len() != control.frames.len() + 1
            || control.status.state_bytes > self.spec.limits().max_state_bytes()
            || control.status.state_rows > self.spec.limits().max_state_rows()
            || control.record_capacity as u64
                > self.spec.limits().max_state_bytes() / size_of::<super::Record>() as u64
        {
            return Err(mismatch("control version, identity or limits differ"));
        }
        Ok(())
    }

    fn decode_replay_log(
        &self,
        snapshot: &OperatorStateSnapshot,
        control: &Control,
    ) -> Result<Log> {
        let mut log = self.new_replay_log()?;
        let bytes = control
            .record_capacity
            .checked_mul(size_of::<super::Record>())
            .ok_or_else(|| mismatch("record capacity overflowed"))?;
        log.credit
            .try_grow(bytes)
            .map_err(|_| mismatch("record workspace exceeds current limits"))?;
        log.records
            .try_reserve_exact(control.record_capacity)
            .map_err(|_| mismatch("record allocation failed"))?;
        let mut seen = BTreeSet::new();
        for descriptor in &control.frames {
            if descriptor.first != log.records.len() as u64
                || descriptor.count == 0
                || !seen.insert(&descriptor.id)
                || descriptor.count > (control.record_capacity - log.records.len()) as u64
            {
                return Err(mismatch("frame ranges or identities differ"));
            }
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
        }
        log.generation = control.generation;
        Ok(log)
    }

    async fn replay_callbacks(
        &mut self,
        readers: &mut [Box<dyn StreamSource>; 2],
        log: &Log,
        job: &StreamJobContext,
    ) -> Result<()> {
        for reader in readers.iter_mut() {
            reader.open(None).await?;
        }
        let mut next = [0_u64; 2];
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
            match record.callback {
                Callback::Data { side, sequence } => {
                    let side = usize::from(side);
                    if next[side] != sequence {
                        return Err(mismatch("source batch sequence differs"));
                    }
                    let (batch, cursor) = next_data(&mut *readers[side], job).await?;
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
                    let metadata = BatchMetadata::new(
                        binding,
                        sequence,
                        batch.metadata().attributes().clone(),
                    )?;
                    self.process_data(
                        SIDES[side],
                        batch
                            .with_metadata(metadata)
                            .with_source_cursor(Some(saved.clone())),
                        &context,
                        &mut Discard,
                    )
                    .await?;
                    next[side] = sequence
                        .checked_add(1)
                        .ok_or_else(|| mismatch("source sequence exhausted"))?;
                }
                Callback::Progress => {
                    self.on_ingress_progress_with_output("", &context, &mut Discard)
                        .await?;
                }
                Callback::End => self.on_end(&context, &mut Discard).await?,
            }
        }
        Ok(())
    }
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
