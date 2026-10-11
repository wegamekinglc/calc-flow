use datafusion::execution::memory_pool::MemoryReservation;

use super::super::super::frame;
use super::super::buffer::{PaidBuffer, ensure, error};
use super::{Base, Encoder, JoinSide, PayloadEntry, accounting, own_segment, segment_charge};
use crate::operator::join::StreamJoinOperator;
use crate::{Result, StateSegment, StreamOperatorContext};

mod funding;
mod input;
mod merge;
mod pending;

pub(in crate::operator::join::checkpoint_v2::writer) use pending::PendingInput;

struct IndexInput {
    side: JoinSide,
    epoch: Option<u64>,
    pending: bool,
    segment: StateSegment,
}

impl IndexInput {
    fn index(&self) -> Result<frame::Index<'_>> {
        input::index(&self.segment, self.side, self.epoch)
    }
}

struct PayloadInput {
    side: JoinSide,
    digest: [u8; 32],
    rows: usize,
    segment: StateSegment,
}

struct SideExpected {
    rows: usize,
    charge: u64,
    next_row_id: u64,
}

pub(in crate::operator::join::checkpoint_v2::writer) struct HistoryInput {
    indexes: Vec<IndexInput>,
    payloads: Vec<PayloadInput>,
    expected: [SideExpected; 2],
}

pub(in crate::operator::join::checkpoint_v2::writer) async fn eligible(
    operator: &StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
) -> Result<bool> {
    input::eligible(operator, context).await
}

pub(in crate::operator::join::checkpoint_v2::writer) async fn copy(
    operator: &StreamJoinOperator,
    workspace: &MemoryReservation,
    context: &StreamOperatorContext<'_>,
) -> Result<HistoryInput> {
    input::copy(operator, workspace, context).await
}

pub(in crate::operator::join::checkpoint_v2::writer) fn encode(
    history: &HistoryInput,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    check()?;
    let mut encoder = Encoder::new(workspace)?;
    prepare_output(history, &mut encoder, check)?;
    for side in [JoinSide::Left, JoinSide::Right] {
        encode_side(history, side, &mut encoder, workspace, check)?;
    }
    check()?;
    Ok(encoder.finish())
}

fn prepare_output(
    history: &HistoryInput,
    encoder: &mut Encoder,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    encoder.retained = accounting::add(encoder.retained, funding::output(history, check)?)?;
    ensure(&encoder.funding, encoder.retained)?;
    encoder.output.payloads = funding::vector(history.payloads.len())?;
    copy_payloads(history, encoder, check)?;
    Ok(())
}

fn append_index_header(bytes: &mut PaidBuffer<'_>, side: JoinSide, rows: usize) -> Result<()> {
    bytes.append(&super::header(*super::BASE_MAGIC, side))?;
    bytes.append(&super::integer(rows)?)?;
    bytes.append(&0_u64.to_le_bytes())?;
    Ok(())
}

fn index_buffer(
    encoder: &Encoder,
    side: JoinSide,
    counts: merge::Counts,
) -> Result<(PaidBuffer<'_>, usize)> {
    let capacity = counts.index_bytes()?;
    let charge = segment_charge(capacity)?;
    ensure(&encoder.funding, accounting::add(encoder.retained, charge)?)?;
    let mut bytes = PaidBuffer::exact(&encoder.funding, encoder.retained, capacity)?;
    append_index_header(&mut bytes, side, counts.rows)?;
    Ok((bytes, charge))
}

fn rebase_index(
    history: &HistoryInput,
    side: JoinSide,
    encoder: &Encoder,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<(Vec<u8>, usize)> {
    let counts = merge::scan(history, side, workspace, check, |_| Ok(()))?;
    let (mut bytes, charge) = index_buffer(encoder, side, counts)?;
    let repeated = merge::scan(history, side, workspace, check, |row| {
        append_upsert(&mut bytes, row)
    })?;
    if counts != repeated {
        return Err(error("V2 rebase record passes differ"));
    }
    let bytes = bytes.into_bytes()?;
    check()?;
    Ok((bytes, charge))
}

fn encode_side(
    history: &HistoryInput,
    side: JoinSide,
    encoder: &mut Encoder,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let (bytes, charge) = rebase_index(history, side, encoder, workspace, check)?;
    let segment = own_segment(bytes, &encoder.funding, charge);
    let name = funding::text(&[side.as_str(), "-base"])?;
    encoder.output.segments.insert(name, segment);
    Ok(())
}

pub(in crate::operator::join::checkpoint_v2::writer) fn encode_pending(
    history: HistoryInput,
    pending: PendingInput,
    schemas: &[super::SchemaRef; 2],
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    pending::encode(history, pending, schemas, workspace, check)
}

fn copy_payloads(
    history: &HistoryInput,
    encoder: &mut Encoder,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for payload in &history.payloads {
        check()?;
        let sha256 = payload.segment.sha256();
        let name = funding::text(&[payload.side.as_str(), "-payload-", sha256])?;
        let entry = PayloadEntry {
            side: payload.side.as_str(),
            sha256: sha256.into(),
            rows: u64::try_from(payload.rows)
                .map_err(|_| error("V2 rebase payload rows overflow"))?,
            bytes: u64::try_from(payload.segment.bytes().len())
                .map_err(|_| error("V2 rebase payload bytes overflow"))?,
        };
        encoder
            .output
            .segments
            .insert(name, payload.segment.clone());
        encoder.output.payloads.push(entry);
    }
    Ok(())
}

fn append_upsert(bytes: &mut PaidBuffer<'_>, row: &frame::Upsert<'_>) -> Result<()> {
    let mut fixed = [0_u8; 72];
    fixed[..8].copy_from_slice(&row.row_id.to_le_bytes());
    fixed[8..16].copy_from_slice(&row.time.to_le_bytes());
    fixed[16..24].copy_from_slice(&row.charge.to_le_bytes());
    fixed[24..56].copy_from_slice(&row.digest);
    fixed[56..64].copy_from_slice(&row.payload_row.to_le_bytes());
    fixed[64..72].copy_from_slice(&super::integer(row.key.len())?);
    bytes.append(&fixed)?;
    bytes.append(row.key)
}
