use super::{
    Bucket, LiveRun, LocatorEntry, RowLocator, Run, RunCharge, RunId, RunState, RunStateSnapshot,
    Side,
};
use crate::operator::keyed::{GroupId, KeyStore};
use crate::state::v4::native::{NativePage, NativePageChange, NativePageSet};
use crate::{CalcFlowError, EventTime, Result, StateSegment};
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array, UInt8Array, UInt32Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    ipc::reader::FileReader,
    record_batch::RecordBatch,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
    sync::Arc,
};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Control {
    version: u32,
    next_run: u64,
    next_output: u64,
    match_limit_failures: u64,
    sides: [SideControl; 2],
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SideControl {
    next_row_id: u64,
    rows: u64,
    bytes: u64,
    expiration_head: u64,
}

#[derive(Default)]
pub(in crate::operator::join) struct NativePageCache {
    runs: [BTreeMap<RunId, CachedRunPages>; 2],
    groups: [BTreeMap<GroupId, NativePage>; 2],
    expiration: [Option<NativePage>; 2],
    control: Option<(Control, NativePage)>,
    pub(in crate::operator::join) lineage: Option<Arc<NativePageSet>>,
    initialized: bool,
    #[cfg(test)]
    reused_identities: usize,
    #[cfg(test)]
    reused_indexes: usize,
}

struct CachedRunPages {
    payload: NativePage,
    identity: NativePage,
    head: usize,
    released_bytes: usize,
}

impl NativePageCache {
    #[cfg(test)]
    pub(in crate::operator::join) fn reuse_counts(&self) -> [usize; 2] {
        [self.reused_identities, self.reused_indexes]
    }
}

pub(super) fn capture(
    state: &mut RunState,
    epoch: crate::Epoch,
    cache: &mut NativePageCache,
) -> Result<NativePageSet> {
    let epoch = epoch.as_u64();
    let initialized = cache.initialized;
    let mut changes = Vec::new();
    for side in 0..2 {
        let dirty = if initialized {
            state.dirty.runs[side].clone()
        } else {
            state.sides[side].runs.keys().copied().collect()
        };
        #[cfg(test)]
        if initialized {
            let dirty_existing = dirty
                .iter()
                .filter(|id| cache.runs[side].contains_key(id))
                .count();
            cache.reused_identities += cache.runs[side].len().saturating_sub(dirty_existing);
        }
        for id in dirty {
            capture_run(state, side, id, epoch, cache, &mut changes)?;
        }
        let dirty_groups = if initialized {
            state.dirty.groups[side].clone()
        } else {
            state.sides[side].buckets.keys().copied().collect()
        };
        #[cfg(test)]
        if initialized {
            let dirty_existing = dirty_groups
                .iter()
                .filter(|group| cache.groups[side].contains_key(group))
                .count();
            cache.reused_indexes += cache.groups[side].len().saturating_sub(dirty_existing);
        }
        for group in dirty_groups {
            capture_group(state, side, group, epoch, cache, &mut changes)?;
        }
        if !initialized || state.dirty.expiration[side] {
            let prefix = if side == 0 { "left" } else { "right" };
            let rows = u64::try_from(state.sides[side].expiration.len())
                .map_err(|_| page_error("expiration rows exceed u64"))?;
            let page = NativePage::arrow(
                format!("{prefix}/expiration"),
                &expiration_record(&state.sides[side])?,
                epoch,
                0..rows,
            );
            changes.push(NativePageChange::upsert(
                page.clone(),
                cache.expiration[side].as_ref(),
            )?);
            cache.expiration[side] = Some(page);
        } else {
            #[cfg(test)]
            {
                cache.reused_indexes += 1;
            }
        }
    }
    let control = Control {
        version: 1,
        next_run: state.next_run,
        next_output: state.next_output,
        match_limit_failures: state.match_limit_failures,
        sides: state.sides.each_ref().map(|side| SideControl {
            next_row_id: side.next_row_id,
            rows: u64::try_from(side.rows).expect("state rows fit u64"),
            bytes: u64::try_from(side.bytes).expect("state bytes fit u64"),
            expiration_head: u64::try_from(side.expiration_head).expect("expiration head fits u64"),
        }),
    };
    let control_page = if let Some((previous, page)) = &cache.control
        && previous == &control
    {
        page.clone()
    } else {
        let page = NativePage::json(
            "control",
            serde_json::to_value(&control).map_err(|error| page_error(error.to_string()))?,
            epoch,
            0..0,
        );
        changes.push(NativePageChange::upsert(
            page.clone(),
            cache.control.as_ref().map(|(_, previous)| previous),
        )?);
        page
    };
    cache.control = Some((control, control_page));
    cache.initialized = true;
    NativePageSet::new(cache.lineage.clone(), changes)
}

fn capture_run(
    state: &RunState,
    side: usize,
    id: RunId,
    epoch: u64,
    cache: &mut NativePageCache,
    changes: &mut Vec<NativePageChange>,
) -> Result<()> {
    let Some(live) = state.sides[side].runs.get(&id) else {
        if let Some(cached) = cache.runs[side].remove(&id) {
            changes.push(NativePageChange::remove(&cached.payload)?);
            changes.push(NativePageChange::remove(&cached.identity)?);
        }
        return Ok(());
    };
    let prefix = if side == 0 { "left" } else { "right" };
    let rows =
        u64::try_from(live.run.row_ids.len()).map_err(|_| page_error("run rows exceed u64"))?;
    let next = if let Some(cached) = cache.runs[side].get(&id) {
        let identity = if cached.head == live.head && cached.released_bytes == live.released_bytes {
            cached.identity.clone()
        } else {
            let page = NativePage::arrow(
                format!("{prefix}/run-{}/identity", id.0),
                &run_identity(id, live)?,
                epoch,
                0..rows,
            );
            changes.push(NativePageChange::upsert(
                page.clone(),
                Some(&cached.identity),
            )?);
            page
        };
        CachedRunPages {
            payload: cached.payload.clone(),
            identity,
            head: live.head,
            released_bytes: live.released_bytes,
        }
    } else {
        let payload = NativePage::arrow(
            format!("{prefix}/run-{}/payload", id.0),
            &live.run.payload,
            id.0,
            0..rows,
        );
        let identity = NativePage::arrow(
            format!("{prefix}/run-{}/identity", id.0),
            &run_identity(id, live)?,
            epoch,
            0..rows,
        );
        changes.push(NativePageChange::upsert(payload.clone(), None)?);
        changes.push(NativePageChange::upsert(identity.clone(), None)?);
        CachedRunPages {
            payload,
            identity,
            head: live.head,
            released_bytes: live.released_bytes,
        }
    };
    cache.runs[side].insert(id, next);
    Ok(())
}

fn capture_group(
    state: &RunState,
    side: usize,
    group: GroupId,
    epoch: u64,
    cache: &mut NativePageCache,
    changes: &mut Vec<NativePageChange>,
) -> Result<()> {
    let Some(bucket) = state.sides[side].buckets.get(&group) else {
        if let Some(previous) = cache.groups[side].remove(&group) {
            changes.push(NativePageChange::remove(&previous)?);
        }
        return Ok(());
    };
    let prefix = if side == 0 { "left" } else { "right" };
    let rows =
        u64::try_from(bucket.entries.len()).map_err(|_| page_error("bucket rows exceed u64"))?;
    let page = NativePage::arrow(
        format!("{prefix}/group-{}", group.index()),
        &bucket_record(group, bucket)?,
        epoch,
        0..rows,
    );
    changes.push(NativePageChange::upsert(
        page.clone(),
        cache.groups[side].get(&group),
    )?);
    cache.groups[side].insert(group, page);
    Ok(())
}

fn run_identity(id: RunId, live: &LiveRun) -> Result<RecordBatch> {
    let run = &live.run;
    let rows = run.row_ids.len();
    let groups = run
        .groups
        .iter()
        .map(|group| u32::try_from(group.index()).map_err(|_| page_error("group exceeds u32")))
        .collect::<Result<Vec<_>>>()?;
    let order = run
        .order
        .iter()
        .map(|row| u32::try_from(*row).map_err(|_| page_error("run offset exceeds u32")))
        .collect::<Result<Vec<_>>>()?;
    let fields = vec![
        Field::new("run", DataType::UInt64, false),
        Field::new("group", DataType::UInt32, false),
        Field::new("time", DataType::Int64, false),
        Field::new("row_id", DataType::UInt64, false),
        Field::new("ordinal", DataType::UInt32, false),
        Field::new("head", DataType::UInt64, false),
        Field::new("released_bytes", DataType::UInt64, false),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![id.0; rows])),
        Arc::new(UInt32Array::from(groups)),
        Arc::new(Int64Array::from(run.times.to_vec())),
        Arc::new(UInt64Array::from(run.row_ids.to_vec())),
        Arc::new(UInt32Array::from(order)),
        Arc::new(UInt64Array::from(vec![
            u64::try_from(live.head)
                .expect("run head fits u64");
            rows
        ])),
        Arc::new(UInt64Array::from(vec![
            u64::try_from(live.released_bytes)
                .expect("released bytes fit u64");
            rows
        ])),
    ];
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|error| page_error(error.to_string()))
}

#[derive(Default)]
struct LocatorColumns {
    kind: Vec<u8>,
    groups: Vec<u32>,
    times: Vec<i64>,
    row_ids: Vec<u64>,
    runs: Vec<u64>,
    rows: Vec<u32>,
    heads: Vec<u64>,
    capacities: Vec<u64>,
}

impl LocatorColumns {
    fn push(
        &mut self,
        value: u8,
        group: GroupId,
        entry: &LocatorEntry,
        head: usize,
        capacity: usize,
    ) {
        self.kind.push(value);
        self.groups
            .push(u32::try_from(group.index()).expect("GroupId is backed by u32"));
        self.times.push(entry.time.as_micros());
        self.row_ids.push(entry.row_id);
        self.runs.push(entry.locator.run.0);
        self.rows.push(entry.locator.row);
        self.heads
            .push(u64::try_from(head).expect("index head fits u64"));
        self.capacities
            .push(u64::try_from(capacity).expect("index capacity fits u64"));
    }
}

fn bucket_record(group: GroupId, bucket: &Bucket) -> Result<RecordBatch> {
    let mut columns = LocatorColumns::default();
    for entry in &bucket.entries {
        columns.push(0, group, entry, bucket.head, bucket.entries.capacity());
    }
    locator_record(columns)
}

fn expiration_record(side: &Side) -> Result<RecordBatch> {
    let mut columns = LocatorColumns::default();
    for entry in &side.expiration {
        columns.push(
            1,
            entry.group,
            entry,
            side.expiration_head,
            side.expiration.capacity(),
        );
    }
    locator_record(columns)
}

fn locator_record(columns: LocatorColumns) -> Result<RecordBatch> {
    let fields = vec![
        Field::new("kind", DataType::UInt8, false),
        Field::new("group", DataType::UInt32, false),
        Field::new("time", DataType::Int64, false),
        Field::new("row_id", DataType::UInt64, false),
        Field::new("run", DataType::UInt64, false),
        Field::new("row", DataType::UInt32, false),
        Field::new("head", DataType::UInt64, false),
        Field::new("capacity", DataType::UInt64, false),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(columns.kind)),
        Arc::new(UInt32Array::from(columns.groups)),
        Arc::new(Int64Array::from(columns.times)),
        Arc::new(UInt64Array::from(columns.row_ids)),
        Arc::new(UInt64Array::from(columns.runs)),
        Arc::new(UInt32Array::from(columns.rows)),
        Arc::new(UInt64Array::from(columns.heads)),
        Arc::new(UInt64Array::from(columns.capacities)),
    ];
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|error| page_error(error.to_string()))
}

pub(super) fn restore(
    name: &str,
    limits: super::JoinStateLimits,
    bounds: super::JoinTimeBounds,
    segments: &BTreeMap<String, StateSegment>,
    keys: &mut KeyStore,
) -> Result<RunState> {
    let control: Control = decode_json(
        segments
            .get("control")
            .ok_or_else(|| page_error("missing control"))?,
    )?;
    if control.version != 1 {
        return Err(page_error("unsupported native Join control version"));
    }
    let key_record = decode_record(
        segments
            .get("keys")
            .ok_or_else(|| page_error("missing keys"))?,
    )?;
    keys.restore(&key_record)
        .map_err(|error| page_error(error.to_string()))?;
    let mut expected = BTreeSet::from([
        "control".to_owned(),
        "keys".to_owned(),
        "left/expiration".to_owned(),
        "right/expiration".to_owned(),
    ]);
    let mut seen_runs = BTreeSet::new();
    let mut sides = [Side::empty(), Side::empty()];
    for (side, state) in sides.iter_mut().enumerate() {
        let prefix = if side == 0 { "left" } else { "right" };
        restore_index_pages(state, side, segments, &mut expected)?;
        for (name, payload) in segments.iter().filter(|(name, _)| {
            name.starts_with(&format!("{prefix}/run-")) && name.ends_with("/payload")
        }) {
            let identity_name = name
                .strip_suffix("payload")
                .expect("payload suffix")
                .to_owned()
                + "identity";
            let identity = decode_record(
                segments
                    .get(&identity_name)
                    .ok_or_else(|| page_error("missing run identity"))?,
            )?;
            let (id, live) = restore_run(payload, &identity)?;
            if name != &format!("{prefix}/run-{}/payload", id.0)
                || !seen_runs.insert(id)
                || state.runs.insert(id, live).is_some()
            {
                return Err(page_error("duplicated or misnamed run identifier"));
            }
            expected.insert(name.clone());
            expected.insert(identity_name);
        }
        let control = &control.sides[side];
        if state.expiration_head
            != usize::try_from(control.expiration_head)
                .map_err(|_| page_error("expiration head exceeds usize"))?
        {
            return Err(page_error("expiration head differs from its index"));
        }
        state.next_row_id = control.next_row_id;
        state.rows =
            usize::try_from(control.rows).map_err(|_| page_error("side rows exceed usize"))?;
        state.bytes =
            usize::try_from(control.bytes).map_err(|_| page_error("side bytes exceed usize"))?;
        state.expiration_head = usize::try_from(control.expiration_head)
            .map_err(|_| page_error("expiration head exceeds usize"))?;
    }
    if segments.keys().any(|name| !expected.contains(name))
        || sides.iter().any(|side| {
            side.runs.values().any(|live| {
                live.run
                    .groups
                    .iter()
                    .any(|group| group.index() >= keys.len())
            }) || side.buckets.keys().any(|group| group.index() >= keys.len())
        })
    {
        return Err(page_error("snapshot contains undescribed pages or groups"));
    }
    RunState::restore(RunStateSnapshot {
        name: name.into(),
        limits,
        bounds,
        sides,
        next_run: control.next_run,
        next_output: control.next_output,
        match_limit_failures: control.match_limit_failures,
    })
}

fn restore_index_pages(
    state: &mut Side,
    side: usize,
    segments: &BTreeMap<String, StateSegment>,
    expected: &mut BTreeSet<String>,
) -> Result<()> {
    let prefix = if side == 0 { "left" } else { "right" };
    let expiration_name = format!("{prefix}/expiration");
    let expiration = decode_record(
        segments
            .get(&expiration_name)
            .ok_or_else(|| page_error("missing expiration index"))?,
    )?;
    if uint8s(&expiration, 0)?.iter().any(|kind| *kind != 1) {
        return Err(page_error("expiration page contains a bucket entry"));
    }
    let mut records = vec![expiration];
    let group_prefix = format!("{prefix}/group-");
    let mut seen_groups = BTreeSet::new();
    for (name, segment) in segments
        .iter()
        .filter(|(name, _)| name.starts_with(&group_prefix))
    {
        let group = name[group_prefix.len()..]
            .parse::<usize>()
            .ok()
            .and_then(|group| GroupId::new(group).ok())
            .ok_or_else(|| page_error("invalid bucket page group"))?;
        let record = decode_record(segment)?;
        if uint8s(&record, 0)?.iter().any(|kind| *kind != 0) {
            return Err(page_error("bucket page contains an expiration entry"));
        }
        let groups = uint32s(&record, 1)?;
        let record_group_id = groups
            .first()
            .copied()
            .and_then(|group| usize::try_from(group).ok())
            .and_then(|group| GroupId::new(group).ok())
            .ok_or_else(|| page_error("empty bucket page"))?;
        let record_group =
            u32::try_from(record_group_id.index()).expect("GroupId is backed by u32");
        if group != record_group_id
            || groups.iter().any(|value| *value != record_group)
            || !seen_groups.insert(group)
        {
            return Err(page_error("duplicated or mismatched bucket page group"));
        }
        expected.insert(name.clone());
        records.push(record);
    }
    restore_index(state, &records)
}

fn restore_run(payload: &StateSegment, identity: &RecordBatch) -> Result<(RunId, LiveRun)> {
    let payload = decode_record(payload)?;
    let rows = payload.num_rows();
    if rows == 0
        || identity.num_rows() != rows
        || identity.num_columns() != 7
        || identity
            .columns()
            .iter()
            .any(|array| array.null_count() != 0)
    {
        return Err(page_error("run identity and payload shapes differ"));
    }
    let ids = uint64s(identity, 0)?;
    let groups = uint32s(identity, 1)?;
    let times = int64s(identity, 2)?;
    let row_ids = uint64s(identity, 3)?;
    let order = uint32s(identity, 4)?;
    let heads = uint64s(identity, 5)?;
    let released = uint64s(identity, 6)?;
    let id = RunId(ids[0]);
    let head = usize::try_from(heads[0]).map_err(|_| page_error("run head exceeds usize"))?;
    let released_bytes =
        usize::try_from(released[0]).map_err(|_| page_error("released bytes exceed usize"))?;
    if ids.iter().any(|value| *value != id.0)
        || heads.iter().any(|value| *value != heads[0])
        || released.iter().any(|value| *value != released[0])
    {
        return Err(page_error("run identity descriptors are not constant"));
    }
    let groups = groups
        .into_iter()
        .map(|group| {
            GroupId::new(usize::try_from(group).expect("u32 group fits usize"))
                .map_err(|error| page_error(error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let order = order
        .into_iter()
        .map(|row| usize::try_from(row).map_err(|_| page_error("run ordinal exceeds usize")))
        .collect::<Result<Vec<_>>>()?;
    let mut permutation = BTreeSet::new();
    if order
        .iter()
        .any(|row| *row >= rows || !permutation.insert(*row))
    {
        return Err(page_error("run order is not a permutation"));
    }
    let charge = RunCharge::read(&payload)?;
    let bytes = charge.total(rows)?;
    let run = Run {
        id,
        payload,
        groups: groups.into(),
        times: times.into(),
        row_ids: row_ids.into(),
        order: order.into(),
        charge,
        bytes,
    };
    Ok((
        id,
        LiveRun {
            run: Arc::new(run),
            head,
            released_bytes,
        },
    ))
}

fn restore_index(side: &mut Side, records: &[RecordBatch]) -> Result<()> {
    let mut buckets = BTreeMap::<GroupId, (usize, usize, Vec<LocatorEntry>)>::new();
    let mut expiration = (0usize, 0usize, Vec::new());
    for record in records {
        if record.num_columns() != 8 || record.columns().iter().any(|array| array.null_count() != 0)
        {
            return Err(page_error("index schema differs"));
        }
        let kinds = uint8s(record, 0)?;
        let groups = uint32s(record, 1)?;
        let times = int64s(record, 2)?;
        let row_ids = uint64s(record, 3)?;
        let runs = uint64s(record, 4)?;
        let rows = uint32s(record, 5)?;
        let heads = uint64s(record, 6)?;
        let capacities = uint64s(record, 7)?;
        for row in 0..record.num_rows() {
            let group = GroupId::new(usize::try_from(groups[row]).expect("u32 group fits usize"))
                .map_err(|error| page_error(error.to_string()))?;
            let entry = LocatorEntry {
                time: EventTime::from_micros(times[row]),
                row_id: row_ids[row],
                locator: RowLocator {
                    run: RunId(runs[row]),
                    row: rows[row],
                },
                group,
            };
            let head =
                usize::try_from(heads[row]).map_err(|_| page_error("index head exceeds usize"))?;
            let capacity = usize::try_from(capacities[row])
                .map_err(|_| page_error("index capacity exceeds usize"))?;
            match kinds[row] {
                0 => {
                    let bucket = buckets
                        .entry(group)
                        .or_insert_with(|| (head, capacity, Vec::with_capacity(capacity)));
                    if bucket.0 != head || bucket.1 != capacity {
                        return Err(page_error("bucket descriptors differ"));
                    }
                    bucket.2.push(entry);
                }
                1 => {
                    if expiration.2.is_empty() {
                        expiration = (head, capacity, Vec::with_capacity(capacity));
                    }
                    if expiration.0 != head || expiration.1 != capacity {
                        return Err(page_error("expiration descriptors differ"));
                    }
                    expiration.2.push(entry);
                }
                _ => return Err(page_error("unknown index entry kind")),
            }
        }
    }
    side.buckets = buckets
        .into_iter()
        .map(|(group, (head, capacity, entries))| {
            if head > entries.len() || capacity < entries.len() {
                return Err(page_error("bucket prefix exceeds its capacity"));
            }
            Ok((group, Bucket { entries, head }))
        })
        .collect::<Result<_>>()?;
    let (head, capacity, entries) = expiration;
    if head > entries.len() || capacity < entries.len() {
        return Err(page_error("expiration prefix exceeds its capacity"));
    }
    side.expiration = entries;
    side.expiration_head = head;
    Ok(())
}

fn decode_record(segment: &StateSegment) -> Result<RecordBatch> {
    let mut reader = FileReader::try_new(Cursor::new(segment.bytes()), None)
        .map_err(|error| page_error(error.to_string()))?;
    let schema = reader.schema();
    let mut records = reader
        .by_ref()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| page_error(error.to_string()))?;
    if records.len() > 1 {
        return Err(page_error("native page contains multiple batches"));
    }
    Ok(records
        .pop()
        .unwrap_or_else(|| RecordBatch::new_empty(schema)))
}

fn decode_json<T: for<'de> Deserialize<'de>>(segment: &StateSegment) -> Result<T> {
    serde_json::from_slice(segment.bytes()).map_err(|error| page_error(error.to_string()))
}

fn uint64s(record: &RecordBatch, column: usize) -> Result<Vec<u64>> {
    record
        .column(column)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .map(|array| array.values().to_vec())
        .ok_or_else(|| page_error("expected UInt64 column"))
}

fn uint32s(record: &RecordBatch, column: usize) -> Result<Vec<u32>> {
    record
        .column(column)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .map(|array| array.values().to_vec())
        .ok_or_else(|| page_error("expected UInt32 column"))
}

fn int64s(record: &RecordBatch, column: usize) -> Result<Vec<i64>> {
    record
        .column(column)
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|array| array.values().to_vec())
        .ok_or_else(|| page_error("expected Int64 column"))
}

fn uint8s(record: &RecordBatch, column: usize) -> Result<Vec<u8>> {
    record
        .column(column)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .map(|array| array.values().to_vec())
        .ok_or_else(|| page_error("expected UInt8 column"))
}

fn page_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("Join native pages: {}", message.into()),
    }
}
