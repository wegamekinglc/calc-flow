//! Rolling's owned state-table seam.
//!
//! Dense entities index keyed and ring columns. Hot-path mutations append undo
//! records; durable snapshots remain whole-table owned values in P2.

use std::{collections::HashMap, mem::size_of, ops::Range, sync::Arc};

use datafusion::arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, StructArray,
        UInt64Array,
    },
    datatypes::{DataType, Field},
};

use crate::{CalcFlowError, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EntityId(u32);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EntityKey {
    bytes: Vec<u8>,
}

impl EntityKey {
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ValueKind {
    UInt64,
    Int64,
    Float64,
    Boolean,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RingKind {
    Float64,
    Int64,
    UInt64,
    PairFloat64,
}

impl RingKind {
    fn width(self) -> usize {
        match self {
            Self::Float64 | Self::Int64 | Self::UInt64 => 8,
            Self::PairFloat64 => 16,
        }
    }

    fn validity_bits(self) -> usize {
        usize::from(self != Self::PairFloat64) + usize::from(self == Self::PairFloat64) * 2
    }

    fn empty(self) -> RingValue {
        match self {
            Self::Float64 => RingValue::Float64(None),
            Self::Int64 => RingValue::Int64(None),
            Self::UInt64 => RingValue::UInt64(None),
            Self::PairFloat64 => RingValue::PairFloat64(None),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Retention {
    Rows(usize),
    Duration,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RingValue {
    Float64(Option<f64>),
    Int64(Option<i64>),
    UInt64(Option<u64>),
    PairFloat64(Option<(f64, f64)>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum KeyedValue {
    UInt64(u64),
    Int64(i64),
    Float64(f64),
    Boolean(bool),
}

impl KeyedValue {
    fn default(kind: ValueKind) -> Self {
        match kind {
            ValueKind::UInt64 => Self::UInt64(0),
            ValueKind::Int64 => Self::Int64(0),
            ValueKind::Float64 => Self::Float64(0.0),
            ValueKind::Boolean => Self::Boolean(false),
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::UInt64(_) | Self::Int64(_) | Self::Float64(_) => 8,
            Self::Boolean(_) => 1,
        }
    }
}

#[derive(Debug)]
struct KeyedColumn {
    name: String,
    kind: ValueKind,
    values: Vec<KeyedValue>,
    stamps: Vec<u64>,
}

impl KeyedColumn {
    fn new(name: String, kind: ValueKind) -> Self {
        Self {
            name,
            kind,
            values: Vec::new(),
            stamps: Vec::new(),
        }
    }

    fn append(&mut self) {
        self.values.push(KeyedValue::default(self.kind));
        self.stamps.push(0);
    }

    fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
        self.stamps.truncate(len);
    }

    fn logical_bytes(&self) -> usize {
        self.values.len() * KeyedValue::default(self.kind).width()
    }
}

#[derive(Debug)]
struct Ring {
    head: usize,
    len: usize,
    slots: Vec<RingValue>,
    slot_stamps: Vec<u64>,
}

impl Ring {
    fn new(kind: RingKind, capacity: usize) -> Self {
        Self {
            head: 0,
            len: 0,
            slots: vec![kind.empty(); capacity],
            slot_stamps: vec![0; capacity],
        }
    }

    fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn position(&self, offset: usize) -> usize {
        (self.head + offset) % self.capacity()
    }
}

#[derive(Debug)]
struct RingColumn {
    name: String,
    kind: RingKind,
    retention: Retention,
    rings: Vec<Ring>,
    stamps: Vec<u64>,
    logical_slots: usize,
    logical_samples: usize,
}

impl RingColumn {
    fn new(name: String, kind: RingKind, retention: Retention) -> Self {
        Self {
            name,
            kind,
            retention,
            rings: Vec::new(),
            stamps: Vec::new(),
            logical_slots: 0,
            logical_samples: 0,
        }
    }

    fn append(&mut self) {
        let capacity = match self.retention {
            Retention::Rows(capacity) => capacity,
            Retention::Duration => 0,
        };
        self.logical_slots += capacity;
        self.rings.push(Ring::new(self.kind, capacity));
        self.stamps.push(0);
    }

    fn truncate(&mut self, len: usize) {
        for ring in &self.rings[len..] {
            self.logical_slots -= ring.capacity();
            self.logical_samples -= ring.len;
        }
        self.rings.truncate(len);
        self.stamps.truncate(len);
    }

    fn logical_bytes(&self) -> usize {
        let validity = (self.logical_slots * self.kind.validity_bits()).div_ceil(8);
        self.logical_slots * self.kind.width() + validity + self.rings.len() * size_of::<Ring>()
    }
}

#[derive(Debug, Default)]
struct ValueLog {
    name: String,
    chunks: Vec<ArrayRef>,
    len: u32,
    logical_bytes: usize,
}

impl ValueLog {
    fn truncate(&mut self, chunks: usize, len: u32) {
        self.chunks.truncate(chunks);
        self.len = len;
        self.logical_bytes = self
            .chunks
            .iter()
            .map(|chunk| chunk.get_array_memory_size())
            .sum();
    }
}

#[derive(Debug)]
enum Undo {
    Entity,
    Keyed {
        column: usize,
        row: usize,
        value: KeyedValue,
    },
    RingMetadata {
        column: usize,
        row: usize,
        head: usize,
        len: usize,
        capacity: usize,
        logical_slots: usize,
        logical_samples: usize,
    },
    RingSlot {
        column: usize,
        row: usize,
        slot: usize,
        value: RingValue,
    },
    ValueLog {
        column: usize,
        chunks: usize,
        len: u32,
    },
}

#[derive(Debug)]
struct Transaction {
    id: u64,
    undo: Vec<Undo>,
}

#[derive(Clone)]
struct KeyedSnapshot {
    name: String,
    kind: ValueKind,
    values: ArrayRef,
}

#[derive(Clone)]
struct RingSnapshot {
    name: String,
    kind: RingKind,
    retention: Retention,
    values: ArrayRef,
    offsets: Vec<usize>,
    capacities: Vec<usize>,
}

#[derive(Clone)]
struct ValueLogSnapshot {
    name: String,
    chunks: Vec<ArrayRef>,
    len: u32,
}

#[derive(Clone)]
pub(crate) struct TableSnapshot {
    keys: Vec<EntityKey>,
    keyed: Vec<KeyedSnapshot>,
    rings: Vec<RingSnapshot>,
    value_logs: Vec<ValueLogSnapshot>,
    logical_bytes: usize,
}

impl TableSnapshot {
    pub(crate) fn logical_bytes(&self) -> usize {
        self.logical_bytes
    }
}

#[derive(Debug, Default)]
pub(crate) struct StateTable {
    keys: Vec<EntityKey>,
    entities: HashMap<Vec<u8>, EntityId>,
    keyed: Vec<KeyedColumn>,
    rings: Vec<RingColumn>,
    value_logs: Vec<ValueLog>,
    dictionary_bytes: usize,
    transaction: Option<Transaction>,
    next_transaction: u64,
}

impl StateTable {
    pub(crate) fn new() -> Self {
        Self {
            next_transaction: 1,
            ..Self::default()
        }
    }

    pub(crate) fn add_keyed_column(&mut self, name: &str, kind: ValueKind) -> usize {
        assert!(
            self.transaction.is_none() && self.keys.is_empty(),
            "keyed schema must precede entities"
        );
        self.keyed.push(KeyedColumn::new(name.to_owned(), kind));
        self.keyed.len() - 1
    }

    pub(crate) fn add_ring_column(
        &mut self,
        name: &str,
        kind: RingKind,
        retention: Retention,
    ) -> usize {
        assert!(
            self.transaction.is_none() && self.keys.is_empty(),
            "ring schema must precede entities"
        );
        self.rings
            .push(RingColumn::new(name.to_owned(), kind, retention));
        self.rings.len() - 1
    }

    pub(crate) fn add_value_log(&mut self, name: &str) -> usize {
        assert!(
            self.transaction.is_none(),
            "cannot alter a state table inside a transaction"
        );
        self.value_logs.push(ValueLog {
            name: name.to_owned(),
            ..ValueLog::default()
        });
        self.value_logs.len() - 1
    }

    pub(crate) fn begin(&mut self) -> Result<()> {
        if self.transaction.is_some() {
            return Err(internal("rolling state transaction is already active"));
        }
        let id = self.next_transaction;
        self.next_transaction = self
            .next_transaction
            .checked_add(1)
            .ok_or_else(|| internal("rolling state transaction stamp exhausted"))?;
        self.transaction = Some(Transaction {
            id,
            undo: Vec::new(),
        });
        Ok(())
    }

    pub(crate) fn commit(&mut self) -> Result<()> {
        self.transaction
            .take()
            .map(|_| ())
            .ok_or_else(|| internal("rolling state transaction is absent"))
    }

    pub(crate) fn rollback(&mut self) -> Result<()> {
        let transaction = self
            .transaction
            .take()
            .ok_or_else(|| internal("rolling state transaction is absent"))?;
        for undo in transaction.undo.into_iter().rev() {
            match undo {
                Undo::Entity => {
                    let key = self
                        .keys
                        .pop()
                        .ok_or_else(|| internal("rolling entity undo underflowed"))?;
                    self.entities.remove(key.bytes.as_slice());
                    self.dictionary_bytes -= key.bytes.len();
                    let len = self.keys.len();
                    for column in &mut self.keyed {
                        column.truncate(len);
                    }
                    for column in &mut self.rings {
                        column.truncate(len);
                    }
                }
                Undo::Keyed { column, row, value } => self.keyed[column].values[row] = value,
                Undo::RingMetadata {
                    column,
                    row,
                    head,
                    len,
                    capacity,
                    logical_slots,
                    logical_samples,
                } => {
                    let column = &mut self.rings[column];
                    let ring = &mut column.rings[row];
                    ring.slots.resize(capacity, column.kind.empty());
                    ring.slot_stamps.resize(capacity, 0);
                    ring.head = head;
                    ring.len = len;
                    column.logical_slots = logical_slots;
                    column.logical_samples = logical_samples;
                }
                Undo::RingSlot {
                    column,
                    row,
                    slot,
                    value,
                } => self.rings[column].rings[row].slots[slot] = value,
                Undo::ValueLog {
                    column,
                    chunks,
                    len,
                } => self.value_logs[column].truncate(chunks, len),
            }
        }
        Ok(())
    }

    pub(crate) fn resolve_or_append(&mut self, key: EntityKey) -> Result<EntityId> {
        if let Some(entity) = self.entities.get(key.bytes.as_slice()) {
            return Ok(*entity);
        }
        let id = u32::try_from(self.keys.len()).map_err(|_| CalcFlowError::Operator {
            node_id: "rolling".to_owned(),
            message: "rolling entity dictionary exhausted u32 IDs".to_owned(),
        })?;
        let transaction = self
            .transaction
            .as_mut()
            .ok_or_else(|| internal("entity append requires a transaction"))?;
        self.dictionary_bytes += key.bytes.len();
        self.entities.insert(key.bytes.clone(), EntityId(id));
        self.keys.push(key);
        for column in &mut self.keyed {
            column.append();
        }
        for column in &mut self.rings {
            column.append();
        }
        transaction.undo.push(Undo::Entity);
        Ok(EntityId(id))
    }

    pub(crate) fn entity_count(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn set_u64(
        &mut self,
        column_index: usize,
        entity: EntityId,
        value: u64,
    ) -> Result<()> {
        let row = self.row(entity)?;
        let transaction = self
            .transaction
            .as_mut()
            .ok_or_else(|| internal("keyed update requires a transaction"))?;
        let column = &mut self.keyed[column_index];
        if column.kind != ValueKind::UInt64 {
            return Err(internal("keyed column type mismatch"));
        }
        if column.stamps[row] != transaction.id {
            column.stamps[row] = transaction.id;
            transaction.undo.push(Undo::Keyed {
                column: column_index,
                row,
                value: column.values[row],
            });
        }
        column.values[row] = KeyedValue::UInt64(value);
        Ok(())
    }

    pub(crate) fn u64(&self, column: usize, entity: EntityId) -> Result<u64> {
        match self.keyed[column].values[self.row(entity)?] {
            KeyedValue::UInt64(value) => Ok(value),
            _ => Err(internal("keyed column type mismatch")),
        }
    }

    pub(crate) fn push_f64(
        &mut self,
        column: usize,
        entity: EntityId,
        value: Option<f64>,
    ) -> Result<Option<Option<f64>>> {
        let row = self.row(entity)?;
        if self.rings[column].kind != RingKind::Float64 {
            return Err(internal("ring column type mismatch"));
        }
        self.push_ring(column, row, RingValue::Float64(value))
            .map(|evicted| match evicted {
                Some(RingValue::Float64(value)) => Some(value),
                None => None,
                _ => unreachable!(),
            })
    }

    fn push_ring(
        &mut self,
        column_index: usize,
        row: usize,
        value: RingValue,
    ) -> Result<Option<RingValue>> {
        let transaction = self
            .transaction
            .as_mut()
            .ok_or_else(|| internal("ring update requires a transaction"))?;
        let id = transaction.id;
        let column = &mut self.rings[column_index];
        if column.stamps[row] != id {
            column.stamps[row] = id;
            let ring = &column.rings[row];
            transaction.undo.push(Undo::RingMetadata {
                column: column_index,
                row,
                head: ring.head,
                len: ring.len,
                capacity: ring.capacity(),
                logical_slots: column.logical_slots,
                logical_samples: column.logical_samples,
            });
        }
        let ring = &mut column.rings[row];
        if ring.len == ring.capacity() {
            if matches!(column.retention, Retention::Rows(_)) && ring.capacity() == 0 {
                return Err(internal("rolling Rows ring has zero capacity"));
            }
            if matches!(column.retention, Retention::Duration) {
                let capacity = if ring.capacity() == 0 {
                    1
                } else {
                    ring.capacity()
                        .checked_mul(2)
                        .ok_or_else(|| internal("rolling Duration ring capacity overflowed"))?
                };
                column.logical_slots += capacity - ring.capacity();
                ring.slots.resize(capacity, column.kind.empty());
                ring.slot_stamps.resize(capacity, 0);
            }
        }
        let slot = ring.position(ring.len);
        if ring.slot_stamps[slot] != id {
            ring.slot_stamps[slot] = id;
            transaction.undo.push(Undo::RingSlot {
                column: column_index,
                row,
                slot,
                value: ring.slots[slot],
            });
        }
        let evicted = (ring.len == ring.capacity()).then_some(ring.slots[slot]);
        if evicted.is_some() {
            ring.head = (ring.head + 1) % ring.capacity();
        } else {
            ring.len += 1;
            column.logical_samples += 1;
        }
        ring.slots[slot] = value;
        Ok(evicted)
    }

    pub(crate) fn f64_history(&self, column: usize, entity: EntityId) -> Result<Vec<Option<f64>>> {
        let ring = &self.rings[column].rings[self.row(entity)?];
        Ok((0..ring.len)
            .map(|offset| match ring.slots[ring.position(offset)] {
                RingValue::Float64(value) => value,
                _ => unreachable!(),
            })
            .collect())
    }

    pub(crate) fn ring_logical_bytes(&self, column: usize) -> usize {
        let column = &self.rings[column];
        column.logical_bytes() - column.rings.len() * size_of::<Ring>()
    }

    pub(crate) fn append_values(&mut self, column: usize, values: ArrayRef) -> Result<Range<u32>> {
        let transaction = self
            .transaction
            .as_mut()
            .ok_or_else(|| internal("value-log append requires a transaction"))?;
        let log = &mut self.value_logs[column];
        if log
            .chunks
            .first()
            .is_some_and(|first| first.data_type() != values.data_type())
        {
            return Err(internal("rolling value-log type changed"));
        }
        let start = log.len;
        let len = u32::try_from(values.len())
            .map_err(|_| internal("rolling value-log append exceeded u32 handles"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| internal("rolling value-log exhausted u32 handles"))?;
        transaction.undo.push(Undo::ValueLog {
            column,
            chunks: log.chunks.len(),
            len: log.len,
        });
        log.logical_bytes += values.get_array_memory_size();
        log.chunks.push(values);
        log.len = end;
        Ok(start..end)
    }

    pub(crate) fn append_strings(
        &mut self,
        column: &usize,
        values: StringArray,
    ) -> Result<Range<u32>> {
        self.append_values(*column, Arc::new(values))
    }

    pub(crate) fn value_log_len(&self, column: usize) -> u32 {
        self.value_logs[column].len
    }

    pub(crate) fn strings(&self, column: usize) -> Result<StringArray> {
        let log = &self.value_logs[column];
        let arrays = log
            .chunks
            .iter()
            .map(|array| array.as_ref())
            .collect::<Vec<_>>();
        let concatenated = datafusion::arrow::compute::concat(&arrays)
            .map_err(|error| internal(&error.to_string()))?;
        concatenated
            .as_any()
            .downcast_ref::<StringArray>()
            .cloned()
            .ok_or_else(|| internal("value-log type mismatch"))
    }

    pub(crate) fn logical_bytes(&self) -> usize {
        self.dictionary_bytes
            + self
                .keyed
                .iter()
                .map(KeyedColumn::logical_bytes)
                .sum::<usize>()
            + self
                .rings
                .iter()
                .map(RingColumn::logical_bytes)
                .sum::<usize>()
            + self
                .value_logs
                .iter()
                .map(|log| log.logical_bytes)
                .sum::<usize>()
    }

    pub(crate) fn snapshot(&self) -> TableSnapshot {
        assert!(
            self.transaction.is_none(),
            "checkpoints occur only between state transactions"
        );
        let keyed = self
            .keyed
            .iter()
            .map(|column| KeyedSnapshot {
                name: column.name.clone(),
                kind: column.kind,
                values: keyed_array(column.kind, &column.values),
            })
            .collect();
        let rings = self
            .rings
            .iter()
            .map(|column| {
                let mut values = Vec::with_capacity(column.logical_samples);
                let mut offsets = Vec::with_capacity(column.rings.len() + 1);
                let mut capacities = Vec::with_capacity(column.rings.len());
                offsets.push(0);
                for ring in &column.rings {
                    values.extend((0..ring.len).map(|offset| ring.slots[ring.position(offset)]));
                    offsets.push(values.len());
                    capacities.push(ring.capacity());
                }
                RingSnapshot {
                    name: column.name.clone(),
                    kind: column.kind,
                    retention: column.retention,
                    values: ring_array(column.kind, &values),
                    offsets,
                    capacities,
                }
            })
            .collect();
        TableSnapshot {
            keys: self.keys.clone(),
            keyed,
            rings,
            value_logs: self
                .value_logs
                .iter()
                .map(|log| ValueLogSnapshot {
                    name: log.name.clone(),
                    chunks: log.chunks.clone(),
                    len: log.len,
                })
                .collect(),
            logical_bytes: self.logical_bytes(),
        }
    }

    pub(crate) fn restore(&mut self, snapshot: TableSnapshot) -> Result<()> {
        if self.transaction.is_some() {
            return Err(internal("cannot restore during a state transaction"));
        }
        validate_inventory(self, &snapshot)?;
        let mut restored = StateTable::new();
        restored.keys = snapshot.keys;
        restored.entities = restored
            .keys
            .iter()
            .enumerate()
            .map(|(index, key)| (key.bytes.clone(), EntityId(index as u32)))
            .collect();
        restored.dictionary_bytes = restored.keys.iter().map(|key| key.bytes.len()).sum();
        restored.keyed = self
            .keyed
            .iter()
            .map(|column| KeyedColumn::new(column.name.clone(), column.kind))
            .collect();
        for (target, source) in restored.keyed.iter_mut().zip(&snapshot.keyed) {
            target.values = decode_keyed(source.kind, &source.values)?;
            target.stamps = vec![0; target.values.len()];
        }
        restored.rings = self
            .rings
            .iter()
            .zip(&snapshot.rings)
            .map(|(target, source)| {
                let mut column =
                    RingColumn::new(target.name.clone(), target.kind, target.retention);
                let values = decode_ring(source.kind, &source.values)?;
                for entity in 0..source.capacities.len() {
                    let mut ring = Ring::new(source.kind, source.capacities[entity]);
                    ring.len = source.offsets[entity + 1] - source.offsets[entity];
                    for offset in 0..ring.len {
                        let slot = ring.position(offset);
                        ring.slots[slot] = values[source.offsets[entity] + offset];
                    }
                    column.logical_slots += ring.capacity();
                    column.logical_samples += ring.len;
                    column.rings.push(ring);
                    column.stamps.push(0);
                }
                Ok(column)
            })
            .collect::<Result<Vec<_>>>()?;
        restored.value_logs = self
            .value_logs
            .iter()
            .zip(&snapshot.value_logs)
            .map(|(target, source)| {
                let logical_bytes = source
                    .chunks
                    .iter()
                    .map(|chunk| chunk.get_array_memory_size())
                    .sum();
                ValueLog {
                    name: target.name.clone(),
                    chunks: source.chunks.clone(),
                    len: source.len,
                    logical_bytes,
                }
            })
            .collect();
        restored.next_transaction = self.next_transaction;
        *self = restored;
        Ok(())
    }

    fn row(&self, entity: EntityId) -> Result<usize> {
        let row = entity.0 as usize;
        (row < self.keys.len())
            .then_some(row)
            .ok_or_else(|| internal("rolling entity ID is out of bounds"))
    }
}

fn internal(message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: message.to_owned(),
    }
}

fn keyed_array(kind: ValueKind, values: &[KeyedValue]) -> ArrayRef {
    match kind {
        ValueKind::UInt64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    KeyedValue::UInt64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<UInt64Array>(),
        ),
        ValueKind::Int64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    KeyedValue::Int64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<Int64Array>(),
        ),
        ValueKind::Float64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    KeyedValue::Float64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<Float64Array>(),
        ),
        ValueKind::Boolean => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    KeyedValue::Boolean(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<BooleanArray>(),
        ),
    }
}

fn decode_keyed(kind: ValueKind, array: &ArrayRef) -> Result<Vec<KeyedValue>> {
    Ok(match kind {
        ValueKind::UInt64 => array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| internal("keyed snapshot type mismatch"))?
            .values()
            .iter()
            .copied()
            .map(KeyedValue::UInt64)
            .collect(),
        ValueKind::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| internal("keyed snapshot type mismatch"))?
            .values()
            .iter()
            .copied()
            .map(KeyedValue::Int64)
            .collect(),
        ValueKind::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| internal("keyed snapshot type mismatch"))?
            .values()
            .iter()
            .copied()
            .map(KeyedValue::Float64)
            .collect(),
        ValueKind::Boolean => array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| internal("keyed snapshot type mismatch"))?
            .values()
            .iter()
            .map(KeyedValue::Boolean)
            .collect(),
    })
}

fn ring_array(kind: RingKind, values: &[RingValue]) -> ArrayRef {
    match kind {
        RingKind::Float64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    RingValue::Float64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<Float64Array>(),
        ),
        RingKind::Int64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    RingValue::Int64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<Int64Array>(),
        ),
        RingKind::UInt64 => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    RingValue::UInt64(value) => *value,
                    _ => unreachable!(),
                })
                .collect::<UInt64Array>(),
        ),
        RingKind::PairFloat64 => {
            let left = values
                .iter()
                .map(|value| match value {
                    RingValue::PairFloat64(value) => value.map(|pair| pair.0),
                    _ => unreachable!(),
                })
                .collect::<Float64Array>();
            let right = values
                .iter()
                .map(|value| match value {
                    RingValue::PairFloat64(value) => value.map(|pair| pair.1),
                    _ => unreachable!(),
                })
                .collect::<Float64Array>();
            Arc::new(StructArray::from(vec![
                (
                    Arc::new(Field::new("left", DataType::Float64, true)),
                    Arc::new(left) as ArrayRef,
                ),
                (
                    Arc::new(Field::new("right", DataType::Float64, true)),
                    Arc::new(right) as ArrayRef,
                ),
            ]))
        }
    }
}

fn decode_ring(kind: RingKind, array: &ArrayRef) -> Result<Vec<RingValue>> {
    Ok(match kind {
        RingKind::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| internal("ring snapshot type mismatch"))?
            .iter()
            .map(RingValue::Float64)
            .collect(),
        RingKind::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| internal("ring snapshot type mismatch"))?
            .iter()
            .map(RingValue::Int64)
            .collect(),
        RingKind::UInt64 => array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| internal("ring snapshot type mismatch"))?
            .iter()
            .map(RingValue::UInt64)
            .collect(),
        RingKind::PairFloat64 => {
            let pairs = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| internal("pair ring snapshot type mismatch"))?;
            let left = pairs
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| internal("pair ring left type mismatch"))?;
            let right = pairs
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| internal("pair ring right type mismatch"))?;
            (0..pairs.len())
                .map(|row| {
                    RingValue::PairFloat64(
                        (left.is_valid(row) && right.is_valid(row))
                            .then_some((left.value(row), right.value(row))),
                    )
                })
                .collect()
        }
    })
}

fn validate_inventory(table: &StateTable, snapshot: &TableSnapshot) -> Result<()> {
    let valid = table.keyed.len() == snapshot.keyed.len()
        && table.rings.len() == snapshot.rings.len()
        && table.value_logs.len() == snapshot.value_logs.len()
        && table
            .keyed
            .iter()
            .zip(&snapshot.keyed)
            .all(|(left, right)| left.name == right.name && left.kind == right.kind)
        && table
            .rings
            .iter()
            .zip(&snapshot.rings)
            .all(|(left, right)| {
                left.name == right.name
                    && left.kind == right.kind
                    && left.retention == right.retention
            })
        && table
            .value_logs
            .iter()
            .zip(&snapshot.value_logs)
            .all(|(left, right)| left.name == right.name);
    if !valid {
        return Err(internal("rolling state-table column inventory mismatch"));
    }
    for (column, snapshot) in table.keyed.iter().zip(&snapshot.keyed) {
        if snapshot.values.len() != snapshot.keys.len()
            || snapshot.values.data_type() != keyed_array(column.kind, &[]).data_type()
        {
            return Err(internal("rolling keyed snapshot shape mismatch"));
        }
        let _ = column;
    }
    for snapshot in &snapshot.rings {
        if snapshot.offsets.len() != snapshot.keys.len() + 1
            || snapshot.capacities.len() != snapshot.keys.len()
            || snapshot.offsets.first() != Some(&0)
            || snapshot.offsets.last() != Some(&snapshot.values.len())
            || snapshot
                .offsets
                .windows(2)
                .any(|offset| offset[0] > offset[1])
            || snapshot
                .capacities
                .iter()
                .zip(snapshot.offsets.windows(2))
                .any(|(capacity, offset)| offset[1] - offset[0] > *capacity)
        {
            return Err(internal("rolling ring snapshot violates its invariants"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{EntityKey, Retention, RingKind, StateTable, ValueKind};
    use datafusion::arrow::array::{Array, StringArray};

    #[test]
    fn rollback_restores_dictionary_keyed_rows_and_overwritten_ring_slots() {
        let mut table = StateTable::new();
        let counter = table.add_keyed_column("counter", ValueKind::UInt64);
        let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(2));
        table.begin().unwrap();
        let entity = table
            .resolve_or_append(EntityKey::new(b"a".to_vec()))
            .unwrap();
        table.set_u64(counter, entity, 1).unwrap();
        table.push_f64(history, entity, Some(1.0)).unwrap();
        table.push_f64(history, entity, Some(2.0)).unwrap();
        table.commit().unwrap();

        let before = table.snapshot();
        table.begin().unwrap();
        table.set_u64(counter, entity, 9).unwrap();
        table.push_f64(history, entity, Some(3.0)).unwrap();
        table.push_f64(history, entity, Some(4.0)).unwrap();
        let other = table
            .resolve_or_append(EntityKey::new(b"b".to_vec()))
            .unwrap();
        table.set_u64(counter, other, 5).unwrap();
        table.rollback().unwrap();

        assert_eq!(table.entity_count(), 1);
        assert_eq!(table.u64(counter, entity).unwrap(), 1);
        assert_eq!(
            table.f64_history(history, entity).unwrap(),
            vec![Some(1.0), Some(2.0)]
        );
        assert_eq!(table.logical_bytes(), before.logical_bytes());
        table.restore(before).unwrap();
        assert_eq!(table.entity_count(), 1);
    }

    #[test]
    fn rows_float64_history_is_charged_as_values_and_validity_bits() {
        let mut table = StateTable::new();
        let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(20));
        table.begin().unwrap();
        let entity = table.resolve_or_append(EntityKey::new(Vec::new())).unwrap();
        table.push_f64(history, entity, Some(1.0)).unwrap();
        table.commit().unwrap();

        assert_eq!(table.ring_logical_bytes(history), 20 * 8 + (20 + 7) / 8);
        assert!(table.logical_bytes() <= 20 * 8 + (20 + 7) / 8 + 128);
    }

    #[test]
    fn snapshot_restore_preserves_float_bits_and_ring_order() {
        let mut table = StateTable::new();
        let history = table.add_ring_column("history", RingKind::Float64, Retention::Rows(3));
        table.begin().unwrap();
        let entity = table
            .resolve_or_append(EntityKey::new(b"entity".to_vec()))
            .unwrap();
        for value in [
            Some(-0.0),
            None,
            Some(f64::from_bits(0x7ff8_0000_0000_0042)),
        ] {
            table.push_f64(history, entity, value).unwrap();
        }
        table.push_f64(history, entity, Some(4.0)).unwrap();
        table.commit().unwrap();
        let snapshot = table.snapshot();

        table.begin().unwrap();
        table.push_f64(history, entity, Some(99.0)).unwrap();
        table.commit().unwrap();
        table.restore(snapshot).unwrap();

        let restored = table.f64_history(history, entity).unwrap();
        assert_eq!(restored.len(), 3);
        assert!(restored[0].is_none());
        assert_eq!(
            restored[1].unwrap().to_bits(),
            f64::from_bits(0x7ff8_0000_0000_0042).to_bits()
        );
        assert_eq!(restored[2].unwrap().to_bits(), 4.0_f64.to_bits());
    }

    #[test]
    fn value_log_appends_are_interleaved_and_rolled_back() {
        let mut table = StateTable::new();
        let log = table.add_value_log("values");
        table.begin().unwrap();
        let first = table
            .append_strings(&log, StringArray::from(vec![Some("a"), None]))
            .unwrap();
        table.commit().unwrap();
        table.begin().unwrap();
        let second = table
            .append_strings(&log, StringArray::from(vec![Some("b")]))
            .unwrap();
        table.rollback().unwrap();

        assert_eq!(first, 0..2);
        assert_eq!(second, 2..3);
        assert_eq!(table.value_log_len(log), 2);
        let values = table.strings(&log).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values.value(0), "a");
        assert!(values.is_null(1));
    }

    #[test]
    fn restore_rejects_column_inventory_mismatch_before_mutating_state() {
        let mut table = StateTable::new();
        table.add_keyed_column("counter", ValueKind::UInt64);
        let snapshot = table.snapshot();
        let mut other = StateTable::new();
        other.add_keyed_column("different", ValueKind::UInt64);
        assert!(other.restore(snapshot).is_err());
        assert_eq!(other.entity_count(), 0);
    }
}
