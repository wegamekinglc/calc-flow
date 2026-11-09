use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    array::{Array, UInt8Array, UInt64Array},
    record_batch::RecordBatch,
};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
#[derive(Clone, Copy, Default)]
pub(super) struct Census {
    pub(super) entities: usize,
    pub(super) initialized: usize,
    pub(super) histories: usize,
    pub(super) pending: usize,
    pub(super) ewma: usize,
    pub(super) cumulative: usize,
    pub(super) indicators: usize,
    pub(super) max_history: usize,
}
impl Census {
    pub(super) fn retained_rows(self) -> Result<u64> {
        let rows = self
            .initialized
            .checked_add(self.histories)
            .and_then(|rows| rows.checked_add(self.pending))
            .ok_or_else(invalid)?;
        u64::try_from(rows).map_err(|_| invalid())
    }
}
pub(super) fn scan(record: &RecordBatch, check: Check<'_>) -> Result<Census> {
    check()?;
    let arrays = Controls::new(record)?;
    let mut cursor = Cursor::default();
    for row in 0..record.num_rows() {
        check()?;
        cursor.observe(&arrays, row)?;
    }
    check()?;
    Ok(cursor.counts)
}
struct Controls<'a> {
    kinds: &'a UInt8Array,
    entities: &'a UInt64Array,
    positions: &'a UInt64Array,
}
impl<'a> Controls<'a> {
    fn new(record: &'a RecordBatch) -> Result<Self> {
        Ok(Self {
            kinds: typed(record, 0)?,
            entities: typed(record, 1)?,
            positions: typed(record, 2)?,
        })
    }
    fn row(&self, row: usize) -> Result<Header> {
        if self.kinds.is_null(row) || self.entities.is_null(row) {
            return Err(invalid());
        }
        Ok(Header {
            kind: self.kinds.value(row),
            entity: self.entities.value(row),
            position: (!self.positions.is_null(row)).then(|| self.positions.value(row)),
        })
    }
}
struct Header {
    kind: u8,
    entity: u64,
    position: Option<u64>,
}
#[derive(Default)]
struct Cursor {
    counts: Census,
    last_kind: u8,
    history: History,
}
impl Cursor {
    fn observe(&mut self, arrays: &Controls<'_>, row: usize) -> Result<()> {
        let header = arrays.row(row)?;
        if header.kind > 5 || header.kind < self.last_kind {
            return Err(invalid());
        }
        if header.kind == 0 {
            self.entity(&header)?;
        } else {
            self.other(arrays, &header)?;
        }
        self.last_kind = header.kind;
        Ok(())
    }
    fn entity(&mut self, header: &Header) -> Result<()> {
        if header.entity != number(self.counts.entities)? {
            return Err(invalid());
        }
        let count = header.position.ok_or_else(invalid)?;
        increment(&mut self.counts.entities)?;
        if count > 0 {
            increment(&mut self.counts.initialized)?;
        }
        Ok(())
    }
    fn other(&mut self, arrays: &Controls<'_>, header: &Header) -> Result<()> {
        self.reference(arrays, header)?;
        position(header)?;
        if header.kind == 1 {
            let retained = self.history.observe(header)?;
            increment(&mut self.counts.histories)?;
            self.counts.max_history = self.counts.max_history.max(retained);
        } else {
            self.increment_kind(header.kind)?;
        }
        Ok(())
    }
    fn reference(&self, arrays: &Controls<'_>, header: &Header) -> Result<()> {
        let entity = usize::try_from(header.entity).map_err(|_| invalid())?;
        if entity >= self.counts.entities {
            return Err(invalid());
        }
        if header.kind != 2 && arrays.positions.value(entity) == 0 {
            return Err(invalid());
        }
        Ok(())
    }
    fn increment_kind(&mut self, kind: u8) -> Result<()> {
        increment(self.counter(kind)?)
    }
    fn counter(&mut self, kind: u8) -> Result<&mut usize> {
        match kind {
            2 => Ok(&mut self.counts.pending),
            3 => Ok(&mut self.counts.ewma),
            4 => Ok(&mut self.counts.cumulative),
            5 => Ok(&mut self.counts.indicators),
            _ => Err(invalid()),
        }
    }
}
#[derive(Default)]
struct History {
    entity: Option<u64>,
    next: usize,
}
impl History {
    fn observe(&mut self, header: &Header) -> Result<usize> {
        self.entity(header.entity)?;
        if header.position != Some(number(self.next)?) {
            return Err(invalid());
        }
        increment(&mut self.next)?;
        Ok(self.next)
    }
    fn entity(&mut self, entity: u64) -> Result<()> {
        if self.entity.is_some_and(|previous| previous > entity) {
            return Err(invalid());
        }
        if self.entity != Some(entity) {
            self.entity = Some(entity);
            self.next = 0;
        }
        Ok(())
    }
}
fn position(header: &Header) -> Result<()> {
    let required = matches!(header.kind, 1 | 4);
    if required != header.position.is_some() {
        return Err(invalid());
    }
    Ok(())
}
fn increment(count: &mut usize) -> Result<()> {
    *count = count.checked_add(1).ok_or_else(invalid)?;
    Ok(())
}
fn number(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid())
}
fn typed<T: 'static>(record: &RecordBatch, index: usize) -> Result<&T> {
    record
        .columns()
        .get(index)
        .and_then(|column| column.as_any().downcast_ref())
        .ok_or_else(invalid)
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling restore control census is invalid".into(),
    }
}

#[cfg(test)]
#[path = "census_tests.rs"]
mod tests;
