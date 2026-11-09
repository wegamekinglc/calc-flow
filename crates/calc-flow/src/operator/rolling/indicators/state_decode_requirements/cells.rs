use super::{Check, Plan, Result, Source, add, json};

#[derive(Default)]
struct Cells {
    retained: u64,
    dictionary: u64,
    control: u64,
    row_work: u64,
    constructor: u64,
    summary: u64,
    key_heap: u64,
    last_buffer: u64,
}
pub(super) fn plan(source: &Source<'_>, check: Check<'_>) -> Result<Plan> {
    let mut cells = Cells::default();
    for row in 0..source.record.num_rows() {
        check()?;
        observe(source, row, &mut cells, check)?;
    }
    cells.finish()
}
fn observe(source: &Source<'_>, row: usize, cells: &mut Cells, check: Check<'_>) -> Result<()> {
    let mut owned = 0;
    for index in 0..source.input.fields().len() {
        let (cell, temp) = source.cell(row, index, check)?;
        owned = add(owned, cell)?;
        cells.constructor = cells.constructor.max(temp);
    }
    let partition = source.partition_heap(row, check)?;
    let kind = source.kind(row);
    match kind {
        0 => cells.entity(source, row, owned, partition),
        1 | 2 => retained(source, row, owned, partition, cells, check),
        _ => control(source, row, owned, partition, cells, check),
    }
}
impl Cells {
    fn entity(
        &mut self,
        source: &Source<'_>,
        row: usize,
        owned: u64,
        partition: u64,
    ) -> Result<()> {
        self.dictionary = add(self.dictionary, add(owned, partition)?)?;
        self.row_work = self.row_work.max(partition);
        if source.initialized(row) {
            self.key_heap = add(self.key_heap, partition)?;
        }
        Ok(())
    }
    fn finish(self) -> Result<Plan> {
        let candidate_bytes = add(self.retained, self.key_heap)?;
        let scratch = add(self.dictionary, self.control)?;
        let scratch = add(scratch, self.row_work)?;
        let scratch = add(scratch, self.constructor)?;
        let scratch_bytes = add(scratch, self.summary)?;
        let scratch_bytes = add(scratch_bytes, self.last_buffer)?;
        Ok(Plan {
            candidate_bytes,
            scratch_bytes,
        })
    }
}
fn retained(
    source: &Source<'_>,
    row: usize,
    owned: u64,
    partition: u64,
    cells: &mut Cells,
    check: Check<'_>,
) -> Result<()> {
    cells.retained = add(cells.retained, add(owned, partition)?)?;
    let sequence = source.sequence_heap(row, check)?;
    let identity = add(partition, sequence)?;
    if source.kind(row) == 1 {
        history(source, row, partition, identity, cells, check)
    } else {
        pending(identity, cells)
    }
}
fn history(
    source: &Source<'_>,
    row: usize,
    partition: u64,
    identity: u64,
    cells: &mut Cells,
    check: Check<'_>,
) -> Result<()> {
    let clone = source.row_clone(source.entity(row)?, check)?;
    let tuple = add(clone, partition)?;
    cells.row_work = cells.row_work.max(add(tuple, identity)?);
    Ok(())
}
fn control(
    source: &Source<'_>,
    row: usize,
    owned: u64,
    partition: u64,
    cells: &mut Cells,
    check: Check<'_>,
) -> Result<()> {
    cells.control = cells.control.max(owned);
    cells.row_work = cells.row_work.max(partition);
    if source.kind(row) == 4
        && let Some(bytes) = source.summary(row)?
    {
        cells.summary = cells.summary.max(json::requirement(bytes, check)?);
    }
    Ok(())
}

fn pending(identity: u64, cells: &mut Cells) -> Result<()> {
    let copies = add(identity, identity)?;
    cells.retained = add(cells.retained, copies)?;
    cells.last_buffer = cells.last_buffer.max(copies);
    Ok(())
}
