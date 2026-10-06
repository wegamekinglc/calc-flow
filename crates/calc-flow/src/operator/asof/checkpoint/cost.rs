use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::operator::asof) struct EncodingCost {
    pub index_rows: usize,
    pub index_bytes: usize,
    pub ipc_rows: usize,
    pub ipc_bytes: usize,
}

thread_local! {
    static COST: Cell<EncodingCost> = const { Cell::new(EncodingCost {
        index_rows: 0, index_bytes: 0, ipc_rows: 0, ipc_bytes: 0,
    }) };
}

fn record(update: impl FnOnce(&mut EncodingCost)) {
    COST.with(|counter| {
        let mut cost = counter.get();
        update(&mut cost);
        counter.set(cost);
    });
}

pub(in crate::operator::asof) fn index_rows(rows: usize) {
    record(|cost| cost.index_rows += rows);
}

pub(in crate::operator::asof) fn index_bytes(bytes: usize) {
    record(|cost| cost.index_bytes += bytes);
}

pub(in crate::operator::asof) fn ipc(rows: usize, bytes: usize) {
    record(|cost| {
        cost.ipc_rows += rows;
        cost.ipc_bytes += bytes;
    });
}

pub(in crate::operator::asof) fn take() -> EncodingCost {
    COST.with(|counter| counter.replace(EncodingCost::default()))
}
