use datafusion::common::ScalarValue;

#[derive(Clone, Copy)]
pub(super) enum Column {
    Partition(usize),
    History(usize),
    Input(usize),
}
#[derive(Clone, Copy)]
pub(super) enum Data<'a> {
    Entity(&'a [Option<ScalarValue>]),
    History(&'a [ScalarValue]),
    Buffer(&'a [ScalarValue]),
    Empty,
}
impl<'a> Data<'a> {
    pub(super) fn cell(self, column: Column) -> Option<&'a ScalarValue> {
        match (self, column) {
            (Self::Entity(values), Column::Partition(index)) => values.get(index)?.as_ref(),
            (Self::History(values), Column::History(index))
            | (Self::Buffer(values), Column::History(index) | Column::Input(index)) => {
                values.get(index)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "cells_tests.rs"]
mod tests;
