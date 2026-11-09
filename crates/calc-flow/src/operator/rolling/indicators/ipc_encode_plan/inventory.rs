use super::{
    Check,
    body::Message,
    geometry::Window,
    math::{add, growing, sum},
};
use crate::Result;
use datafusion::arrow::array::ArrayData;

#[derive(Default)]
pub(super) struct Inventory {
    pub(super) record: Message,
    pub(super) dictionary_file: u64,
    pub(super) dictionary_owned: u64,
    pub(super) dictionary_headers: u64,
    pub(super) dictionary_count: u64,
    pub(super) max_header: u64,
    pub(super) temporaries: u64,
}
impl Inventory {
    pub(super) fn column(&mut self, data: &ArrayData, check: Check<'_>) -> Result<()> {
        self.record.visit(data, Window::whole(data), check)?;
        self.dictionaries(data, check)
    }
    fn dictionaries(&mut self, data: &ArrayData, check: Check<'_>) -> Result<()> {
        check()?;
        for child in data.child_data() {
            self.dictionaries(child, check)?;
        }
        if matches!(
            data.data_type(),
            datafusion::arrow::datatypes::DataType::Dictionary(_, _)
        ) {
            self.dictionary(data, check)?;
        }
        Ok(())
    }
    fn dictionary(&mut self, data: &ArrayData, check: Check<'_>) -> Result<()> {
        let values = data.child_data().first().ok_or_else(super::math::invalid)?;
        let mut message = Message::default();
        message.visit(values, Window::whole(values), check)?;
        self.push(&message)
    }
    fn push(&mut self, message: &Message) -> Result<()> {
        let header = message.header(true)?;
        self.dictionary_count = add(self.dictionary_count, 1)?;
        self.max_header = self.max_header.max(header);
        self.dictionary_headers = add(self.dictionary_headers, header)?;
        self.dictionary_file = sum([Ok(self.dictionary_file), super::frame(header, message.body)])?;
        self.dictionary_owned = sum([
            Ok(self.dictionary_owned),
            Ok(header),
            growing(message.body, 1),
        ])?;
        self.temporaries = sum([Ok(self.temporaries), message.workspace()])?;
        Ok(())
    }
}
