use datafusion::arrow::{datatypes::DataType, ipc};

use super::{
    Check, Result,
    arrays::{self, Graph},
    invalid,
    math::{add, count},
    messages,
};

pub(super) struct Walk<'a, 'c> {
    pub(super) bytes: &'a [u8],
    pub(super) footer: ipc::Footer<'a>,
    pub(super) record: ipc::RecordBatch<'a>,
    pub(super) node: usize,
    pub(super) variadic: usize,
    pub(super) check: Check<'c>,
}
impl Walk<'_, '_> {
    pub(super) fn array(&mut self, field: ipc::Field<'_>, dtype: &DataType) -> Result<Graph> {
        (self.check)()?;
        let nodes = self.record.nodes().expect("verified nodes");
        let node = nodes.get(self.node);
        self.node = self
            .node
            .checked_add(1)
            .ok_or_else(|| invalid("node index"))?;
        let len = count(node.length())?;
        let buffers = self.buffer_count(dtype)?;
        let mut child = self.children(field, dtype)?;
        if matches!(dtype, DataType::Dictionary(_, _)) {
            child.work = 0;
        }
        arrays::build(dtype, len, buffers, child)
    }

    fn children(&mut self, field: ipc::Field<'_>, dtype: &DataType) -> Result<Graph> {
        if let DataType::Dictionary(_, value) = dtype {
            return self.dictionary(field, value);
        }
        let mut fields = field.children().into_iter().flatten();
        let mut children = Graph::default();
        messages::visit_children(dtype, &mut |dtype| {
            children.merge(self.array(
                fields.next().expect("verified child field"),
                dtype.data_type(),
            )?)
        })?;
        Ok(children)
    }

    fn dictionary(&mut self, field: ipc::Field<'_>, value: &DataType) -> Result<Graph> {
        let id = field.dictionary().expect("verified dictionary").id();
        let index = usize::try_from(id).map_err(|_| invalid("dictionary id"))?;
        let block = self
            .footer
            .dictionaries()
            .expect("verified dictionaries")
            .get(index);
        let record = messages::dictionary(self.bytes, block)?
            .data()
            .expect("verified dictionary data");
        let mut walk = Walk {
            bytes: self.bytes,
            footer: self.footer,
            record,
            node: 0,
            variadic: 0,
            check: self.check,
        };
        let graph = walk.array(field, value)?;
        let mut child = Graph::default();
        child.merge(graph)?;
        Ok(child)
    }

    fn buffer_count(&mut self, dtype: &DataType) -> Result<u64> {
        if matches!(dtype, DataType::BinaryView | DataType::Utf8View) {
            let counts = self
                .record
                .variadicBufferCounts()
                .expect("verified variadic counts");
            let buffers = add(count(counts.get(self.variadic))?, 1)?;
            self.variadic = self
                .variadic
                .checked_add(1)
                .ok_or_else(|| invalid("variadic index"))?;
            return Ok(buffers);
        }
        Ok(arrays::buffers(dtype))
    }
}
