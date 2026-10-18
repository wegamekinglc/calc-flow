use crate::operator::join::retained::Rows;
use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;

use super::{
    Bindings, BirthBudget, Dictionary, Entry, KeyFlow, Lease, NamespaceLease, SideBindings,
    canonical_credit, counter_overflow, grant, table_bytes,
};
use crate::Result;

pub(in crate::operator::join) struct Rebuilt {
    pub(in crate::operator::join) dictionary: Dictionary,
    pub(in crate::operator::join) bindings: Bindings,
}

struct Builder<'a> {
    dictionary: Dictionary,
    master: &'a MemoryReservation,
    name: &'a str,
    ids: Vec<u32>,
    counts: [usize; 2],
    _scratch: MemoryReservation,
}

impl Dictionary {
    pub(in crate::operator::join) async fn rebuild(
        &self,
        rows: [&dyn Rows; 2],
        master: MemoryReservation,
        context: Option<&crate::StreamOperatorContext<'_>>,
        name: &str,
    ) -> Result<Option<Rebuilt>> {
        let mut flow = KeyFlow::new(context);
        self.build_rebuilt(rows, &master, name, &mut flow).await
    }

    pub(in crate::operator::join) fn rebuild_checked(
        &self,
        rows: [&dyn Rows; 2],
        master: &MemoryReservation,
        name: &str,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Option<Rebuilt>> {
        let mut flow = KeyFlow::checked(check);
        futures::executor::block_on(self.build_rebuilt(rows, master, name, &mut flow))
    }

    async fn build_rebuilt<C: Fn() -> Result<()>>(
        &self,
        rows: [&dyn Rows; 2],
        master: &MemoryReservation,
        name: &str,
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<Option<Rebuilt>> {
        if rows.iter().all(|rows| rows.is_empty()) {
            return Ok(Some(Rebuilt {
                dictionary: Dictionary::default(),
                bindings: Bindings::default(),
            }));
        }
        let Some(mut builder) = Builder::prepare(self, rows, master, name, flow).await? else {
            return Ok(None);
        };
        if !builder.visit_rows(rows, flow).await? {
            return Ok(None);
        }
        builder.finish(rows, flow).await
    }
}

impl<'a> Builder<'a> {
    async fn prepare<C: Fn() -> Result<()>>(
        source: &Dictionary,
        rows: [&dyn Rows; 2],
        master: &'a MemoryReservation,
        name: &'a str,
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<Option<Self>> {
        let count = rows[0]
            .len()
            .checked_add(rows[1].len())
            .ok_or_else(|| counter_overflow(name, "key_bindings"))?;
        let bytes = count
            .checked_mul(size_of::<u32>())
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| counter_overflow(name, "key_bindings"))?;
        let Some(scratch) = grant(master, bytes, name)? else {
            return Ok(None);
        };
        let Some(dictionary) = copy_dictionary(source, master, name, flow).await? else {
            return Ok(None);
        };
        Ok(Some(Self {
            dictionary,
            master,
            name,
            ids: Vec::with_capacity(count),
            counts: [0; 2],
            _scratch: scratch,
        }))
    }

    async fn visit_rows<C: Fn() -> Result<()>>(
        &mut self,
        rows: [&dyn Rows; 2],
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<bool> {
        for (side, rows) in rows.into_iter().enumerate() {
            for batch in rows.batches() {
                for reference in batch.references() {
                    flow.step(1, 0).await?;
                    let Some(id) = self.resolve(batch.key(reference), flow).await? else {
                        return Ok(false);
                    };
                    self.record(side, id, batch.key(reference))?;
                }
            }
        }
        Ok(true)
    }

    async fn resolve<C: Fn() -> Result<()>>(
        &mut self,
        key: &Arc<super::FramedKey>,
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<Option<u32>> {
        let (hash, existing) = self.dictionary.find_bytes(key, flow).await?;
        if existing.is_some() {
            return Ok(existing);
        }
        let id = self.dictionary.next_slot(0, self.name)?;
        if !self.grow(id as usize + 1, flow).await? {
            return Ok(None);
        }
        let Some(lease) = self.birth(id, key, flow).await? else {
            return Ok(None);
        };
        let entry = Some(Entry {
            lease,
            hash,
            retained: 1,
            next: None,
            previous: None,
            provisional: 0,
            touched_next: None,
        });
        self.install(id, entry);
        self.dictionary.link(id);
        Ok(Some(id))
    }

    fn install(&mut self, id: u32, entry: Option<Entry>) {
        if id as usize == self.dictionary.slots.len() {
            self.dictionary.slots.push(entry);
        } else {
            assert_eq!(
                self.dictionary.free.pop(),
                Some(id),
                "prepared free identity"
            );
            self.dictionary.slots[id as usize] = entry;
        }
    }

    async fn grow<C: Fn() -> Result<()>>(
        &mut self,
        count: usize,
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<bool> {
        if count <= self.dictionary.slots.capacity() {
            return Ok(true);
        }
        let capacity = count
            .checked_next_power_of_two()
            .map(|count| count.max(4))
            .ok_or_else(|| counter_overflow(self.name, "key_capacity"))?;
        let bytes =
            table_bytes(capacity).ok_or_else(|| counter_overflow(self.name, "key_capacity"))?;
        let Some(credit) = grant(self.master, bytes, self.name)? else {
            return Ok(false);
        };
        self.dictionary = self
            .dictionary
            .copy_for_capacity(capacity, credit, flow, true)
            .await?;
        Ok(true)
    }

    async fn birth<C: Fn() -> Result<()>>(
        &mut self,
        id: u32,
        bytes: &[u8],
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<Option<Lease>> {
        if !self.namespace()? {
            return Ok(None);
        }
        let budget = BirthBudget {
            master: self.master,
            name: self.name,
        };
        let Some(credits) = canonical_credit(bytes.len(), &budget)? else {
            return Ok(None);
        };
        let mut encoded = Vec::with_capacity(bytes.len());
        for chunk in bytes.chunks(4096) {
            flow.step(0, chunk.len()).await?;
            encoded.extend_from_slice(chunk);
        }
        #[cfg(test)]
        super::super::note_join_work(|work| work.key_encodings += 1);
        let namespace = self.dictionary.namespace.as_ref().unwrap();
        let address = namespace
            .address(id)
            .ok_or_else(|| counter_overflow(self.name, "key_id"))?;
        let key = super::super::columnar::identified_key(encoded, credits.canonical, address);
        Ok(Lease::new(namespace, id, key, credits.lease))
    }

    fn namespace(&mut self) -> Result<bool> {
        if self.dictionary.namespace.is_some() {
            return Ok(true);
        }
        let bytes = NamespaceLease::control_bytes()
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| counter_overflow(self.name, "key_namespace"))?;
        let Some(credit) = grant(self.master, bytes, self.name)? else {
            return Ok(false);
        };
        self.dictionary.namespace = Some(
            NamespaceLease::new(credit)
                .ok_or_else(|| counter_overflow(self.name, "key_namespace"))?,
        );
        Ok(true)
    }

    fn record(&mut self, side: usize, id: u32, key: &Arc<super::FramedKey>) -> Result<()> {
        let entry = self.dictionary.slots[id as usize].as_mut().unwrap();
        entry.provisional = entry
            .provisional
            .checked_add(1)
            .ok_or_else(|| counter_overflow(self.name, "key_references"))?;
        if !Arc::ptr_eq(entry.lease.owner(), key) {
            self.counts[side] = self.counts[side]
                .checked_add(1)
                .ok_or_else(|| counter_overflow(self.name, "key_bindings"))?;
        }
        self.ids.push(id);
        Ok(())
    }

    async fn finish<C: Fn() -> Result<()>>(
        mut self,
        rows: [&dyn Rows; 2],
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<Option<Rebuilt>> {
        self.finish_counts(flow).await?;
        let Some(left) = SideBindings::prepare(self.counts[0], self.master, self.name)? else {
            return Ok(None);
        };
        let Some(right) = SideBindings::prepare(self.counts[1], self.master, self.name)? else {
            return Ok(None);
        };
        let mut bindings = Bindings { left, right };
        let mut ids = self.ids.iter();
        for (side, rows) in [
            (&mut bindings.left, rows[0]),
            (&mut bindings.right, rows[1]),
        ] {
            for reference in rows.references() {
                flow.step(1, 0).await?;
                let lease = &self.dictionary.slots[*ids.next().unwrap() as usize]
                    .as_ref()
                    .unwrap()
                    .lease;
                if !Arc::ptr_eq(lease.owner(), rows.key(reference)) {
                    side.insert(rows.identity(reference).1, lease.clone());
                }
            }
        }
        Ok(Some(Rebuilt {
            dictionary: self.dictionary,
            bindings,
        }))
    }

    async fn finish_counts<C: Fn() -> Result<()>>(
        &mut self,
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<()> {
        for entry in self
            .dictionary
            .slots
            .iter_mut()
            .flatten()
            .filter(|entry| entry.retained > 0)
        {
            flow.step(1, 0).await?;
            if entry.provisional == 0 {
                return Err(super::super::operator_error(
                    self.name,
                    "live identity lacks retained rows",
                ));
            }
            entry.retained = entry.provisional;
            entry.provisional = 0;
        }
        Ok(())
    }
}

async fn copy_dictionary<C: Fn() -> Result<()>>(
    source: &Dictionary,
    master: &MemoryReservation,
    name: &str,
    flow: &mut KeyFlow<'_, '_, C>,
) -> Result<Option<Dictionary>> {
    if source.slots.is_empty() {
        return Ok(Some(Dictionary::default()));
    }
    let bytes = table_bytes(source.slots.capacity())
        .ok_or_else(|| counter_overflow(name, "key_capacity"))?;
    let Some(credit) = grant(master, bytes, name)? else {
        return Ok(None);
    };
    source
        .copy_for_capacity(source.slots.capacity(), credit, flow, false)
        .await
        .map(Some)
}
