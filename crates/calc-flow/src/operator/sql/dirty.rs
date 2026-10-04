use super::{MemoryReservation, Result, checked_bytes, df_error, ensure_reservation};

pub(super) struct DirtyGroups {
    bits: Vec<u64>,
    slots: Vec<usize>,
    reservation: MemoryReservation,
}

impl DirtyGroups {
    pub(super) fn empty(reservation: MemoryReservation) -> Self {
        Self {
            bits: Vec::new(),
            slots: Vec::new(),
            reservation,
        }
    }

    pub(super) fn slots(&self) -> &[usize] {
        &self.slots
    }

    pub(super) fn prepare(&self, groups: usize, name: &str) -> Result<Option<Self>> {
        if groups <= self.slots.capacity() {
            return Ok(None);
        }
        let capacity = groups
            .max(4)
            .checked_next_power_of_two()
            .ok_or_else(|| df_error(name, "dirty group capacity overflowed"))?;
        let words = capacity.div_ceil(64);
        let bytes = checked_bytes(
            4096,
            [(capacity, size_of::<usize>()), (words, size_of::<u64>())],
            name,
        )?;
        let reservation = self.reservation.new_empty();
        ensure_reservation(&reservation, bytes, name)?;
        let mut bits = vec![0; words];
        bits[..self.bits.len()].copy_from_slice(&self.bits);
        let mut slots = Vec::with_capacity(capacity);
        slots.extend_from_slice(&self.slots);
        Ok(Some(Self {
            bits,
            slots,
            reservation,
        }))
    }

    pub(super) fn mark(&mut self, slot: usize) {
        let bit = 1u64 << (slot % 64);
        let word = &mut self.bits[slot / 64];
        if *word & bit == 0 {
            *word |= bit;
            self.slots.push(slot);
        }
    }

    pub(super) fn clear(&mut self) {
        for &slot in &self.slots {
            self.bits[slot / 64] &= !(1u64 << (slot % 64));
        }
        self.slots.clear();
    }
}
