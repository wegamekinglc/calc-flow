use super::Encoding;

pub(in super::super) const KEY_SHARDS: usize = 8;

pub(in super::super) fn key_shard(key: &Encoding) -> usize {
    let hash = key
        .as_slice()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        });
    usize::try_from((hash ^ (hash >> 32)) % KEY_SHARDS as u64).expect("bounded ASOF key shard")
}
