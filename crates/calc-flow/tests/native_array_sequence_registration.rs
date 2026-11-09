use std::sync::Arc;

use calc_flow::{
    BatchOperatorFactory, ProviderRegistry, StreamOperatorFactory,
    array::{
        NativeArrayFactory, NativeArraySequenceFactory, NativeTableArrayFactory, register_native,
    },
};
use serde_json::json;

#[test]
fn test_native_bundle_registers_exactly_four_lifecycle_entries() {
    let registry = ProviderRegistry::default();
    register_native(&registry).unwrap();
    assert!(registry.resolve_batch("native", "array_expr", "1").is_ok());
    assert!(registry.resolve_batch("native", "table_array", "1").is_ok());
    assert!(
        registry
            .resolve_stream("native", "table_array", "1")
            .is_ok()
    );
    assert!(
        registry
            .resolve_stream("native", "array_sequence", "1")
            .is_ok()
    );
    assert!(
        registry
            .resolve_stream("native", "array_expr", "1")
            .is_err()
    );
    assert!(
        registry
            .resolve_batch("native", "array_sequence", "1")
            .is_err()
    );
}

#[test]
fn test_native_descriptors_have_explicit_sequence_only_state_layouts() {
    let array = NativeArrayFactory::capability_descriptor();
    let table = NativeTableArrayFactory::capability_descriptor();
    let sequence = NativeArraySequenceFactory::capability_descriptor();
    assert_eq!(array["state_layouts"], json!([]));
    assert_eq!(table["state_layouts"], json!([]));
    assert_eq!(sequence["state_layouts"], json!([1, 2]));
    assert_eq!(sequence["partition_contract"], "single_ordered_sequence");
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Slot {
    ArrayBatch,
    TableBatch,
    TableStream,
    SequenceStream,
}
const SLOTS: [Slot; 4] = [
    Slot::ArrayBatch,
    Slot::TableBatch,
    Slot::TableStream,
    Slot::SequenceStream,
];

enum Factory {
    Batch(Arc<dyn BatchOperatorFactory>),
    Stream(Arc<dyn StreamOperatorFactory>),
}
impl Slot {
    fn name(self) -> &'static str {
        match self {
            Self::ArrayBatch => "array_expr",
            Self::TableBatch | Self::TableStream => "table_array",
            Self::SequenceStream => "array_sequence",
        }
    }
    fn factory(self) -> Factory {
        match self {
            Self::ArrayBatch => Factory::Batch(Arc::new(NativeArrayFactory::default())),
            Self::TableBatch => Factory::Batch(Arc::new(NativeTableArrayFactory::default())),
            Self::TableStream => Factory::Stream(Arc::new(NativeTableArrayFactory::default())),
            Self::SequenceStream => {
                Factory::Stream(Arc::new(NativeArraySequenceFactory::default()))
            }
        }
    }
    fn resolve(self, registry: &ProviderRegistry) -> Option<Factory> {
        match self {
            Self::ArrayBatch | Self::TableBatch => registry
                .resolve_batch("native", self.name(), "1")
                .ok()
                .map(Factory::Batch),
            Self::TableStream | Self::SequenceStream => registry
                .resolve_stream("native", self.name(), "1")
                .ok()
                .map(Factory::Stream),
        }
    }
}
impl Factory {
    fn register(&self, registry: &ProviderRegistry, name: &str) {
        match self {
            Self::Batch(factory) => registry
                .register_batch("native", name, "1", Arc::clone(factory))
                .unwrap(),
            Self::Stream(factory) => registry
                .register_stream("native", name, "1", Arc::clone(factory))
                .unwrap(),
        }
    }
    fn same_owner(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Batch(left), Self::Batch(right)) => Arc::ptr_eq(left, right),
            (Self::Stream(left), Self::Stream(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }
}
fn assert_only_original(registry: &ProviderRegistry, occupied: Slot, original: &Factory) {
    for slot in SLOTS {
        match slot.resolve(registry) {
            Some(factory) => {
                assert_eq!(slot, occupied);
                assert!(factory.same_owner(original));
            }
            None => assert_ne!(slot, occupied),
        }
    }
}
#[test]
fn test_native_bundle_collision_at_any_lifecycle_preserves_all_existing_owners() {
    for occupied in SLOTS {
        let registry = ProviderRegistry::default();
        let original = occupied.factory();
        original.register(&registry, occupied.name());
        assert!(
            register_native(&registry).is_err(),
            "collision at {occupied:?}"
        );
        assert_only_original(&registry, occupied, &original);
    }
}
