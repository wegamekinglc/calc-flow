use calc_flow::{
    BatchOperatorFactory, ProviderRegistry, StreamOperatorFactory,
    array::{self, NativeArrayFactory, NativeTableArrayFactory},
};
use std::sync::Arc;

#[test]
fn test_native_registration_installs_one_atomic_three_factory_bundle() {
    let registry = ProviderRegistry::default();
    array::register_native(&registry).unwrap();
    assert!(registry.resolve_batch("native", "array_expr", "1").is_ok());
    assert!(registry.resolve_batch("native", "table_array", "1").is_ok());
    assert!(
        registry
            .resolve_stream("native", "table_array", "1")
            .is_ok()
    );
    assert!(
        registry
            .resolve_stream("native", "array_expr", "1")
            .is_err()
    );
}

#[test]
fn test_native_registration_collision_never_publishes_other_lifecycles_or_replaces_existing_owner()
{
    for collision in 0..3 {
        let registry = ProviderRegistry::default();
        let batch: Option<Arc<dyn BatchOperatorFactory>> = match collision {
            0 => Some(Arc::new(NativeArrayFactory::default())),
            1 => Some(Arc::new(NativeTableArrayFactory::default())),
            _ => None,
        };
        let stream: Option<Arc<dyn StreamOperatorFactory>> = (collision == 2).then(|| {
            Arc::new(NativeTableArrayFactory::default()) as Arc<dyn StreamOperatorFactory>
        });
        let batch_name = if collision == 0 {
            "array_expr"
        } else {
            "table_array"
        };
        install_collision(&registry, batch_name, batch.as_ref(), stream.as_ref());
        assert!(
            array::register_native(&registry).is_err(),
            "a collision must reject the whole bundle"
        );
        assert_eq!(
            registry.resolve_batch("native", "array_expr", "1").is_ok(),
            collision == 0
        );
        assert_eq!(
            registry.resolve_batch("native", "table_array", "1").is_ok(),
            collision == 1
        );
        assert_eq!(
            registry
                .resolve_stream("native", "table_array", "1")
                .is_ok(),
            collision == 2
        );
        assert_retained_owners(&registry, batch_name, batch, stream);
    }
}

fn install_collision(
    registry: &ProviderRegistry,
    batch_name: &str,
    batch: Option<&Arc<dyn BatchOperatorFactory>>,
    stream: Option<&Arc<dyn StreamOperatorFactory>>,
) {
    if let Some(factory) = batch {
        registry
            .register_batch("native", batch_name, "1", Arc::clone(factory))
            .unwrap();
    }
    if let Some(factory) = stream {
        registry
            .register_stream("native", "table_array", "1", Arc::clone(factory))
            .unwrap();
    }
}
fn assert_retained_owners(
    registry: &ProviderRegistry,
    batch_name: &str,
    batch: Option<Arc<dyn BatchOperatorFactory>>,
    stream: Option<Arc<dyn StreamOperatorFactory>>,
) {
    if let Some(factory) = batch {
        let actual = registry.resolve_batch("native", batch_name, "1").unwrap();
        assert!(Arc::ptr_eq(&factory, &actual));
    }
    if let Some(factory) = stream {
        assert!(Arc::ptr_eq(
            &factory,
            &registry
                .resolve_stream("native", "table_array", "1")
                .unwrap()
        ));
    }
}
