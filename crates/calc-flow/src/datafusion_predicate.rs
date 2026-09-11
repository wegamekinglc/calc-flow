//! Exact unsigned-integer specialization of Boolean SQL predicates.

use datafusion::{
    arrow::datatypes::DataType,
    common::{
        DFSchema, Result, ScalarValue,
        tree_node::{Transformed, TreeNode},
    },
    logical_expr::{Expr, ExprSchemable, LogicalPlan, Operator, expr::BinaryExpr},
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
};

/// Avoids decimal intermediates only where the complete unsigned domain has
/// exactly the same Boolean result. Arithmetic projections keep their types.
#[derive(Debug)]
pub(crate) struct UInt64ModuloPredicate;

impl OptimizerRule for UInt64ModuloPredicate {
    fn name(&self) -> &'static str {
        "calc_flow_uint64_modulo_predicate"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Filter(mut filter) = plan else {
            return Ok(Transformed::no(plan));
        };
        let predicate = filter.predicate.transform_up(|expression| {
            Ok(
                match specialize_comparison(&expression, filter.input.schema()) {
                    Some(replacement) => Transformed::yes(replacement),
                    None => Transformed::no(expression),
                },
            )
        })?;
        filter.predicate = predicate.data;
        Ok(if predicate.transformed {
            Transformed::yes(LogicalPlan::Filter(filter))
        } else {
            Transformed::no(LogicalPlan::Filter(filter))
        })
    }
}

fn specialize_comparison(expression: &Expr, schema: &DFSchema) -> Option<Expr> {
    let Expr::BinaryExpr(comparison) = expression else {
        return None;
    };
    if !matches!(
        comparison.op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
    ) {
        return None;
    }
    let (left, right) = if let Some(modulo) = unsigned_modulo(&comparison.left, schema) {
        (
            modulo,
            unsigned_literal(decimal_integer(&comparison.right)?),
        )
    } else {
        (
            unsigned_literal(decimal_integer(&comparison.left)?),
            unsigned_modulo(&comparison.right, schema)?,
        )
    };
    Some(Expr::BinaryExpr(BinaryExpr::new(
        Box::new(left),
        comparison.op,
        Box::new(right),
    )))
}

fn unsigned_modulo(expression: &Expr, schema: &DFSchema) -> Option<Expr> {
    let Expr::BinaryExpr(modulo) = expression else {
        return None;
    };
    if modulo.op != Operator::Modulo {
        return None;
    }
    let divisor = decimal_integer(&modulo.right)?;
    if divisor == 0 {
        return None;
    }
    let Expr::Cast(cast) = modulo.left.as_ref() else {
        return None;
    };
    if cast.field.data_type() != &DataType::Decimal128(20, 0)
        || !matches!(cast.expr.as_ref(), Expr::Column(_))
        || cast.expr.get_type(schema).ok()? != DataType::UInt64
    {
        return None;
    }
    Some(Expr::BinaryExpr(BinaryExpr::new(
        cast.expr.clone(),
        Operator::Modulo,
        Box::new(unsigned_literal(divisor)),
    )))
}

fn decimal_integer(expression: &Expr) -> Option<u64> {
    let Expr::Literal(ScalarValue::Decimal128(Some(value), 20, 0), _) = expression else {
        return None;
    };
    u64::try_from(*value).ok()
}

fn unsigned_literal(value: u64) -> Expr {
    Expr::Literal(ScalarValue::UInt64(Some(value)), None)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use datafusion::{
        arrow::{
            array::UInt64Array,
            compute::concat_batches,
            datatypes::{DataType, Field, Schema},
            record_batch::RecordBatch,
        },
        common::{DFSchema, ScalarValue},
        datasource::MemTable,
        execution::context::{SessionConfig, SessionContext},
        logical_expr::{Expr, ExprSchemable, Operator, col, expr::Cast},
    };

    use super::{specialize_comparison, unsigned_literal};
    use crate::{Batch, BatchMetadata, DataFusionConfig, DataFusionRuntime};

    fn unsigned_input() -> Batch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "sequence",
            DataType::UInt64,
            true,
        )]));
        let values = vec![
            None,
            Some(0),
            Some(1),
            Some(3),
            Some(4),
            Some(7),
            Some(8),
            Some((1_u64 << 63) - 1),
            Some(1_u64 << 63),
            Some((1_u64 << 63) + 1),
            Some(u64::MAX - 1),
            Some(u64::MAX),
        ];
        let record =
            RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(values))]).unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    }

    #[tokio::test]
    async fn uint64_filter_avoids_decimal_modulo_with_either_rolling_setting() {
        for enable_rolling_rewrite in [false, true] {
            let runtime = DataFusionRuntime::new(DataFusionConfig {
                enable_rolling_rewrite,
                ..DataFusionConfig::default()
            })
            .unwrap();
            let result = runtime
                .sql(
                    "SELECT sequence FROM input WHERE sequence % 4 = 0",
                    &BTreeMap::from([("input".into(), unsigned_input())]),
                    None,
                )
                .await
                .unwrap();
            let actual = result.table_payload().unwrap().batches()[0]
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(actual, &UInt64Array::from(vec![0, 4, 8, 1_u64 << 63]));
            let plan = &runtime.metrics()[0].physical_plan;
            assert!(
                !plan.contains("Decimal128"),
                "UInt64 modulo comparison still materializes decimal arithmetic: {plan}"
            );
        }
    }

    async fn baseline_sql(query: &str, input: &Batch) -> RecordBatch {
        let context =
            SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
        let table = input.table_payload().unwrap();
        context
            .register_table(
                "input",
                Arc::new(
                    MemTable::try_new(Arc::clone(table.schema()), vec![table.batches().to_vec()])
                        .unwrap(),
                ),
            )
            .unwrap();
        let dataframe = context.sql(query).await.unwrap();
        let schema = Arc::new(dataframe.schema().as_arrow().clone());
        concat_batches(&schema, &dataframe.collect().await.unwrap()).unwrap()
    }

    async fn optimized_sql(query: &str, input: &Batch) -> RecordBatch {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let result = runtime
            .sql(
                query,
                &BTreeMap::from([("input".into(), input.clone())]),
                None,
            )
            .await
            .unwrap();
        let table = result.table_payload().unwrap();
        concat_batches(table.schema(), table.batches()).unwrap()
    }

    #[tokio::test]
    async fn uint64_filter_comparisons_match_datafusion_across_integer_boundaries() {
        let input = unsigned_input();
        for divisor in [1_u64, 3, 4, (1_u64 << 63) - 1, u64::MAX] {
            for constant in [0, 1, 3, 4, 1_u64 << 63, u64::MAX] {
                for operator in ["=", "!=", "<", "<=", ">", ">="] {
                    for predicate in [
                        format!("sequence % {divisor} {operator} {constant}"),
                        format!("{constant} {operator} sequence % {divisor}"),
                    ] {
                        let query = format!("SELECT sequence FROM input WHERE {predicate}");
                        assert_eq!(
                            optimized_sql(&query, &input).await,
                            baseline_sql(&query, &input).await,
                            "{query}"
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn uint64_filter_keeps_projected_decimal_type_and_boolean_nullability() {
        let input = unsigned_input();
        for query in [
            "SELECT sequence % 4 = 0 AS predicate, sequence % 4 AS remainder FROM input",
            "SELECT sequence % 4 AS remainder FROM input WHERE sequence % 4 = 0",
            "SELECT sequence FROM input WHERE sequence % 4 = 0 OR sequence IS NULL",
            "SELECT sequence FROM input WHERE NOT (sequence % 4 != 0) AND sequence > 2",
        ] {
            assert_eq!(
                optimized_sql(query, &input).await,
                baseline_sql(query, &input).await,
                "{query}"
            );
        }
        let query = "SELECT sequence % 4 = 0 AS predicate, sequence % 4 AS remainder FROM input";
        let output = optimized_sql(query, &input).await;
        assert_eq!(output.schema().field(0).data_type(), &DataType::Boolean);
        assert!(output.schema().field(0).is_nullable());
        assert_eq!(
            output.schema().field(1).data_type(),
            &DataType::Decimal128(20, 0)
        );
    }

    fn decimal(value: i128) -> Expr {
        Expr::Literal(ScalarValue::Decimal128(Some(value), 20, 0), None)
    }

    fn cast(expression: Expr, data_type: DataType) -> Expr {
        Expr::Cast(Cast::new(Box::new(expression), data_type))
    }

    fn modulo_predicate(operand: Expr, divisor: Expr, constant: Expr) -> Expr {
        Expr::BinaryExpr(datafusion::logical_expr::expr::BinaryExpr::new(
            Box::new(operand % divisor),
            Operator::Eq,
            Box::new(constant),
        ))
    }

    #[test]
    fn uint64_filter_specialization_rejects_unproved_types_and_arithmetic() {
        let schema = DFSchema::try_from(Schema::new(vec![
            Field::new("sequence", DataType::UInt64, true),
            Field::new("signed", DataType::Int64, true),
        ]))
        .unwrap();
        let operand = cast(col("sequence"), DataType::Decimal128(20, 0));
        let predicate = modulo_predicate(operand.clone(), decimal(4), decimal(0));
        let specialized = specialize_comparison(&predicate, &schema).unwrap();
        assert_eq!(specialized.get_type(&schema).unwrap(), DataType::Boolean);
        assert!(specialized.nullable(&schema).unwrap());
        for divisor in [
            decimal(0),
            decimal(-1),
            decimal(i128::from(u64::MAX) + 1),
            Expr::Literal(ScalarValue::Float64(Some(4.0)), None),
            Expr::Literal(ScalarValue::Decimal128(Some(40), 20, 1), None),
            Expr::Literal(ScalarValue::Decimal128(None, 20, 0), None),
        ] {
            assert!(
                specialize_comparison(
                    &modulo_predicate(operand.clone(), divisor, decimal(0)),
                    &schema,
                )
                .is_none()
            );
        }
        for constant in [decimal(-1), decimal(i128::from(u64::MAX) + 1)] {
            assert!(
                specialize_comparison(
                    &modulo_predicate(operand.clone(), decimal(4), constant),
                    &schema,
                )
                .is_none()
            );
        }
        for unproved in [
            cast(col("signed"), DataType::Decimal128(20, 0)),
            cast(col("sequence"), DataType::Decimal128(20, 1)),
            cast(col("sequence"), DataType::Decimal128(19, 0)),
            cast(
                col("sequence") + unsigned_literal(1),
                DataType::Decimal128(20, 0),
            ),
            cast(col("missing"), DataType::Decimal128(20, 0)),
        ] {
            assert!(specialize_comparison(
                &modulo_predicate(unproved, decimal(4), decimal(0)),
                &schema,
            )
            .is_none());
        }
        assert!(specialize_comparison(&(operand % decimal(4)), &schema).is_none());
        assert!(specialize_comparison(&col("sequence"), &schema).is_none());
    }

    #[tokio::test]
    async fn uint64_filter_zero_divisor_retains_query_error_context() {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let error = runtime
            .sql(
                "SELECT sequence FROM input WHERE sequence % 0 = 0",
                &BTreeMap::from([("input".into(), unsigned_input())]),
                Some("filter_node"),
            )
            .await
            .unwrap_err();
        let crate::CalcFlowError::DataFusion { node_id, message } = error else {
            panic!("expected DataFusion error, got {error}");
        };
        assert_eq!(node_id.as_deref(), Some("filter_node"));
        assert!(
            message.to_lowercase().contains("divide by zero"),
            "{message}"
        );
    }
}
