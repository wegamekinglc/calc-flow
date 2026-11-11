//! Named finite relational reductions, lowered through the SQL planner.
use super::*;

/// An immutable reduction of a direct column, distinct from row-local expressions.
#[derive(Clone, Debug)]
pub struct AggregateExpr {
    function: &'static str,
    value: Option<ColumnExpr>,
}
/// An aggregate with an explicit output column name.
#[derive(Clone, Debug)]
pub struct NamedAggregate {
    pub(super) name: String,
    value: AggregateExpr,
}
/// Counts all input rows, including rows containing NULL.
pub fn count_rows() -> AggregateExpr {
    AggregateExpr {
        function: "COUNT",
        value: None,
    }
}
impl AggregateExpr {
    /// Names this reduction in an aggregate relation.
    ///
    /// # Errors
    /// Rejects an empty name.
    pub fn alias(&self, name: &str) -> Result<NamedAggregate> {
        check_name(name, "aggregate.alias")?;
        Ok(NamedAggregate {
            name: name.into(),
            value: self.clone(),
        })
    }
}
macro_rules! reducer {
    ($name:ident, $sql:literal, $doc:literal) => {
        impl ColumnExpr {
            #[doc = $doc]
            pub fn $name(&self) -> AggregateExpr {
                AggregateExpr {
                    function: $sql,
                    value: Some(self.clone()),
                }
            }
        }
    };
}
reducer!(
    sum,
    "SUM",
    "Sums non-NULL numeric values; empty input yields typed NULL."
);
reducer!(count, "COUNT", "Counts non-NULL values of a direct column.");
reducer!(min, "MIN", "Returns the minimum non-NULL numeric value.");
reducer!(max, "MAX", "Returns the maximum non-NULL numeric value.");
reducer!(
    mean,
    "AVG",
    "Returns the mean of non-NULL numeric values as Float64."
);
impl TableExpr {
    /// Reduces a finite relation, or emits cumulative snapshots in a stream.
    /// Columns are group keys followed by named reductions in declaration order.
    ///
    /// # Errors
    /// Rejects empty reductions and duplicate names. Planning validates direct
    /// column lineage and the supported numeric/group-key types.
    pub fn aggregate(
        &self,
        aggregates: Vec<NamedAggregate>,
        group_by: Vec<String>,
    ) -> Result<Self> {
        if aggregates.is_empty() {
            return Err(invalid(
                "aggregate.aggregates",
                "invalid_literal",
                "provide at least one aggregate",
            ));
        }
        let mut names = BTreeSet::new();
        for name in group_by.iter().chain(aggregates.iter().map(|a| &a.name)) {
            check_name(name, "aggregate")?;
            if !names.insert(name) {
                return Err(invalid(
                    "aggregate",
                    "duplicate_name",
                    format!("duplicate field '{name}'"),
                ));
            }
        }
        let id = digest(&json!([
            "aggregate",
            self.0.id,
            group_by,
            aggregates
                .iter()
                .map(|a| (
                    &a.name,
                    a.value.function,
                    a.value.value.as_ref().map(ColumnExpr::identity)
                ))
                .collect::<Vec<_>>()
        ]));
        Ok(Self(Arc::new(Relation {
            id,
            kind: RelationKind::Aggregate {
                input: self.clone(),
                aggregates,
                group_by,
            },
        })))
    }
}
pub(crate) fn query(
    input: &TableExpr,
    schema: &SchemaRef,
    aggregates: &[NamedAggregate],
    groups: &[String],
    path: &str,
) -> Result<String> {
    for (i, key) in groups.iter().enumerate() {
        let role = format!("{path}.aggregate.group_by[{i}]");
        let dtype = input.column(key)?.dtype(input.lineage(), schema, &role)?;
        if !group_type(&dtype) {
            return Err(invalid(
                &role,
                "unsupported_type",
                "group key requires boolean, integer or string",
            ));
        }
    }
    let mut select = groups.iter().map(|key| quote(key)).collect::<Vec<_>>();
    for aggregate in aggregates {
        let role = segment(&format!("{path}.aggregate.aggregates"), &aggregate.name);
        let operand = if let Some(value) = &aggregate.value.value {
            let dtype = value.dtype(input.lineage(), schema, &format!("{role}.value"))?;
            if !matches!(value.0.as_ref(), Scalar::Column { .. })
                || (aggregate.value.function != "COUNT" && !numeric_type(&dtype))
            {
                return Err(invalid(
                    &format!("{role}.value"),
                    "unsupported_type",
                    format!(
                        "{} requires a direct numeric column; found {dtype}",
                        aggregate.value.function
                    ),
                ));
            }
            value.render()
        } else {
            "*".into()
        };
        select.push(format!(
            "{}({operand}) AS {}",
            aggregate.value.function,
            quote(&aggregate.name)
        ));
    }
    let group = if groups.is_empty() {
        String::new()
    } else {
        format!(
            " GROUP BY {}",
            groups
                .iter()
                .map(|g| quote(g))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    Ok(format!("SELECT {} FROM input{group}", select.join(", ")))
}
pub(crate) fn numeric_type(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}
pub(crate) fn group_type(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Boolean | DataType::Utf8 | DataType::LargeUtf8
    ) || (numeric_type(dtype) && !matches!(dtype, DataType::Float32 | DataType::Float64))
}
