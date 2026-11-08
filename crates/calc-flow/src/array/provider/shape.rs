use super::super::invalid;
use crate::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Shape {
    Scalar,
    Vector(Option<usize>),
    Matrix(Option<usize>, Option<usize>),
}
impl Shape {
    pub(super) fn from_declaration(shape: &[Option<usize>]) -> Self {
        match shape {
            [] => Self::Scalar,
            [len] => Self::Vector(*len),
            [rows, columns] => Self::Matrix(*rows, *columns),
            _ => unreachable!("declaration rank was validated"),
        }
    }
    pub(super) fn known(shape: &[usize]) -> Result<Self> {
        let shape = match shape {
            [] => Self::Scalar,
            [len] => Self::Vector(Some(*len)),
            [rows, columns] => Self::Matrix(Some(*rows), Some(*columns)),
            _ => {
                return Err(invalid(
                    "options.nodes.shape",
                    "only rank zero through two is supported",
                ));
            }
        };
        shape.elements()?;
        Ok(shape)
    }
    pub(super) fn dimensions(self) -> Vec<Option<usize>> {
        match self {
            Self::Scalar => vec![],
            Self::Vector(len) => vec![len],
            Self::Matrix(rows, columns) => vec![rows, columns],
        }
    }
    pub(super) fn concrete(self) -> Result<Vec<usize>> {
        self.dimensions()
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("nodes.shape", "runtime extents must be concrete"))
    }
    pub(super) fn elements(self) -> Result<Option<usize>> {
        match self {
            Self::Scalar => Ok(Some(1)),
            Self::Vector(len) => Ok(len),
            Self::Matrix(Some(0), _) | Self::Matrix(_, Some(0)) => Ok(Some(0)),
            Self::Matrix(Some(rows), Some(columns)) => rows
                .checked_mul(columns)
                .map(Some)
                .ok_or_else(|| invalid("options.nodes.shape", "shape product overflowed usize")),
            Self::Matrix(..) => Ok(None),
        }
    }
    pub(super) fn len(self) -> Result<usize> {
        self.elements()?
            .ok_or_else(|| invalid("nodes.shape", "runtime extents must be concrete"))
    }
    pub(super) const fn rank(self) -> usize {
        match self {
            Self::Scalar => 0,
            Self::Vector(_) => 1,
            Self::Matrix(..) => 2,
        }
    }
    pub(super) fn broadcast(self, other: Self) -> Result<Self> {
        match (self, other) {
            (Self::Scalar, other) | (other, Self::Scalar) => Ok(other),
            (Self::Vector(left), Self::Vector(right)) => dimension(left, right).map(Self::Vector),
            (Self::Matrix(a, b), Self::Matrix(c, d)) => matrix(a, b, c, d),
            _ => Err(invalid(
                "options.nodes.shape",
                "non-scalar operand ranks must agree",
            )),
        }
    }
    pub(super) const fn transpose(self) -> Self {
        match self {
            Self::Matrix(rows, columns) => Self::Matrix(columns, rows),
            other => other,
        }
    }
    pub(super) fn reshape(self, target: &[usize]) -> Result<Self> {
        let target = Self::known(target)?;
        let target_count = target.len()?;
        if self.elements()?.is_some_and(|count| count != target_count) {
            return Err(invalid(
                "options.nodes.shape",
                "reshape must retain the element count",
            ));
        }
        Ok(target)
    }
    pub(super) fn matmul(self, right: Self) -> Result<Self> {
        let (Self::Matrix(rows, inner), Self::Matrix(other, columns)) = (self, right) else {
            return Err(invalid(
                "options.nodes.shape",
                "matmul requires two rank-two operands",
            ));
        };
        dimension(inner, other)?;
        if columns == Some(0) {
            return Err(invalid(
                "options.nodes.shape",
                "matmul requires positive output width",
            ));
        }
        Ok(Self::Matrix(rows, columns))
    }
    pub(super) fn reduced(self, axis: Option<usize>, keepdims: bool) -> Result<Self> {
        let Some(axis) = axis else {
            return Ok(self.full_reduction(keepdims));
        };
        if axis >= self.rank() {
            return Err(invalid(
                "options.nodes.axis",
                "axis must be smaller than input rank",
            ));
        }
        Ok(self.axis_reduction(axis, keepdims))
    }
    fn full_reduction(self, keepdims: bool) -> Self {
        if !keepdims {
            return Self::Scalar;
        }
        match self {
            Self::Scalar => Self::Scalar,
            Self::Vector(_) => Self::Vector(Some(1)),
            Self::Matrix(..) => Self::Matrix(Some(1), Some(1)),
        }
    }
    fn axis_reduction(self, axis: usize, keepdims: bool) -> Self {
        match (self, axis, keepdims) {
            (Self::Vector(_), 0, false) => Self::Scalar,
            (Self::Vector(_), 0, true) => Self::Vector(Some(1)),
            (Self::Matrix(_, columns), 0, false) => Self::Vector(columns),
            (Self::Matrix(_, columns), 0, true) => Self::Matrix(Some(1), columns),
            (Self::Matrix(rows, _), 1, false) => Self::Vector(rows),
            (Self::Matrix(rows, _), 1, true) => Self::Matrix(rows, Some(1)),
            _ => unreachable!("reduction axis was validated"),
        }
    }
    pub(super) fn lane_width(self, axis: Option<usize>) -> Result<Option<usize>> {
        match (self, axis) {
            (Self::Matrix(rows, _), Some(0)) => Ok(rows),
            (Self::Matrix(_, columns), Some(1)) => Ok(columns),
            _ => self.elements(),
        }
    }
}
fn dimension(left: Option<usize>, right: Option<usize>) -> Result<Option<usize>> {
    if matches!((left,right),(Some(left),Some(right)) if left!=right) {
        return Err(invalid("options.nodes.shape", "operand extents must agree"));
    }
    Ok(left.or(right))
}
fn matrix(a: Option<usize>, b: Option<usize>, c: Option<usize>, d: Option<usize>) -> Result<Shape> {
    Ok(Shape::Matrix(dimension(a, c)?, dimension(b, d)?))
}
