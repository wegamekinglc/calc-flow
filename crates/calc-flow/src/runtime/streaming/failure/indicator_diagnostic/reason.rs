use crate::RollingOutputSpec;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum Reason {
    NonFinite,
    Order(String),
    Arithmetic(&'static str),
    CounterOverflow,
}

const ARITHMETIC: [&str; 7] = [
    "RSI price difference from previous price",
    "RSI/ATR seed/recurrence difference",
    "RSI/ATR seed/recurrence increment",
    "RSI/ATR seed/recurrence mean",
    "ATR high - low",
    "ATR high - previous_close",
    "ATR low - previous_close",
];

impl Reason {
    pub(super) fn field(output: &RollingOutputSpec, field: &str, text: &str) -> Option<Self> {
        if text == "RSI/ATR requires finite non-missing samples" {
            return Some(Self::NonFinite);
        }
        if let RollingOutputSpec::Atr { high, low, .. } = output
            && field == high
            && text == format!("must be greater than or equal to {low}")
        {
            return Some(Self::Order(low.clone()));
        }
        let operation = text.strip_suffix(" is not representable as finite float64")?;
        ARITHMETIC
            .into_iter()
            .find(|actual| *actual == operation)
            .map(Self::Arithmetic)
    }
    pub(super) fn counter(text: &str) -> Option<Self> {
        (text == "processed position count would exceed u64::MAX").then_some(Self::CounterOverflow)
    }
    pub(super) fn text(&self) -> String {
        match self {
            Self::NonFinite => "RSI/ATR requires finite non-missing samples".into(),
            Self::Order(low) => format!("must be greater than or equal to {low:?}"),
            Self::Arithmetic(operation) => {
                format!("{operation} is not representable as finite float64")
            }
            Self::CounterOverflow => "processed position count would exceed u64::MAX".into(),
        }
    }
}
