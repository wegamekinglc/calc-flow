#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum Coordinate {
    Ordered(u64),
    InputRow(u64),
}

impl Coordinate {
    pub(super) fn parse(raw: &str) -> Option<(Self, &str)> {
        if let Some(raw) = raw.strip_prefix("ordered position ") {
            return ordered(raw);
        }
        input_row(raw.strip_prefix("input row ")?)
    }

    pub(super) fn text(self) -> String {
        match self {
            Self::Ordered(position) => format!("ordered position {position}"),
            Self::InputRow(row) => format!("input row {row}; ordered position not assigned"),
        }
    }
}

fn ordered(raw: &str) -> Option<(Coordinate, &str)> {
    let (position, reason) = raw.split_once(": ")?;
    Some((Coordinate::Ordered(number(position)?), reason))
}
fn input_row(raw: &str) -> Option<(Coordinate, &str)> {
    let (position, reason) = raw.split_once("; ordered position not assigned: ")?;
    Some((Coordinate::InputRow(number(position)?), reason))
}
fn number(raw: &str) -> Option<u64> {
    let parsed = raw.parse::<u64>().ok()?;
    (parsed.to_string() == raw).then_some(parsed)
}
