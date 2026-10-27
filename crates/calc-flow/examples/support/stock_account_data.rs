//! Lazy Rust equivalents of `examples/stock_account_data.py` and quote fixtures.
#![allow(dead_code)]
use crate::support::{self, ExampleResult, floats, ints, strings, times};
use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, Cursor, NativeWatermarkCapability, ReplayPositioning, Result,
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, StreamSink,
    StreamSource,
};
use datafusion::arrow::{
    array::{Array, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use std::{collections::BTreeMap, sync::Arc};
use tokio::sync::Mutex;

#[derive(Clone, Copy)]
pub struct Options {
    pub rows: u64,
    pub accounts: u32,
    pub stocks: u32,
    pub batch_size: u32,
    pub quote_rows: u64,
}
impl Options {
    pub fn parse(valuation: bool) -> ExampleResult<Self> {
        let mut options = Self {
            rows: 10_000_000,
            accounts: 100,
            stocks: 100,
            batch_size: 1_000,
            quote_rows: 10_000,
        };
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args == ["--help"] {
            println!(
                "--rows N --accounts N --stocks N --batch-size N{}",
                if valuation { " --quote-rows N" } else { "" }
            );
            std::process::exit(0);
        }
        if !args.len().is_multiple_of(2) {
            return Err("options require integer values; use --help".into());
        }
        for pair in args.chunks_exact(2) {
            match pair[0].as_str() {
                "--rows" => options.rows = pair[1].parse()?,
                "--accounts" => options.accounts = pair[1].parse()?,
                "--stocks" => options.stocks = pair[1].parse()?,
                "--batch-size" => options.batch_size = pair[1].parse()?,
                "--quote-rows" if valuation => options.quote_rows = pair[1].parse()?,
                _ => return Err(format!("unknown option {}", pair[0]).into()),
            }
        }
        if options.accounts == 0 || options.stocks == 0 || options.batch_size == 0 {
            return Err("accounts, stocks and batch size must be positive".into());
        }
        if valuation && options.quote_rows < u64::from(options.stocks) {
            return Err("quote rows must cover every stock".into());
        }
        Ok(options)
    }
    fn pairs(self) -> u64 {
        u64::from(self.accounts) * u64::from(self.stocks)
    }
}

pub fn trade_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("account", DataType::Utf8, true),
        Field::new("stock", DataType::Utf8, true),
        Field::new("quantity", DataType::Int64, true),
        Field::new("side", DataType::Utf8, true),
        Field::new("price", DataType::Float64, true),
        Field::new("flow_type", DataType::Utf8, true),
    ]))
}
pub fn quote_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("stock", DataType::Utf8, true),
        Field::new("latest_price", DataType::Float64, true),
        Field::new(
            "timestamp",
            DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            true,
        ),
    ]))
}

pub struct DemoSource {
    options: Options,
    offset: u64,
    quotes: bool,
}
impl DemoSource {
    pub fn trades(options: Options) -> Self {
        Self {
            options,
            offset: 0,
            quotes: false,
        }
    }
    pub fn quotes(options: Options) -> Self {
        Self {
            options,
            offset: 0,
            quotes: true,
        }
    }
    fn trade_batch(&self, end: u64) -> Result<Batch> {
        let pairs = self.options.pairs();
        let rows: Vec<_> = (self.offset..end)
            .map(|i| {
                let seed = i < pairs;
                let index = if seed { i } else { i - pairs };
                let pair = index % pairs;
                let cycle = (index / pairs) % 10;
                let account = format!("account-{:03}", pair / u64::from(self.options.stocks));
                let stock = format!("stock-{:03}", pair % u64::from(self.options.stocks));
                let buy = cycle.is_multiple_of(2);
                let flow = (pair + cycle) % 10;
                (
                    account,
                    stock,
                    if seed {
                        0
                    } else if buy {
                        200
                    } else {
                        100
                    },
                    if buy { "buy" } else { "sell" },
                    if seed || flow < 8 {
                        "trade"
                    } else if flow == 8 {
                        "order"
                    } else {
                        "cancel"
                    },
                )
            })
            .collect();
        let record = RecordBatch::try_new(
            trade_schema(),
            vec![
                strings(&rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>()),
                strings(&rows.iter().map(|r| r.1.as_str()).collect::<Vec<_>>()),
                ints(&rows.iter().map(|r| r.2).collect::<Vec<_>>()),
                strings(&rows.iter().map(|r| r.3).collect::<Vec<_>>()),
                floats(&vec![10.0; rows.len()]),
                strings(&rows.iter().map(|r| r.4).collect::<Vec<_>>()),
            ],
        )
        .map_err(|e| support::invalid("trades", e.to_string()))?;
        Batch::table(vec![record], BatchMetadata::default())
    }
    fn quote_batch(&self, end: u64) -> Result<Batch> {
        let rounds = self.options.quote_rows / u64::from(self.options.stocks);
        let rows: Vec<_> = (self.offset..end)
            .map(|i| {
                let cycle = i / u64::from(self.options.stocks);
                let stock = u32::try_from(i % u64::from(self.options.stocks)).unwrap();
                let cycle = if rounds >= 2 && cycle == rounds - 2 {
                    rounds - 1
                } else if rounds >= 2 && cycle == rounds - 1 {
                    rounds - 2
                } else {
                    cycle
                };
                Ok((
                    format!("stock-{stock:03}"),
                    10.0 + f64::from(u32::try_from(cycle)?) * 0.5 + f64::from(stock) * 0.25,
                    i64::try_from(cycle + 1)? * 1_000_000,
                ))
            })
            .collect::<ExampleResult<Vec<_>>>()
            .map_err(|e| support::invalid("quotes", e.to_string()))?;
        let record = RecordBatch::try_new(
            quote_schema(),
            vec![
                strings(&rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>()),
                floats(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
                times(&rows.iter().map(|r| r.2).collect::<Vec<_>>()),
            ],
        )
        .map_err(|e| support::invalid("quotes", e.to_string()))?;
        Batch::table(vec![record], BatchMetadata::default())
    }
}
#[async_trait]
impl StreamSource for DemoSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossy,
            max_batch_rows: usize::try_from(self.options.batch_size).unwrap(),
            max_batch_bytes: 64 << 20,
            schema: SourceSchema::Exact(if self.quotes {
                quote_schema()
            } else {
                trade_schema()
            }),
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _: Option<Cursor>) -> Result<()> {
        self.offset = 0;
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let total = if self.quotes {
            self.options.quote_rows
        } else {
            self.options
                .pairs()
                .checked_add(self.options.rows)
                .ok_or_else(|| support::invalid("rows", "row count overflow"))?
        };
        if self.offset == total {
            return Ok(None);
        }
        let boundary = if !self.quotes && self.offset < self.options.pairs() {
            self.options.pairs()
        } else {
            total
        };
        let end = self
            .offset
            .saturating_add(u64::from(self.options.batch_size))
            .min(boundary);
        let batch = if self.quotes {
            self.quote_batch(end)?
        } else {
            self.trade_batch(end)?
        };
        self.offset = end;
        Ok(Some(SourceEvent::Data {
            batch,
            cursor: Cursor::unbound(end.to_be_bytes().to_vec(), BTreeMap::new())?,
        }))
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

pub type Latest = Arc<Mutex<Option<Batch>>>;
pub struct LastSink(pub Latest);
#[async_trait]
impl StreamSink for LastSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        *self.0.lock().await = Some(batch.clone());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Independent arithmetic reference, bounded by account/stock keys, not trade rows.
pub fn expected_positions(options: Options) -> ExampleResult<BTreeMap<(String, String), i64>> {
    let pairs = options.pairs();
    let period = pairs
        .checked_mul(10)
        .ok_or("too many account/stock pairs")?;
    (0..pairs)
        .map(|pair| {
            let mut position = 0_i64;
            for cycle in 0..10 {
                if (pair + cycle) % 10 >= 8 {
                    continue;
                }
                let index = cycle * pairs + pair;
                let count = options.rows / period + u64::from(index < options.rows % period);
                position += i64::try_from(count)?
                    .checked_mul(if cycle.is_multiple_of(2) { 200 } else { -100 })
                    .ok_or("position overflow")?;
            }
            Ok((
                (
                    format!("account-{:03}", pair / u64::from(options.stocks)),
                    format!("stock-{:03}", pair % u64::from(options.stocks)),
                ),
                position,
            ))
        })
        .collect()
}
pub fn check_positions(batch: &Batch, options: Options) -> ExampleResult {
    let mut actual = BTreeMap::new();
    for record in batch.table_payload()?.batches() {
        let accounts = record
            .column_by_name("account")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let stocks = record
            .column_by_name("stock")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let positions = record
            .column_by_name("position")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..record.num_rows() {
            actual.insert(
                (accounts.value(row).to_owned(), stocks.value(row).to_owned()),
                positions.value(row),
            );
        }
    }
    assert_eq!(actual, expected_positions(options)?);
    Ok(())
}
pub fn check_valuation(batch: &Batch, options: Options) -> ExampleResult {
    let mut expected = BTreeMap::<String, f64>::new();
    for ((account, stock), position) in expected_positions(options)? {
        let stock: u32 = stock.strip_prefix("stock-").unwrap().parse()?;
        let last_cycle = (options.quote_rows - 1 - u64::from(stock)) / u64::from(options.stocks);
        let price = 10.0 + f64::from(u32::try_from(last_cycle)?) * 0.5 + f64::from(stock) * 0.25;
        // Match SQL's Int64-to-Float64 cast, including its precision boundary.
        #[allow(clippy::cast_precision_loss)]
        let position = position as f64;
        *expected.entry(account).or_default() += position * price;
    }
    let mut actual = BTreeMap::new();
    for record in batch.table_payload()?.batches() {
        let accounts = record
            .column_by_name("account")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let values = record
            .column_by_name("market_value")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for row in 0..record.num_rows() {
            assert!(!values.is_null(row));
            actual.insert(accounts.value(row).to_owned(), values.value(row));
        }
    }
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>()
    );
    for (account, value) in actual {
        assert!((value - expected[&account]).abs() < 1e-8 * value.abs().max(1.0));
    }
    Ok(())
}
