"""Final DISTINCT with owned peak accounting; borrowed ingress backing is excluded."""

import pyarrow as pa

import calc_flow as cf

events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
out = events.distinct(
    ["x"], spec=cf.DistinctSpec.bounded_source(cf.StateBudget(5, 8 << 20))
)
program = cf.Program("unique", engine="sql", inputs=[events], outputs={"out": out})
result = program.collect(
    {"events": pa.table({"x": pa.array([1, None, 1, None, 2], pa.int64())})}
)
if set(result["out"]["x"].to_pylist()) != {None, 1, 2}:
    raise RuntimeError("unexpected DISTINCT tuple set")
print(result["out"])
