# SMA20 千万行公开作业的阶段诊断协议

只测定时用例 `engines/10000000/calc-flow-stream-1t/sma20`。历史定时任务的公开 p50 为 408.74 → 560.29 ms，两次配对确认回退。此诊断可回答同一源码版本在哪类滚动回调、哪个已插桩阶段花费更多时间。它使用正常的生产 Python wheel 和 Rust 内置指标。

## 版本与入口

`protocol.json` 在运行前固定两端 SHA、整棵树、独立制品、标准 harness 提交和样本数。11.14 是 `7012b5b4`；候选对照点是定时任务当时的 main `72072c1d`。本机候选 wheel 实际从 `89cc36a8` 构建，两者整棵 Git 树均为 `cec8e19a`，source digest 和该版本的 Cargo.lock 相同。比较时保持这两个 SHA 的身份。两者都有独立 site、同一 CPython 3.13.9、Rust 1.94.0 和 cp313 abi3 release/locked 编译参数。历史定时 runner 使用 CPython 3.13.18 及其自己的制品；本机结果另列。

每个轮次由原维护的 `measure_case` 和 `Worker` JSON 请求调度。子进程仍调用维护的 `worker.main`、`EngineCase`、公开 `StreamingRunner` 与 Arrow sink。专用启动器在内层时钟结束后保存作业引用与第一份状态，在 `sample` 或 `prepare` 完成并通过 oracle 后读最终状态。子进程验证实际导入位置，维护的 hello 验证加载的 native 哈希和双方机器、依赖与线程。控制端还核对两个 wheel 的哈希、wheel 内每份 Python 包文件和已安装的对应文件。

## 触发与边界

- 每次 sample 创建空状态。按 64 个 symbol 轮转输入一千万行，时间、序号及 Float64 价格确定。行窗宽度和最少有效期数都为 20。fixture 以 64,000 行分 Arrow chunks，标准输入将其合成十个 1,000,000 行数据批，各自随后送一个明确水位。Data 校验并缓冲，watermark 可释放已成序输入并产生更新和输出，EOF 可结算剩余状态。三个 callback 类别分列，用计数器验证实际发生在哪一类。
- 公共计时从源 ready 后、首次投递前一直到 sink 收齐一千万行并完成 Arrow 表拼接。建图、源启动、EOF、终止检查点、oracle、两次状态序列化均在此时钟外。生产作业有 ManagedCheckpointRuntime 和 24 小时间隔，第一份观察验证此时没有活动或已完成 epoch。EOF 后的 terminal 检查点保留。
- `before_eof` 在公共时钟停止且推送 EOF 之前取得。最后几行已交付时，某回调可能刚发出结果但尚未登记成功；各类只发布已完成的阶段，`started` 可高于 `succeeded`。该观察仅作边界核对。用于阶段比较的 `terminal` 要求自然结束、每个回调都成功结算、无溢出或节点错误、边已排空、输入和 sink 行数吻合。
- callback 的十个独占阶段按原代码对应输入检查、排序证明、实体路由、状态准备、数值更新、历史维护、Arrow 输出、预算准备、下游发送等待、其余时间。每类累积阶段之和必须等于其 `callback_duration_ns`。公开总时钟与其中的计算、发送、sink 等过程重叠；源、其他算子、sink、edge 时钟也可能并发。不能把这些数相加或把差额解释为 Python 开销。terminal 的 `end` 在公开时钟外，最后的 watermark 尾部也可能延伸到时钟外。
- 计数另读 `resolved_rows`、`numeric_rows`、`numeric_serial_calls`、`numeric_general_calls`、`numeric_parallel_calls`、各类 fallback、复制与物化。按类和每轮统计；不要假设 1 个 Tokio worker 等于 1 个原生数值 lane。main 的 `fill_slab_rows` 把 full-tick 检查、按行 fallback、数值更新和结果追加包在 NumericUpdate 内，两种串行分支都记一次 `numeric_serial_calls`。现有指标不能把它们分开；各阶段变化只支持进一步定位。本对照点的 route_slab 会累计 resolved_rows、touched_entities 并返回内部 entity_encode_ns，但没有建立 RollingStage::EntityResolution 的作用域；该公开阶段为零仍可能执行了路由，其耗时可落在 other。slab 的内部状态表与作业级日志分别控制。

## 正确性与预算

标准 oracle 以 NumPy 对 64 个 symbol 分别作 20 行卷积并采用 TA-Lib 的 full-window 暖启动约定。每个实体前 19 行缺值，共 1,216 个，有限输出为 9,998,784。maintained validator 将所有行按 sequence 排序，验证非 value 列的每个元素，并以 `rtol=atol=1e-10`、`equal_nan=True` 比对 value。它允许 Arrow Null 与转入 NumPy 后的 NaN 以相同缺值呈现；它不检测原始结果顺序。保留每次正确性返回的最大绝对误差。

只跑一项、两轮，双方每轮各预热一次并排除。每轮 10 对按 AB/BA 交替的正式样本，各 20 个正式观测，合计 40 个正式和 4 个预热。两臂均为 Tokio=1，保留 catalog 的 Polars=4 和 BLAS=1。本地主机使用维护的稀疏 changelog 资源守卫，预检三秒后父子进程绑定同一个物理核心的一枚逻辑 CPU，逐秒记录 SMT 同胞繁忙程度、内存及分页。原定时任务允许四枚逻辑 CPU；不可把本地主机的绝对延迟混入原分片。

仅复用成品，不安排新构建。单次 240 秒总预算涵盖资格检查、启动、fixture、预热、全部 oracle、采样、终止与清理；控制器内部 210 秒后取消，外层 235 秒发 TERM，最后五秒用于退出。出错或超时即保留部分记录并停；不更换样本、重测直到显著或用一轮结果分类。公开总时钟沿用仓库两轮配对中位数区间与 +5% 回退判据。阶段表给出按 callback 的两端中位数与每轮配对绝对毫秒差的中位数/均值，仅用于定位，避免除以零的阶段相对百分比与从中位数差分摊总体差额。

## 重放

Linux 上使用下面两个现有目录。只应在 CPU 测量窗口由协调者启动一次。输出目录须为全新路径；全局 timeout 结束时检查无此目录所属的 worker 仍在运行。原维护的 controller 在正常取消时先关闭和回收两个子进程。

```sh
suite=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/managed-runtime-20261124/round4-harness
diag=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/sma20-phase-diagnosis-nov25
interp=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/worktrees/benchmark-large-engines-20261114/target/benchmark-venv/bin/python
PYTHONPATH="$suite" PYTHONDONTWRITEBYTECODE=1 timeout -k 5s 235s "$interp" "$diag/run.py" "$diag/observed"
```

`qualification.json` 保存实际环境、协议和脚本哈希；`resources.jsonl` 保存资源条件；`measurement/round-*/` 中的每个 prepare/sample 请求包含真实公开耗时、oracle 和两份状态；`samples.json` 保存二十对完整样本；`summary.json` 给出公开判定及阶段描述。若执行提前停止，用 `failure.json`、部分请求和进程返回状态判读，不能据此推出回退归因。
