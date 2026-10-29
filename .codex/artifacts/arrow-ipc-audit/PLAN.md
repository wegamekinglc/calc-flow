# DAL-417 Arrow/IPC 重测与 benchmark 审计预记

启动时固定 main `b5ae227e48ac9bb8a81583a23485e2c6064cee52`；Arroyo v0.15.0
`4ef15f6336a596651f10cf27c1ee29c29bf81577`，不改写 #494/#510 的证据。
calc-flow 产品源码、lock/toolchain/build 配置与上轮 b9a10d4 完全相同；核对后复用
release native `a0708fcf5fa84a1cf2c5347a66b73680d7426a3313746a1ef99519d5b2c7bfb8`。
构建/接口探测单列；Arroyo 必须执行真实引擎，不以 DataFusion 单独执行冒充它。

固定四个场景：32,768 行整表转换、64×512 小批转换、整表四秒 tumbling 窗口、
64×512 小批同窗口。转换为 `id >= 0 AND k < 8`，输出 `id, x*2+1`；窗口为
16 key 的整数 SUM/COUNT，沿用原 event-time 值和数据。窗口追加过滤 marker 及显式
watermark 以关闭结果，native job 不靠 EOF 产出。SQL 原生窗口不支持则保留拒绝证据，
有限 GROUP BY 仅作最终值语义 probe，不混入持续窗口排名。

转换固定 CF streaming、SQL collect_async、SQL 预编译 plan 三路径；窗口固定
CF streaming。每路径五次 fresh-process 观测，按 ABC/CBA 或单路径顺序采集，共
40 次正式观测。若 Arroyo 原生 Arrow/最小 connector 适配可用，预留四场景各五次；
若不可用，留下实际接口/构建失败，不补替代算法或 JSON 数据通路。另预留最多四次
独立诊断和两个最小 harness 审计 probe。正式采样前可做一次全路径预检，失败不混入正式
统计，不增加有利样本或规模。

测量总预算 600 秒（包括预检、采集、正确性、诊断与最小计时 probe），collector 限
500 秒，剩余用于 probe/清理。源码阅读、脚本编写与构建不计入测量预算。Arroyo 最小
适配构建最多 900 秒，失败则不宣称取得其核心速度数据；兼容 CF 构建不重建。

控制/观察进程固定 CPU 0–1；worker CPU 4–7，Tokio 4 threads，BLAS/Arrow helpers
单线程，DF target_partitions=1，Arroyo -p 1。同机不承诺 CPU 独占或锁频；记录主机、
依赖、输入/输出 SHA、native seal、命令与进程树 RSS。

所有数据输入、图间值及结果为 Arrow/Batch；进程间仅 Arrow IPC stream。
JSON 只准保存配置、计划与性能元数据，禁止作为数据结果的测试出口。
fixture 生成与输入 IPC 编码单列并在 round-trip 计时外；worker 的声明/编译、job
启动与 ready gate 单列，在父进程发送输入前完成。输入 IPC 解码、输入绑定、核心
执行、输出 IPC 写出、父进程 IPC 消费分别记录，核心时间从预备输入放行/首有限调用到
完整 Arrow 结果可消费（包括输出 Batch→Arrow/拼接）。SQL collect 的正常 convenience
准备保留在核心 API 时间，预编译对照把 native 图编译移到计时外；不能把它等同于
DataFusion 物理计划缓存。结果全表 IPC 消费完成才停止 round-trip 时钟；校验其后执行。
IPC 阶段可能重叠，不把 round-trip−core 称为独占传输时间。检查点关闭，不测恢复。

审计维护中的 catalog/pytest/Rust/lifecycle/standalone 入口、harness、计时器、汇总与 CI。
逐组区分 fixture、序列化、启动、声明/物理规划、消费和校验；静态不清时只做最小 probe。
不跑全套，不修改门禁或基线。更新优化优先级只针对 Arrow 证据，不把外部 JSON 适配
成本当核心产品热点。所有结果仅作描述性比较，不冒用版本回归判定。
