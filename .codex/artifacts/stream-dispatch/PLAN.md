# DAL-417 第二阶段：streaming 回调调度

基线固定 main `1b8ff9cf360b5c05d62cf97e48d12d19f652c17e`，后续专用分支交付。
先诊断 source.next / sink.write 的 GIL 接入、排队、分发及回调完成，区分重叠阶段；
仅消除每批重复的桥接准备或通知工作，不改变 Arrow 包装、批形状、缓冲容量和控制流。
不并入 DAL-475 或第三阶段。

只比较既有 transform-full（1×32768）和 transform-512（64×512）的 CF streaming。
沿用 Arrow fixture、严格列/值/顺序 oracle、已备输入 gate 到父进程 read_all 完整消费，
数据通路全部 Arrow/Batch/IPC。每个新进程暖身首批一次后放行正式输入，暖身另外记录。
输出 IPC 和完整 Arrow 转换在计时内；声明、编译、输入 IPC、首批暖身、关闭单列。
固定两轮、每场景每轮十对相邻 AB/BA（偶数 base/head，奇数 head/base），80 次正式观测。
另固定四次预检、基线/候选各两场景一次诊断，不纳入正式排名，不为结果补样。
使用仓库 paired_round / _paired_verdict 的两轮 +5% 规则；不相加跨任务重叠时钟。

累计采集预算 600 秒，含诊断、fixture、预检、暖身、进程启动/退出、正确性与归档审计。
每次调用 30 秒上限，保留至少 35 秒清理；构建另列，若需 native 重建预估 10–15 分钟。
复用旧 wheel 前核对 Rust/Python 产品文件、默认依赖图及 native hash；不改历史测量身份。
worker CPU 4–7、controller CPU 0–1、Tokio 4，其余计算线程 1，无独占/锁频保证。
正式采集持续检测 cargo/rustc/maturin 争用，发现即停止，不扩大或重新挑选有利样本。

回归只覆盖直接影响的调度和通知边界：逐批/顺序、不可变输入、慢 sink 背压、回调错误、
取消及资源释放、既有水印/checkpoint/EOF 定向证据。全量、跨平台和覆盖率交 CI。

定位阶段记录：四个诊断进程实际均使用兼容基线 wheel（每场景两个），文件名中的
head 是定位时的重复基线，并非随后构建的候选；总诊断次数仍为四，不追加定位采样。
正式统计只读取独立 results 目录和最终 base/head seals。
