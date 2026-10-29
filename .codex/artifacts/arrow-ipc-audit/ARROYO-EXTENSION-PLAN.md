# Arroyo 增量构建与相邻对照预记

接续 PR #511 `c4dc6b585097c442267f37f28effbb36850251c9`，复用原分支、
CF 源基线 b5ae227e、sealed native 和 Arroyo v0.15.0 / 4ef15f6 的构建缓存。
不重复原 CF 内部比较、benchmark 审计或历史采样。

本次必要增量构建累计硬预算 1,800 秒。每次记录 UTC 起止、monotonic 时长、命令、
配置、退出码及完整日志；后续构建仅使用剩余时间，不重新获得 30 分钟预算。
沿用已通过检查的 gnu17/CMake compatibility 配置。实际真实引擎可运行、输出正确后才
接受性能样本；仅 typecheck 或 build 通过不算结果证据。

采集范围仅原 transform-full / transform-512 两个转换场景。在任何新原生结果/性能
预检之前收敛为最小必要对照：CF streaming 与真实 Arroyo 各五次，按 AB/BA 交替，
共 20 次严格邻近观测；SQL/预编译的原 CF 内部对照不重跑，也不冒充新跨引擎配对。
只有该新批次的邻近样本可形成跨引擎成对比较。
相同 32,768 行、Arrow IPC bytes、CPU 4–7、controller 0–1、Tokio 4 threads、Arroyo -p1。
预检不混入正式样本，正式测量不因噪声追加。总采集/probe 预算仍为 600 秒，构建另计。

Arroyo 必须运行真实执行图，输入/结果 Arrow IPC，图间 Arrow；禁止 JSON 数据通路和
替代 DataFusion 算法。核心执行、绑定/类型适配、IPC 和启动分别记录，结果完整可消费后
停钟。对照同时验证输出列、整数值及顺序；额外 schema 元数据差异明确说明。
不同 transport 或 startup 边界不合并成一个性能排名。

窗口不自动加入矩阵：仅做必要开放源/WM/EOF 前输出语义 probe。若未能证明与 CF
原窗口轨迹等价，则保持未排名，交付实际缺口；不使用 file EOF 最终输出替代该证明。
所有增量证据补入同一 PR，不改 #494/#510 或原 CF 数据，不实施产品优化。
