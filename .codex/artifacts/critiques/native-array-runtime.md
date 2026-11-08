# Native array runtime 设计审查记录

本记录汇总 A3b 的设计结论；实现源码审查和 CI 是独立合并条件。

A3b bounded IR、共享预算和原生 batch factory 设计通过。factory 为其所有
operator 共享两个私有 worker slots。取消及 waiter 丢失时，独立 supervisor
持有 blocking worker，permit 和输入输出所有者保留到实际结束；长循环及
包含 null 的历史扫描必须观察取消。默认 Tokio Runtime Drop 负责排空 blocking
pool；调用者显式选择 shutdown_timeout/background 时，不保证返回前 worker
结束，但必须已请求合作取消且 closure 持有资源。不得为此引入额外 cleanup
线程或通用 runtime 框架。

capability schema 4 明确加入 options 定义端口及不可变 JSON schema 文本。
`native_provider` 保存真实注册身份，不借用 Python callback 或 stateless
stream 证明；描述查询不构成注册。失败注册不推进记录、revision 或缓存。

## 已解析字面量修订

专项审查通过 [A3 字面量边界修订](../api-notes/native-array-runtime.md#a3-已解析数值字面量边界)。
这是对原始 token 保证的明确更正：当前 JsonMap 和锁定的 serde_json 解析器
已丢失部分整数拼写，无法在保留完整有限 f64 范围的同时恢复它。必须公开
保留整数表示与已舍入 f64 的非单调原始拼写边界，并补充真实解析到执行 bits
的回归。不得声称 JSON Schema 单独表达该表示相关规则。

本修订不需更改数值算法、启用全局任意精度解析或增加依赖；Python 声明前端
仍在编码前限制 Python 整数。保留原生数组数据、f32 有限转换及 pow 界限。

## M2 与 A6 合同复核摘要

最终设计结论：**Pass**。本节归档已批准的
[API 合同](../api-notes/native-array-runtime.md)；设计通过不替代实现审查、
实际执行和 required CI。A6a 与 A6b 保持独立交付，不能提前声称状态恢复能力。

M2 的直接多 matmul attachment 不自动扩围；每次表桥接仍限一次 matmul，
多次运算通过显式 Table 阶段，独立数组收集保留每段一次的分段合同。
静态 digest-v1 含物理绑定名：共享 owner 可只冻结、规范编码和放置一次，
每个别名仍独立校验声明、计算并核对原协议摘要。每个消费者的真实准备预算、
有界编码 scratch、实际取消和完整候选发布必须落实，不能假设旧准备入口已
提供这些机制；不得改变旧 NumPy/JAX 的计费或恢复解释。

A6a 的直接 owned mean/EWMA 接口、数值合同和状态/峰值预算通过。
A6b 的真实 stateful lifecycle、capability schema 5、版本化 raw-IEEE codec
及 managed restart 合同，在补齐以下两项后通过：丢失 awaiter 后，未结束的
工作必须继续持有预算，由同 operator 的 pending-work gate 阻止或完整计费
后续候选、capture 和 restore；同步 reset/restore 不能取消未结束工作的计费。
terminal-complete manifest 的每个 native sequence 节点，也必须在返回成功
或打开 sink 前完成同样的有界 preload、checksum、codec、identity 和限额校验。
仅增加普通 restore_async 路径不能覆盖 terminal fast path。

A6a 不发布 sequence/checkpoint capability。A6b 的工厂、codec、真实恢复、
能力版本迁移及对应测试须一起验收；所有阶段继续禁止 TA-Lib 代码依赖。

## M2 native metadata budget boundary review

最终结论：**Pass for the explicit boundary clarification**。这是对
[M2 native preparation budget 边界](../api-notes/native-array-runtime.md#m2-native-preparation-budget-boundary-clarification)
的定点设计审查，不是当前源码、copy multipliers、整个 M2 功能、实际测试/CI 或
合并就绪批准。既有 API/A6 审查历史保留。

来源为设计输入 `m2-chain-attachment-api.md` §3，完整文件 SHA-256：

```text
675302e98bdf69f32ca12f0eefa0ae3b42449d52b9497673924fb348fed927d0
```

最终审查输入 `m2-manifest-budget-critic.md` 的 SHA-256：

```text
28e852b3fb1d3f47f0d1af70e0a5160e9b25c04a8d0294aef1e46e8650268a0e
```

接受的边界：D 继续包含独立 native alias/version/SHA/descriptor、Run/Job
settings、checkpoint identity、live selected/current manifest 和 native JSON-map
字符串。真实共享 Arc allocation 只计一次，同值独立 clone 分别计费。通用整个
canonical String/Vec、global I/O envelope、历史 retention 维护 Vec 是明确排除项，
不虚构 native 字段占其容量的比例；不能把 manifest 内独立 native 字段也排除。
现有 allocator/map-node/外层 settings frame 排除保持，保证范围是指定 native
storage/metadata，不是 RSS、all checkpoint memory 或 all runtime live bytes。

phase max 只在实际所有权证明 non-overlap 时成立，包括 worker/cancel/staging、
SQL 临时对象释放与并发 epochs；否则覆盖实际重叠或经证明的保守组合。resident
context inventory 和完整 copy-site 数量仍需最终源码审核，此 Pass 不认证 evolving
公式/常量。fresh job 的一个 possible selected slot 可显式保守预留。

F/P 的实际容量必须独立核实；未来 D 不是掩盖 frozen/source/placement 超额的
可借用 credit。拒绝超额或连同完整仍需要的 D 与 W 对每个 consumer 重新准入。
R_j+W<=B_j 和后续 numeric B_j-R_j 边界保持，宽预算不补贴窄预算，不加公共 knob、
协议版本、digest 变化或旧 NumPy/JAX 预算重解释。

源边界包括 `state/manifest.rs::canonical_bytes/recompute_state_checksum` 的 native
JSON 字符串与 whole-document buffer 分离，以及
`state/transaction.rs::retain_locked/load_retention_manifests` 的历史维护 Vec。
最终实现审查须证明实际 clone sites、容量与重叠，不以重复生产常量的算术测试
代替证据；应覆盖 fresh/recovered、selected+current+JSON copies、取消/late drop、
紧消费者预算和未来 D 尚未使用时的 F/P 超额。保留无 publication/source-open、
per-alias digest golden、非 native 行为与 bounded worker settlement 验收。

本归档不修改产品，不运行构建/产品测试/benchmark，不操作 remote；没有未决
边界设计问题，但上述源码和行为证明责任仍属于对应 M2 实现阶段。
