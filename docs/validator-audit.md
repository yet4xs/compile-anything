# Validator Invariant Audit — Phase 2

审计对象：`src/validator/validator.py`（对应 `spec/taskir-spec.md` §6）。
方法：逐条不变量人工审计 + `scripts/fuzz_taskir.py`（10000 随机程序，
builder 合法构造 ⇒ validator 必须接受；10000/10000 通过，无一误拒）。

## V1–V6 覆盖与缺陷

| invariant | 覆盖 | 缺陷 / 备注 |
|---|---|---|
| V1 结构 | id 格式/唯一性、必填字段、params/guard/retry 形状 | 未知顶层字段目前静默容忍（spec §7 说应警告）——低优先级 |
| V2 def-use | 数据/after/guard/retry.on 全部检查；数据引用强制先定义后使用（规范序） | 无已知漏洞。`retry.on` 允许引用**后面**的 VERIFY，这是设计（verify 必然下游）|
| V3 DAG | data+after+guard 边 DFS 找环 | **在 V2 严格排序下不可达**（环要求 A 先于 B 且 B 先于 A）。保留为防御层：将来允许乱序输入时激活 |
| V4 类型 | 输入/签名兼容、显式 output_type 收窄、guard.cond=Bool、SELECT 分支兼容 | 刻意的精确匹配（无 coercion lattice）：`Int` 不能进 `Float` 槽。fuzzer builder 因此踩过一次——是特性不是 bug，但训练数据生成时要小心 |
| V5 skill 可用性 | op ∈ registry ∪ {VERIFY, SELECT} | `params` 完全开放无 schema（如 FILTER.predicate 只是字符串）——v0.1 已知限制，参数语义校验属于 skill 级 schema 工作 |
| V6 控制流 | arity、retry.on 指向 VERIFY 且传递依赖本节点、max_attempts≥1、output 已定义 | **本轮新增** `GUARDED_OUTPUT` 警告：输出节点带 guard ⇒ 程序可能"完成但无值"，建议 SELECT 兜底 |

## Missing invariant candidates

### M1: Side-effect ordering（**真实缺口，最重要**）

`SEND` / `EXEC_ACTION` / `SAVE` 输出 `Any`。两个无数据依赖的 action 节点，
IR **无法表达**它们之间需要的顺序；一个并行 scheduler 可以合法重排它们
（"先扣款再发确认邮件"无法编码）。v0.1 模拟器按依赖序确定性执行所以暴露
不了，但这是 IR 表达力缺口，不是 runtime 缺口。

**建议的设计修改（v0.2，不建议现在 patch）**：引入 effect token，类比 LLVM
memorySSA / MLIR side-effect modeling：

- SkillSpec 增加 `effect_class: None | "memory" | "network" | "fs"`；
- 同类 effect 的 action 节点隐式串成一条 effect chain（每个 action 隐式依赖
  前一个同类 action），validator 检查链完整性；
- optimizer 做并行化时把 effect chain 作为不可跨越的屏障。

### M2: Resource legality

`hints.resource_class` 目前不校验（hints 按规范是 advisory）。例如
`GENERATE` 挂 `resource_class=api` 不会被拒。**结论**：v0.1 可接受——hints
非语义；当 scheduler 落地时必须规定 **ISA 的 resource_class 是权威，hints
只能收窄不能改写**，并加 validator 警告（hints 与 ISA 冲突时）。

### M3: Cost consistency

registry 是唯一权威成本来源；simulator 只对 latency 施加声明的 ±10%
seeded jitter，tokens/flops/energy 原样采用；报告聚合的是事件实际值。
**结论**：构造上一致，无发散风险。真实 profiling 接入前这是唯一成本来源。

### M4（审计中新识别）: SELECT 分支-guard 一致性

"cond=true 但 true 分支被 guard 关掉"是**静态不可判定**的（cond 值运行时
才知道）。当前处理：运行时 `SELECT chosen branch was skipped` 已定义失败
+ fuzz 白名单。保持运行时判定是正确的。

### M5（审计中新识别）: 死 VERIFY 与 rollback 的交互

被 `retry.on` 引用但 otherwise 不可达的 VERIFY，run() 会显式 demand 它
（rollback 需要 verdict）。即：retry 使死节点"半活"。已在 runtime-review.md
记录，语义自洽，但 optimizer 的 DCE 必须**不得删除被 retry.on 引用的
VERIFY**——这要写进未来 DCE pass 的约束。

## 结论

V1–V6 对 v0.1 范围（无循环、单 program、mock 执行）是**完备的**：
fuzzer 10000 例零误拒、零漏拒（builder 非法构造另行验证于单测）。
最重要的真实缺口是 **M1 effect ordering**——它是 v0.2 IR 设计的第一优先级。
