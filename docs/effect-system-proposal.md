# Effect System Proposal — TaskIR v0.2

> 状态：**提案**（Phase 3 按计划只做设计，不动 IR 代码）。
> 动机：`docs/validator-audit.md` M1 —— 两个无数据依赖的 action（SEND、
> SAVE、EXEC_ACTION、DB mutation）目前**可以被 scheduler 合法交换**，
> IR 无法表达"先扣款、后发确认邮件"。

---

## 1. 问题

当前 TaskIR 是纯数据流 + 显式控制依赖（`after`）：

```
%a = SEND(%msg1)          ; 订阅确认
%b = SEND(%msg2)          ; 支付凭证
```

- `a`、`b` 无数据依赖 → list scheduler 会并行/重排它们；
- `after` 可以手工表达顺序，但它是**advisory 且不可扩展**：编译器生成的
  IR（Qwen2B compiler、optimizer 做并行化时）需要**结构化的顺序屏障**，
  而不是依赖人手写 `after`。

真实 AI task 的 action 不是纯 SSA value：它们改变世界状态，不可交换、
不可重复、不可随意回滚。这是"任务数据流"与"可执行程序表示"的分界线。

## 2. 设计：Effect Token（类 MemorySSA）

每个 effectful 节点消费一个 effect token、产出一个新 token：

```
%e0 = <implicit entry token, per effect class>

%x  = SEND(%msg)          effect_in=%e0  effect_out=%e1
%y  = SAVE(%x)            effect_in=%e1  effect_out=%e2
```

- token 是**有序链**：同类 action 首尾相接，天然全序；
- 链是 scheduler 的**硬屏障**：`SEND(b)` 最早开始时间 ≥ `SEND(a)` 结束；
- 不同 effect class 是**独立链**，可并行（见 §5）。

### 为什么选这个表示（候选方案对比）

| 方案 | 描述 | 结论 |
|---|---|---|
| **A. 专用字段**（本提案） | 节点加 `effect_in` / `effect_out` 两个字段；`effect_out` 是隐式第二定义 | ✅ 数据流保持单值输出；token 有显式名字可做 def-use；文本/JSON 序列化自然 |
| B. 链节点 | `%e1 = EFFECT(%e0, %x)` 显式节点 | ❌ 每个动作多一个节点，程序膨胀 2×；且链节点只"观察"动作，动作本身仍需依赖链才有序，两跳才生效 |
| C. 隐式推断 | validator/scheduler 按 action 类自动串行 | ❌ 顺序成为约定而非表示——optimizer 无法推理、无法局部重排；与 `after` 一样是黑盒 |

方案 A 的代价：节点从"恰好一个定义"变为"至多两个定义（value + effect）"。
可接受：effect 定义不参与类型系统、不参与 SELECT/φ，是一条独立的
最小化通道（与 LLVM 把 MemorySSA 做成 overlay 而非混入 value 系统同理）。

---

## 3. 三个设计问题的回答

### Q1：effect token 是否属于 SSA value？

**是——它是 SSA value（类型 `Effect`），但由 action 节点隐式定义。**

- 它有唯一的定义点（产出它的 action）、可被引用、进入 def-use 检查
  （V2 免费获得：未定义的 `effect_in` 直接报错）；
- 它**不是普通数据**：不能进 SELECT、不能进算子输入、guard.cond 不能
  指向它。v0.2 的 validator 限制 `Effect` 类型只允许出现在同类 action 的
  `effect_in` 槽位——一条受约束的 SSA 通道。

### Q2：effect 依赖是否进入 DAG（V3）？

**进入。** effect 边是真实的顺序边（同 `after`）：

- 参与环检测（`effect_in` 引用后定义 → 环）；
- 参与 critical path 计算（同类 action 串行是真实成本）；
- scheduler 的 `_edges()` 加入 `effect_in`（`list_scheduler.py` 已预留该
  扩展点，见其 docstring"effect respect"段）；
- **不参与** retry 环判定（与 retry.on 同理，时间回边另论）。

### Q3：pure skill 是否需要 effect token？

**不需要，完全不需要。** SEARCH / FILTER / ARGMIN / GENERATE / VERIFY /
TRANSLATE 等纯计算（含 lm 只读调用）**零 effect 开销**：

- 不写 `effect_in/effect_out`，不接触链；
- 类比 MemorySSA：只有 memory op 进入 SSA-of-memory，纯计算不参与；
- 收益：数据集里绝大多数节点（~90%+，见 dataset_stats 的 skill 频率）
  不付任何表示代价，编译器生成的 IR 也不会被 token 噪声污染。

**副作用分级表**（进 SkillSpec，v0.2）：

| skill | effect_class | 说明 |
|---|---|---|
| `SEND` | `world` | 外部不可逆动作 |
| `EXEC_ACTION` | `world` | 兜底动作按不可逆处理（保守） |
| `SAVE` | `state` | 本地状态写入 |
| `LOAD` / `QUERY_DB` / `SEARCH` / 全部 lm/compute/transform | `None`（纯） | 只读 |

`world` 与 `state` 是两条独立链：本地保存与外部发送可并行。DB 写入
（未来 `DB_MUTATE` skill）归 `state`；`QUERY_DB` 保持只读纯函数。

---

## 4. 运行时语义（必须与 IR 一起定义）

1. **Guarded action 被跳过**：token **直通**（`effect_out ≡ effect_in`）。
   跳过 = 动作未发生 = 链不前进。链不断裂。
2. **Action 的 retry**：`world` 类动作**默认不允许 verify-retry 重执行**
   —— 不可逆动作重放 = 重复扣款。约束：
   - 重试前必须先 VERIFY（pattern：COMPUTE → VERIFY → ACTION，动作是
     链尾）；
   - `world` 类节点携带 `retry` → validator **错误**（v0.2 规则，优于
   事后修补）；幂等动作可显式标注 `hints.idempotent=true` 豁免。
3. **Rollback 与 effect**：回滚使 memo 失效重算的是**纯计算**；一旦
   `world` token 已推进（动作已发出），回滚边界**不得**跨越它——
   checkpoint 点必须包含 effect 链位置。这是 runtime checkpoint 设计的
   硬约束（记入 runtime v0.2 需求）。
4. **SELECT 与 φ**：v0.2 **禁止** effect token 进 SELECT（线性链无分叉
   天然规避）。分支中的 action（两分支各有一个 SEND）要求分支汇聚后
   **显式补链**：两条链在汇合点必须被串行化（validator 检查链拓扑是
   线性的，见 V7 提案）——这是保守但正确的 v0.2 立场；region 级
   effect-φ 留给 v0.3。

## 5. 表示细节

文本形式：

```
inputs: @task: Str

%1  = SEARCH(@task, domain="flight")
%2  = ARGMIN(%1, key="price")
%3  = SAVE(%2)                        effect[%e0 -> %e1]
%4  = SEND(@task)                     effect[%e1 -> %e2]

return %4
effects: world: %e0 -> %e2   state: %e0s -> %e1s
```

JSON 形式（节点级）：

```json
{"id": "%3", "op": "SAVE", "inputs": ["%2"],
 "effect": {"class": "state", "in": "%e0s", "out": "%e1s"}}
```

- 入口 token 约定：每个 class 的链头是隐式 `%e0<class>`（ENTRY），
  无需节点定义；
- `effect.out` 命名空间与 `%` 节点共享，validator 查重。

## 6. 谁受益

- **scheduler**：链 = 硬屏障；链间（world ∥ state）与纯计算自由并行——
  并行化的正确性依据从"猜"变成"读 IR"；
- **optimizer**：DCE 不得删 effectful 节点（否则改变世界状态）；
  code motion 不得跨越同类链重排；
- **Neural Compiler（Qwen2B）**：训练数据里 action 顺序有显式监督信号
  （链结构），模型学的是可执行程序而不是伪代码；
- **benchmark/论文**：可以量化"effect 序列化造成的并行度损失"
  （链长 vs critical path），这是 architecture 会议关心的
  parallelism-limitation 分析。

## 7. 落地计划（v0.2 实施顺序）

1. IR：`Node.effect` 字段 + `to_text`/JSON 序列化（向后兼容，缺省 null）；
2. validator：V7（见 validator-v7-proposal.md），0.1 版程序降级为警告；
3. lifter/synthetic 生成器：action 节点自动补链（数据再生成）；
4. scheduler：`_edges()` 接入 `effect.in`；
5. fuzzer：effect 链随机生成 + 链拓扑不变量（线性、无分叉、无环）；
6. runtime：checkpoint 包含 effect 链位置；guarded-skip 直通语义。
