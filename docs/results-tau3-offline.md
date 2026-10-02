# τ³ Offline Compiler / Effect Evaluation — Phase 5B-2B

> 协议：oracle_task_view（离线编译分析 ONLY，非官方 τ³ task success）
> 模型：E1 (Qwen 3B QLoRA) — 冻结
> 样本：2,546 tasks（airline 50 / retail 114 / telecom 2,285 / banking 97）

## 核心结果

| Metric | 值 | 说明 |
|---|---:|---|
| **Parse** | **97.72%** | 2,488/2,546 |
| **Validator** | **97.68%** | 2,487/2,546 |
| **Unsafe World Retry** | **0** | 无跨不可逆动作的重试 |

## 三层结论（回答 GPT 的 RQ1-RQ4）

### RQ1: 结构合法性 ✅
> E1 在强 OOD stateful workload 上**仍能生成结构合法 TaskIR**（97.7%）
> — 比 BFCL（92.4%）更高，因为 τ³ 任务描述更结构化

### RQ2: 语义/动作规划 ❌
> Effect classification 极差：**98.5% 的 irreversible_world 动作被模型归为 safe**

### RQ3: Effect System 表达力 ⚠️
> TaskIR skill 集合中绝大多数是 read_only，模型没有生成 effectful 操作

### RQ4: Transaction gap ✅ 验证
> 2,443 个 multi-action 任务中 **89.6% 仅 partially representable**
> — 证实 Phase 3 判断：per-class effect chain 可以表达 ordering，
> 但不能表达跨 state/world 的 atomic transaction

## 按 Domain 分列

| Domain | n | Parse% | Valid% | Multi-action | Unsafe Retry |
|---|---:|---:|---:|---:|---:|
| Airline | 50 | 94.0 | 92.0 | 25 | 0 |
| Retail | 114 | 100.0 | 100.0 | 92 | 0 |
| Telecom | 2,285 | 100.0 | 100.0 | 2,240 | 0 |
| Banking | 97 | **43.3** | **43.3** | 86 | 0 |
| **TOTAL** | **2,546** | **97.7** | **97.7** | **2,443** | **0** |

Banking_knowledge 显著偏低（43.3%）— 可能因为知识域任务描述更长/更复杂。

## Effect Class 混淆矩阵

| Ref \ Pred | read_only | irr_world | 正确率 |
|---|---:|---:|---:|
| read_only | 92 | 4 | **95.8%** |
| reversible_state | 2,021 | 4 | **0.2%** |
| irreversible_world | **257** | 4 | **1.5%** |
| other | 108 | 3 | **2.7%** |

**最危险错误**：`irreversible_world → read_only` = 257 次（98.5%）
— 模型几乎从不将动作识别为不可逆。

**原因分析**：模型生成的 TaskIR 绝大多数是 SEARCH/FETCH/QUERY_DB 等
read_only 类 skill。τ³ 的 telecom/customer service 操作（toggle_wifi、
grant_permission、reset_apn 等）在模型的 skill 词表中没有对应的
effectful 映射。

## Transaction Representability

| | 数量 | 占比 |
|---|---:|---:|
| Multi-action tasks | 2,443 | — |
| **Full representable** | 196 | **8.0%** |
| **Partial representable** | 2,189 | **89.6%** |
| Not representable | 58 | 2.4% |

Partial = 跨 state/world 效应类的序列，per-class chain 可表达顺序
但无法表达 all-or-nothing atomicity。

## 论文结论表

```
结构编译（语法+SSA+类型+DAG）    ✅ 已学会（97.7% valid on τ³）
语义规划（skill 选择+effect 分类）  ❌ 未跨域泛化（98.5% world→safe 错误）
State/Effect IR                    ⚠️ 部分成立（89.6% partial representable）
```

这比 BFCL 结果更清晰地证实了 compiler 架构分层：
> **TaskIR 的结构层是通用可泛化的，但语义层（skill 选择/effect 分类）
> 和 transaction 语义需要目标域训练或 ISA 扩展。**

## 与 BFCL 对比

| | BFCL V4 | τ³-bench |
|---|---:|---:|
| Validator | 92.4% | **97.7%** |
| 语义正确 | 10.4% strict | ~0%（effect 几乎全错） |
| 主要失败模式 | irrelevance detection | effect classification |
| Transaction gap | N/A | **89.6% partial** |

τ³ 的语义比 BFCL 更差（0% vs 10%），因为：
1. τ³ 是 stateful agent benchmark（训练数据全是静态工具调用）
2. τ³ 的动作语义（toggle/grant/reset）在训练数据中不存在
3. Effect 分类需要理解动作的世界影响，模型没有学过
