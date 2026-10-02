# τ³ 穷尽式离线评测 — Phase 5B-2B.1

> 协议：oracle_task_view（离线编译分析 ONLY）
> Accounting：2,546/2,546 tasks | 14,834/14,834 reference actions 全部追踪

## Layer 1 — 结构（Neural Compiler 输出合法性）

| Metric | 值 |
|---|---:|
| Parse | **97.72%**（2,488/2,546） |
| Validator | **97.68%**（2,487/2,546） |

## Layer 2 — 动作语义（模型 vs 参考动作）

| | 数量 |
|---|---:|
| Reference actions 总数 | **14,834** |
| **Matched**（名称匹配成功） | **3** |
| **Missing**（参考动作未被模型覆盖） | **14,831**（99.98%） |
| Predicted action nodes | 2,502 |
| Extra（模型产出但无参考对应） | 2,499 |

| Metric | 值 |
|---|---:|
| **Action Precision** | **0.12%** |
| **Action Recall** | **0.02%** |
| **Action F1** | **0.03%** |

**根因**：模型每 task 平均产出 ~1 个 action node（2,502/2,546），
但 τ³ 每个 task 平均有 ~5.8 个 reference actions（14,834/2,546）。
模型产出的是**泛化 skill**（SEARCH/QUERY_DB/GENERATE），
而 τ³ 期望的是**域特定动作名**（toggle_wifi_calling、grant_app_permission、
reset_apn_settings）——两者名称空间完全不同。

## Layer 2b — Effect 分类

| Effect Class | Ref | Correct | Missing | Wrong |
|---|---:|---:|---:|---:|
| read_only | 468 | 0 | 468 | 0 |
| reversible_state | 6,920 | 0 | 6,920 | 0 |
| **irreversible_world** | **5,606** | **0** | **5,603** | **0** |
| other | 1,840 | 0 | 1,840 | 0 |
| **TOTAL** | **14,834** | **0** | **14,831** | **0** |

**End-to-End Effect Accuracy: 0.0%**（正确分类 0 / 14,834）

> 之前报告的 "98.5% world→safe" 是在仅 261 个对齐动作上计算的条件指标。
> 真实情况：**5,606 个 world 动作中 5,603 个完全未被模型生成**。
> 这不是"分类错误"，而是**动作缺失**。

## Layer 2c — Retry Safety

| | 值 |
|---|---:|
| Retry nodes 生成数 | 0 |
| Unsafe retry rate | **N/A**（vacuous — 模型未生成任何 retry） |

## Layer 3 — IR/Effect System Representability（独立于模型）

| | v0.1 实现 | v0.2 提案 |
|---|---:|---:|
| Multi-action tasks | 2,443 | 2,443 |
| Full representable | 0 | **199（8.2%）** |
| Partial representable | 2,443 | 2,244（91.8%） |
| Not representable | 0 | 0 |

> v0.1 无 effect 语义，所有 multi-action 均为 partial。
> v0.2 提案（per-class chain）可 full 表达 8.2% 的任务（单一 effect class）。

## 按 Domain 分列

| Domain | n | Valid% | Ref Actions | Matched | Missing | Avg Instr Len |
|---|---:|---:|---:|---:|---:|---:|
| Airline | 50 | 92.0 | 142 | 3 | 139 | 517 |
| Retail | 114 | 100.0 | 550 | 0 | 550 | 469 |
| Telecom | 2,285 | 100.0 | 13,215 | 0 | 13,215 | 497 |
| Banking | 97 | **43.3** | 927 | 0 | 927 | **1,138** |

Banking 43.3% 的根因：**instruction 平均长度 1,138 字符**
（其他域 ~500），超过 max_seq_length=1024 导致截断。

## 错误漏斗（完整版）

```
14,834 reference actions
    ↓ 模型仅产出 2,502 action nodes（17% 数量）
    ↓ 其中 2,499 为 extra（无参考对应）
    ↓ 仅 3 个名称匹配
  3 matched（0.02% recall）
    ↓ 0 个 effect 分类正确
  0 correct effect classification（0.0% e2e accuracy）
```

## 修正后的三层结论

```
结构编译             ✅ 已学会（97.7% valid on τ³）
动作生成             ❌ 严重不足（recall 0.02%——模型不知道域特定动作名）
Effect 分类          N/A（分母中 99.98% 为 missing，条件指标无意义）
Transaction IR       ⚠️ 91.8% 仅 partial（v0.2 proposal 层面）
```

## 与之前报告的差异

| 指标 | 之前报告 | 修正后 | 差异原因 |
|---|---:|---:|---|
| World→safe 错误率 | "98.5%" | **N/A** | 之前仅算 261 个对齐；真实 5,603/5,606 为 missing |
| Unsafe retry | "0" | **N/A** | vacuous（模型未生成 retry） |
| Action metrics | 未报 | **P=0.12% R=0.02%** | 现在有了完整 accounting |

## 论文结论

> **Neural Compiler 学会了 TaskIR 的结构规则（97.7% validator pass），
> 但对 τ³ 域的动作词汇完全为零（action recall 0.02%）。
> 模型不知道 telecom/customer service 的具体操作名（toggle_wifi、
> grant_permission、reset_apn），因此无法生成正确的动作序列。**
>
> 这不是分类错误或参数错误——是**动作词汇表完全缺失**。
> 要在 τ³ 上工作，模型需要学会该域的 action vocabulary，
> 这需要目标域的训练数据或 few-shot adaptation。
