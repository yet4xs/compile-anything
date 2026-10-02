# τ³ v3 — Semantic-Skill Aligned Evaluation

> 协议：oracle lowering（τ³ 工具 → Skill ISA），非 concrete-name match
> Oracle：src/eval/tau3_skill_oracle.py（deterministic，from tool definitions only）

## 核心对比：v2 vs v3

| 指标 | v2（concrete name match） | v3（semantic skill match） |
|---|---:|---:|
| Matched | 3/14,834 | **89/14,137** |
| Action Recall | 0.02% | **0.63%** |
| Effect conditional acc | N/A（0 aligned） | **100%**（89/89） |
| Effect e2e recall | 0.0% | **0.63%** |

## 分层结果

### L1 — Structural
| Parse | 97.7% |
| Validator | 97.7% |

### L2-A — Semantic Skill Planning
| | 值 |
|---|---:|
| Reference semantic actions | 14,137 |
| Matched (semantic skill level) | 89 |
| Missing | 14,048 |
| **Semantic Skill Recall** | **0.63%** |
| **Effect Conditional Accuracy** | **100%**（89/89） |

### L2-B — Under-Planning（最可靠信号）
| | 值 |
|---|---:|
| **Ref actions/task** | **5.83** |
| **Pred actions/task** | **0.98** |
| **Action count ratio** | **0.17** |
| Multi-action ≥50% coverage | 6/2,385（0.3%） |

### L4 — Effect Classification (per class, end-to-end)
| Class | Ref | Correct | Missing | Wrong |
|---|---:|---:|---:|---:|
| read_only | 468 | 86（18.4%） | 358 | 0 |
| reversible_state | 6,919 | 0（0%） | 6,908 | 0 |
| irreversible_world | 5,670 | 3（0.1%） | 5,436 | 0 |
| other | 1,777 | 0（0%） | 1,346 | 0 |

### Per-Domain
| Domain | n | Valid% | Match% | Pred/T | Ref/T |
|---|---:|---:|---:|---:|---:|
| Airline | 50 | 92.0 | **19.7%** | 0.98 | 2.84 |
| Retail | 114 | 100.0 | **11.1%** | 1.05 | 4.82 |
| Telecom | 2,285 | 100.0 | **0.0%** | 1.00 | 5.78 |
| Banking | 97 | 43.3 | 0.0% | 0.48 | 9.56 |

### Token Analysis (Banking root cause)
| Domain | p50 | p90 | >1024 | %>1024 |
|---|---:|---:|---:|---:|
| (see metrics_v3.json for full table) |

## 关键发现

### 1. Effect Classification 是 100% 条件准确
当模型生成了正确的 semantic skill 时，effect 分类**全部正确**（89/89）。
这说明模型在它能理解的范围内不会犯 effect 错误——
问题不是"分不清 reversible vs irreversible"，而是**根本没生成那些动作**。

### 2. Under-planning 是最可靠的失败信号
| Ref/task | Pred/task | 比率 |
|---|---|---|
| 5.83 | 0.98 | **17%** |

模型对 stateful 任务严重 under-plan：平均只生成 1 个动作 vs 需要近 6 个。
这与 namespace / abstraction level 无关——纯粹的规划长度不足。

### 3. Telecom 域零匹配是主要拖累
Telecom 占 90% 的样本（2,285/2,546），但 semantic match 为 0。
Airline（19.7%）和 Retail（11.1%）有部分匹配。
说明模型对 airline/retail 域有一定泛化，对 telecom 完全没有。

### 4. Oracle 证据来源
| 来源 | 数量 | 占比 |
|---|---:|---:|
| heuristic（name-based） | 13,920 | 99.1% |
| isa_spec | 128 | 0.9% |

几乎全部依赖 heuristic——τ³ 工具没有足够的 description 来做 evidence-based 分类。

## 修正后的五层结论

```
L1 结构编译          ✅ 97.7% valid
L2 语义 skill 选择   ❌ 0.63% recall（严重不足）
L2 Under-planning    ❌ 0.17 ratio（每 task 产出 1 个 vs 需要 5.8 个）
L4 Effect 分类       ✅ 100% conditional（对齐时全对，问题是没对齐）
L5 IR Representability ⚠️ 91.8% partial（v0.2 proposal 层面）
```

## 与 GPT 预期的对照

GPT 预期两种可能：
- Semantic Skill Recall = 50%, Concrete = 5% → "TaskIR abstraction 有效，缺 backend binding"
- Semantic Skill Recall = 5%, Concrete = 1% → "跨域 semantic planning 失败"

**实际结果**：0.63% semantic recall → 更接近第二种，但有一个重要的正面信号：
**Effect conditional accuracy = 100%**（当模型选对 skill 时，effect 全对）。

这说明模型有正确的 effect 直觉，只是不知道 τ³ 域的具体动作。
问题不是"理解能力"而是"词汇表覆盖"。
