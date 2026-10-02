# τ³ Phase 5B-2B.2 — Complete Numbers for Decision

> 协议：oracle_task_view，semantic-skill aligned evaluation + binder layer
> 全部 14,834 reference actions accounted for

## GPT 决策树数字

| 指标 | 值 |
|---|---:|
| **Semantic Skill Recall** | **0.63%**（89/14,137） |
| **Concrete Tool Recall** | **0.02%**（3/14,834） |
| **Action Count Ratio** | **0.17**（17%） |

### → 判定：第二种情况
> "模型学会了 TaskIR 结构，但跨域 semantic planning 本身没有学会。"

## 全部数字

### L1 — Structural

| | 值 |
|---|---:|
| Parse | 97.72% |
| Validator | 97.68% |

### L2-A — Semantic Skill Planning

| | 值 |
|---|---:|
| **Semantic Skill Precision** | **3.56%** |
| **Semantic Skill Recall** | **0.63%** |
| **Semantic Skill F1** | **1.07%** |
| **Semantic Sequence Exact** | 10/2,478（0.4%） |
| Ref semantic actions | 14,137 |
| Matched | 89 |
| Missing | 14,048 |
| Pred action nodes | 2,502 |
| Extra（unmatched pred） | 2,413 |

### L2-B — Under-Planning

| | 值 |
|---|---:|
| **Ref actions/task** | **5.83** |
| **Pred actions/task** | **0.98** |
| **Count ratio** | **0.17** |
| Multi-action tasks | 2,385 |
| Multi ≥50% coverage | 6（0.3%） |

### L2-C — Concrete Tool Binding（Binder Layer）

| | 值 |
|---|---:|
| **Concrete Tool Recall** | **0.02%**（3/14,834） |
| **Concrete Tool Precision** | **0.12%** |
| **Concrete Tool F1** | **0.03%** |

### L4 — Effect Classification

| | 值 |
|---|---:|
| **Effect F1（conditional）** | **100%**（89/89） |
| **Effect E2E Recall** | **0.63%**（89/14,137） |

Per class：
| Class | Ref | Correct | Missing |
|---|---:|---:|---:|
| read_only | 468 | 86（18.4%） | 358 |
| reversible_state | 6,919 | 0 | 6,908 |
| irreversible_world | 5,670 | 3（0.05%） | 5,436 |
| other | 1,777 | 0 | 1,346 |

### L2-R — Retry

| | 值 |
|---|---:|
| Retry nodes | 0 |
| Unsafe retry | N/A（vacuous） |

### Banking Token Analysis（Root Cause）

| Domain | p50 | p90 | >1024 | %>1024 |
|---|---:|---:|---:|---:|
| Airline | 105 | 203 | 0 | 0% |
| Retail | 29 | 41 | 0 | 0% |
| Telecom | 232 | 264 | 0 | 0% |
| **Banking** | **883** | **1,439** | **37** | **38.1%** |

Banking 43.3% valid 的根因确认：**38.1% 的 banking instruction 超过 1024 tokens**。
其他域 0% 超限。

### L5 — IR Representability（独立于模型）

| | v0.1 实现 | v0.2 提案 |
|---|---:|---:|
| Multi-action tasks | 2,443 | 2,443 |
| Full | 0 | 199（8.2%） |
| Partial | 2,443 | 2,244 |
| None | 0 | 0 |

### Per-Domain 完整

| Domain | n | Valid% | Match% | Pred/T | Ref/T |
|---|---:|---:|---:|---:|---:|
| Airline | 50 | 92.0 | **19.7** | 0.98 | 2.84 |
| Retail | 114 | 100.0 | **11.1** | 1.05 | 4.82 |
| Telecom | 2,285 | 100.0 | **0.0** | 1.00 | 5.78 |
| Banking | 97 | 43.3 | 0.0 | 0.48 | 9.56 |

### Effect Confusion Matrix（含 MISSING）

| Ref \ Pred | read_only | irr_world | MISSING |
|---|---:|---:|---:|
| read_only | 86 | 0 | 358 |
| reversible_state | 0 | 0 | **6,908** |
| irreversible_world | 0 | 3 | **5,436** |
| other | 0 | 0 | 1,346 |

### Oracle Evidence Distribution

| 来源 | 数量 | 占比 |
|---|---:|---:|
| heuristic | 13,920 | 99.1% |
| isa_spec | 128 | 0.9% |

---

## 三层错误漏斗

```
14,834 concrete reference actions
    ↓ oracle lowering
14,137 semantic reference actions
    ↓ E1 model generates only 2,502 nodes (17% of needed)
    ↓ only 89 semantically match (0.63% recall)
    ↓ only 3 match at concrete tool level (0.02%)
  3 concrete tool matches
```

## 核心结论

1. **结构编译** ✅：97.7% valid on τ³（最强结果）
2. **语义规划** ❌：0.63% recall — 模型不知道 τ³ 域的 skill 选择
3. **Under-planning** ❌：0.98 vs 5.83 actions/task（17%）— 独立于 ontology
4. **Effect 理解** ✅：100% conditional（89/89 全对）
5. **Concrete binding** ❌：0.02% — 没有工具词汇覆盖
6. **Banking** ❌：38.1% 超 token 限制（root cause 确认）
7. **v0.2 提案** ⚠️：91.8% partial representability

**最大瓶颈：semantic planning（skill 选择 + multi-step 分解）**
