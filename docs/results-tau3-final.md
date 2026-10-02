# τ³ Phase 5B-2B.3 — Final Grounded Evaluation

> Oracle hardened with evidence sources. Binder upper bound computed.
> Banking truncation causally split. Planning depth distribution complete.

## 核心发现

### 1. Oracle 硬化后：88% grounded（vs 之前 99.1% heuristic）

| Evidence Source | Count | % |
|---|---:|---:|
| isa_rule | 12,547 | 84.6% |
| heuristic | 1,777 | 12.0% |
| tool_definition | 505 | 3.4% |
| policy | 5 | 0.0% |
| **Grounded (policy+tool_def+isa)** | **13,057** | **88.0%** |

之前报告的 "99.1% heuristic" 是因为旧 oracle 没有 isa_rule 作为 evidence source。
加入 isa_rule 后（这确实是确定性规则，不是猜测），grounded 覆盖率升至 88%。

### 2. Semantic Skill Metric（grounded-only）

| Subset | n | Matched | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| **Grounded-only** | 12,791 | 89 | 3.56% | **0.70%** | **1.16%** |
| Heuristic-only | 1,346 | 0 | 0% | 0% | 0% |
| All | 14,137 | 89 | 3.56% | 0.63% | 1.07% |

Grounded-only 的 recall 略高（0.70% vs 0.63%），但整体仍然极低。

### 3. Binder Oracle Upper Bound：99.95% ✅

| | P | R | F1 |
|---|---:|---:|---:|
| **Oracle plan → binder → concrete** | **99.95%** | **99.95%** | **99.95%** |

这证明：**如果 frontend 给 binder 正确的 semantic plan，binder 几乎完美**。
Binder 不是瓶颈。瓶颈完全在 frontend（semantic planning）。

### 4. Effect Consistency（given correct semantic skill）

| Class | n | Correct | % | Note |
|---|---:|---:|---:|---|
| read_only | 86 | 86 | 100% | |
| reversible_state | 0 | — | — | no matches to test |
| irreversible_world | 3 | 3 | 100% | **small-n (n=3)** |
| other | 0 | — | — | no matches |

Effect consistency 100% 但：
- 主要是 read_only（86/89 = 97%）
- reversible_state 和 other 完全没有 matched cases
- irreversible_world 仅 3 个（small-n）
- **不能声称"模型理解了 effect"**，只能说"Skill ISA 的一致性注释是自洽的"

### 5. Banking Truncation：因果确认 ✅

| | n | Valid | P(valid) |
|---|---:|---:|---:|
| **Truncated (>1024 tokens)** | 37 | **0** | **0.0%** |
| **Non-truncated** | 60 | **42** | **70.0%** |

因果结论：
- Truncated: 37/37 全部失败（100% failure rate）
- Non-truncated: 18/60 失败（30% failure rate）
- **Truncation 是 banking 失败的 primary contributor**（67% of failures）
- 但也有 18 个 non-truncated 失败（可能是内容复杂度或其他因素）

### 6. Planning Depth Distribution（Under-planning 细化）

| Ref Actions | n | Ref/T | Pred/T | Ratio |
|---|---:|---:|---:|---:|
| **1** | 94 | 1.00 | 0.99 | **0.99** ✅ |
| **2-3** | 276 | 2.69 | 1.00 | **0.37** |
| **4-6** | 1,255 | 5.11 | 1.00 | **0.19** |
| **7+** | 912 | 8.31 | 0.96 | **0.12** |

**Action Count MAE: 4.85**

关键发现：
- 1-action tasks: ratio=0.99 → 模型对单步任务几乎完美
- 多步任务: ratio 从 0.37 降到 0.12 → **模型无论需要多少步，始终只产出 ~1 个 action**
- 这不是"逐步变差"，而是**模型被训练成了单步输出**（训练数据中大多数任务是 1-2 步）

## 最终论文级结论

### A. Structural Generalization ✅
> 97.7% validator pass on τ³（与内部测试 99.1% 接近）

### B. Semantic Grounding ❌
> Semantic skill recall 0.70%（grounded-only），88% 的 oracle 有确定性证据

### C. Planning Depth ❌（最可靠信号）
> 1-action ratio=0.99 → 完美
> 7+ action ratio=0.12 → 严重 under-planning
> MAE=4.85（平均差 5 个 action）
> **模型被训练成了"总是输出 1 个 action"**

### D. Binder ✅
> Oracle upper bound 99.95% → binder 完全不是瓶颈

### E. Effect Semantics ⚠️
> Conditional 100%（89/89），但 97% 是 read_only，不足以声称"理解了 effect"

### F. Context-Length ✅
> Banking truncated: 0% valid / non-truncated: 70% valid → truncation 是 primary contributor

### G. IR Representability ⚠️
> v0.1: 0% full（无 effect 语义）
> v0.2 提案: 8.2% full / 91.8% partial
> 需要 v0.2 effect system 才能完整表达跨类事务
