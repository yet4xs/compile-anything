# 完整实验结果索引 — Compile Anything 项目

> 更新：2026-10-10
> 本文档汇总全部实验结果，供论文写作引用。

---

## Phase 1-3：基础设施

### TaskIR + Skill ISA + 验证器

| 组件 | 结果 | 工件 |
|---|---|---|
| TaskIR v0.1 SSA IR | 规范 + 文本/JSON 双序列化 | `spec/taskir-spec.md` |
| Skill ISA（31 技能 + 2 控制） | 五维成本模型 | `spec/skill-isa.md` |
| 验证器 V1-V6 | 150 单测 + 10k fuzz 零失败 | `src/validator/` |
| 调度器 + 运行时 | 投机执行 + 回滚 + trace | `src/runtime/`, `src/optimizer/` |
| 名义成本 | 能耗 3.4% / makespan 37.9% (vs 70B) | `benchmark/` |

---

## Phase 5B：神经编译器训练

| 实验 | 模型 | Parse% | Valid% | OpSeq% | Skill F1 |
|---|---|---:|---:|---:|---:|
| E0 zero-shot | 3B | 65.92 | 0.00 | 0.00 | 0.000 |
| E1 QLoRA | 3B | 99.55 | 99.10 | 88.54 | 0.92 |
| E2 zero-shot | 7B | 68.23 | 0.00 | 0.00 | 0.000 |

**结论**：参数规模不产生编译器语义；28k 审计监督使 3B 从 0% → 99.1%。

---

## Phase 5C：2×2 消融 + 外部评测

### Schema/Depth 消融（matched protocol）

| 模型 | BFCL E2E% | τ³ Pred/T | τ³ SemRecall% |
|---|---:|---:|---:|
| E1-A 基线 | 15.50 | 1.00 | 0.62 |
| E5C-S（marker） | 15.86 | 1.75 | 0.58 |
| E5C-D（depth） | 14.47 | 1.65 | 0.55 |
| E5C-SD | 15.66 | 1.75 | 0.66 |

**结论**：深度可训练（+0.75），正确性不可（recall 持平）。schema 条件化实为常量标记 bug。

### 结构/语义分离

| 基准 | 结构指标 | 语义指标 |
|---|---:|---:|
| Internal | Valid 99.1% | Skill F1 0.92 |
| BFCL | Valid 92.4% | Functional E2E 15.5% |
| τ³ | Valid 98.9% | SemRecall 0.62% |
| AgentBoard | Valid 91.7% | UNSCORED-OFFLINE |

**结论**：结构规则跨域泛化，语义选择严格绑定训练分布。

---

## Phase 6A：接地探针

### Canonicalizer（Grounder）

| 指标 | 值 | 说明 |
|---|---:|---|
| OOD family F1 | 0.917 ± 0.05 | 142 未见工具家族 |
| EXEC_ACTION Recall | 0.681 | 三 seed 一致 |
| name-masked F1 | 0.849→0.952 | 遮名不降，真语义接地 |

### 反事实（name-shortcut 检验）

| 模型 | full | name-masked | desc-masked | name-perturbed |
|---|---:|---:|---:|---:|
| G1+G2 Grounder | 0.847 | **0.849** | 0.847 | 0.622 |
| E5C-S | 0.782 | 0.577 | 0.889 | 0.488 |

**结论**：Grounder 是多通道语义；E5C-S 是名字模式匹配。

### 三个数据前端 bug

| Bug | 影响 | 修复 |
|---|---|---|
| 参数保全 16.7%→98.9% | xLAM 训练数据质量 | lifter 修复 |
| EXEC_ACTION 子串污染 539/1649 | 32.7% 错标 | 冻结过滤器 |
| xLAM capabilities 常量标记 | 19,700 条全为 ["EXEC_ACTION"] | adapter 修复 |

---

## Phase 6B：三段式前端

### 6B-1 Composer

| Arm | Valid% | OpSeq% | EA-R% | CondR-E2E% |
|---|---:|---:|---:|---:|
| C0T 裸重训对照 | 98.69 | 78.17 | 50.00 | 84.05 |
| **C2 oracle-selected** | **98.29** | **95.88** | **96.77** | **98.24** |

**B1-mechanism: CONFIRMED** / B1-seed-robustness: NOT CONFIRMED（s43 valid 67%）

### 6B-2 Resolver

| 模型 | Set F1 | Exact Set | Explicit NONE |
|---|---:|---:|---:|
| R1 generative | 0.9913 | 0.9899 | 0.0 |
| R2 scorer | **0.9947** | 0.9846 | **0.90** |

**Refusal gate 不可部署**（格式捷径：full-table prompt → 99% 误拒绝）。

### 6B-3 内部 Oracle 分解

| Arm | Final% | EA-R% |
|---|---:|---:|
| I0 E1-A 单体 | 89.94 | 88.89 |
| I1 oracle/oracle | 92.35 | 87.96 |
| I3 oracle+learned-res | **92.86** | **99.07** |
| I4 全学习 | 86.02 | 75.00 |

**结论**：Resolver 零代价，**Canonicalizer 唯一误差源**（−6.1pp）。

### 6B-3 外部 τ³——论文核心架构证据

| 系统 | SemRecall% | EA输出率% | EA ref-recall% |
|---|---:|---:|---:|
| E5C-S 单体 | 0.58 | 0.25 | ~0 |
| E5C-S + oracle IR | 0.22 | 3.28 | ~0 |
| **模块化前端（全学习）** | **23.02** | **60.23** | **100** |

**EXEC_ACTION 恢复 240 倍，语义召回恢复 40 倍。**

---

## BFCL Track A：Function-Calling Competition

### 配置演进

| 版本 | Held-out% | 训练集% | 记忆差距 |
|---|---:|---:|---:|
| 3B 1ep seq1024 | 52.49 | 80.25 | 27.8pp |
| 7B 1ep seq1024 | 77.01 | — | — |
| **7B 3ep seq2048 全量** | **89.20** | **89.81** | **0.61pp** |

### vs 竞争者

| 系统 | 分数 | 我们领先 |
|---|---:|---:|
| **我们 7B MAX** | **89.20%** | — |
| xLAM-7B | ~85% | +4pp |
| Hammer-7B | ~82% | +7pp |
| BTL-3（闭源 top） | ~88.5% | 持平 |

---

## AgentBoard Untouched 首测

| Task | n | Parse% | Valid% | EA输出率% |
|---|---:|---:|---:|---:|
| tool-query | 60 | 98.3 | 78.3 | 2.5 |
| tool-operation | 40 | 95.0 | 87.5 | **23.9** |
| webshop | 251 | 98.4 | 96.0 | 0.0 |

---

## 论文核心叙事链

```
1. 结构编译可学        valid 0→99.1%（3B + 28k 审计监督）
2. 结构跨域泛化        三基准家族 92-99%
3. 语义鸿沟存在        BFCL 15.5% / τ³ 0.6%
4. 故障定位            三 bug + EXEC_ACTION 边界 + Composer 瓶颈
5. 分层修复            Canonicalizer 0.917 / Resolver 0.995 / Composer 0.981
6. 模块化跨域恢复      EA 0.25%→60.23%（240×）
7. BFCL 竞争力         7B 89.20%（超越 xLAM-7B）
```

---

## 全部工件索引

| 阶段 | 关键文档 | 关键数据 |
|---|---|---|
| 基础设施 | `spec/taskir-spec.md`, `spec/skill-isa.md` | `docs/validator-audit.md` |
| Phase 5B | `docs/results-phase5b1-complete.md` | `experiments/phase5b1/` |
| Phase 5C | `docs/results-phase5c.md` | `results/phase5c/` |
| Phase 5D | `docs/paper-evaluation.md` | `experiments/paper_snapshot_v1.json` |
| Phase 6A | `docs/phase6/results-phase6a-grounding-probe.md` | `results/phase6/` |
| Phase 6B | `docs/phase6/results-phase6b1.md`~`results-6b3-external.md` | `results/phase6b/` |
| BFCL Track A | `docs/results-bfcl-track-a.md` | `results/bfcl_track/` |
| AgentBoard | `docs/results-agentboard-first-test.md` | `results/agentboard/` |
