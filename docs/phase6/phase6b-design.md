# Phase 6B 架构设计 — 三段式 Compiler Frontend

> 状态：spec 冻结于 Phase 6B-0 结果之前（先设计后看数，架构选择待 6B-0 表格裁决）
> 依据：Phase 6A 六问裁决（`docs/phase6/results-phase6a-grounding-probe.md`）+ paper_snapshot_v2
> 禁令：Skill ISA 不修改；CapabilityIR 是 frontend ABI/metadata 层，不是新 IR

## 1. 架构

```text
Raw Capability (name / description / parameter schema)
      ↓
[1] Capability Canonicalizer          ← task-independent，可预计算/缓存
      ↓
CapabilityIR {
    concrete_name
    canonical_skill
    skill_family
    confidence
    evidence_channel: name | description | schema | hybrid
}
      ↓
[2] Task-conditioned Resolver         ← Phase 6B 新核心模块
      ↓
selected CapabilityIR ids (+ NONE/NO_CALL decision)
      ↓
[3] TaskIR Composer
      ↓
TaskIR（结构 / 依赖 / 技能序列 / 参数 / 控制语义）
```

编译器类比：

```text
Canonicalization ≈ ABI / symbol semantic typing
Resolver         ≈ name resolution / instruction selection
Composer         ≈ IR construction
```

## 2. 各模块设计依据（Phase 6A 实证）

### [1] Canonicalizer — task-independent

- 依据：**Probe A（schema-only）> Probe B（task+schema）**——canonical 语义类型是
  capability 的静态属性；task context 不应污染这一层。
- 已达指标（G1+G2 grounder，未见家族）：F1 0.917±0.05、EXEC_ACTION R 0.681。
- **multi-channel 训练（强制）**：channel dropout 增广——
  `full / name-only / description+schema-only / name-masked / description-masked`
  五种输入形态随机出现，确保同一 canonicalizer 在不同 metadata 可得性下都工作
  （τ³ 只有名字、xLAM/ToolBench 有全 schema、AgentBoard/BFCL 介于其间）。
- 输出 evidence_channel 字段：自评主要证据来源，供 downstream 置信处理与审计。

### [2] Resolver — Phase 6B 新核心

- 依据：**Probe C selection ≈ 0.20（随机）而 skill 分类 0.65**——
  "知道工具是什么" ≠ "知道任务该用哪个工具"，必须分层。
- 输入：task + N 个 CapabilityIR；输出：selected capability ids（可空）。
- **NONE/NO_CALL 是 resolver decision，不是 Skill ISA instruction**。
  负例必须 `synthetic_negative=true` 并单独报告真实/合成比例；
  显式监督（Phase 6A 证明不监督则不涌现：NO_CALL rate 0.0）。
- 训练数据：corpus 内单/多工具任务天然提供正例选择监督；
  负例 = 移除正确 capability 留 distractors（沿用 Phase 6A 构造）。

### [3] Composer — 只管程序

- 输入：task + selected canonical capabilities（含 canonical_skill 标注）。
- 职责边界：program structure / dependencies / skill sequence / arguments /
  control semantics（guard/SELECT/VERIFY/retry）。**不重新猜 concrete tool ontology**。
- Phase 6B-0 的 A/B/C/D 表格直接决定 Composer 是否为瓶颈（见 §4）。

## 3. 实验矩阵（Task 7，先冻结）

| Arm | 配置 | 回答 |
|---|---|---|
| B0 | E5C-S whole compiler（现状） | 基线 |
| B1 | Oracle CapabilityIR → E5C-S Composer | 上界诊断（非端到端成绩） |
| B2 | Learned Canonicalizer → Resolver → Composer | 模块化三段式 |
| B3 | Shared multi-task frontend（grounding + selection + TaskIR 生成联合训练） | 模块化 vs 共享表征 |

B2 vs B3 回答：**modular compiler frontend 是否优于 shared multi-task model**
（Phase 6A 信号分歧：grounder 内部赢、E5C-S 外部赢——不预设 two-stage 一定赢）。

## 4. Phase 6B-0 结果与裁决（2026-10-04，四协议终表）

| Protocol | Capability info | Parse% | Valid% | Pred/T | SemRecall% | EXEC_ACTION 输出率% | EXEC_ACTION ref-R% |
|---|---|---:|---:|---:|---:|---:|---:|
| A | none（冻结基线） | 97.25 | 96.74 | 1.75 | 0.58 | 0.25 | — |
| B | names | 87.04 | 87.00 | 1.66 | 0.24 | 2.78 | 100* |
| C | learned CapabilityIR | 87.12 | 87.04 | 1.66 | 0.23 | 3.01 | — |
| D | **oracle CapabilityIR** | 87.04 | 87.00 | 1.66 | 0.22 | **3.28** | 100* |

\* 条件值：仅在已匹配参考子集内统计，分母极小，只表示方向性。

**裁决（按 §4 预冻结规则）：D 路径失败 —— 瓶颈在 Composer（TaskIR 程序生成），不在 information routing。**

1. **oracle 标注救不回 whole compiler**：即便把每个工具的正确 canonical skill 显式写进 prompt，EXEC_ACTION 输出率仅 0.25%→3.28%，语义召回 0.58%→0.22%（不升反降）。B→C→D 的微弱单调爬升（2.78→3.01→3.28）说明标注质量有方向性影响，但量级远不足以修复。
2. **Phase 6A 的"知识存在但不表达"结论现在受控变量支持**：同模型、同基准，显式注入 grounding 知识仍无效 → 失败不在"知识没送进 frontend"，而在程序生成本身。
3. **附带发现（格式混淆项，如实披露）**：任何 capability 块都使 parse/valid 掉 ~10pp（97→87）——E5C-S 从未学过以真实 capability 表为条件的程序生成（5C 的"条件"是常量标记，真实表格对其为 OOD）。A-vs-B 混合了信息与格式两个变量，但 D 的核心结论不受影响。
4. **对 6B 架构的直接含义**：按预冻结规则，继续训练 grounder 无意义；Phase 6B 主攻 **Composer**——以 grounded capability 表为条件的 TaskIR 生成训练（需 corpus v4：用修复后的 adapter 产出真实 capabilities），三段式架构中的 Canonicalizer/Resolver 已有 Phase 6A 证据支撑，Composer 成为下一个训练靶。

### 原 §4 预冻结规则（保留存档）

| Protocol | 输入 | 若 EXEC_ACTION ref-recall 仍 ≈0 | 若大幅恢复 |
|---|---|---|---|
| B（names） | 具体工具名 | 名字进 frontend 也不够 | routing 缺失是主因 |
| C（learned IR） | grounder 标注 | 学习标注不够（或标注错） | two-stage 成立 |
| D（oracle IR） | oracle 标注 | **composition/生成路径是瓶颈，训 grounder 无意义，改 Composer** | grounding information routing 是瓶颈 |

判定链：`B↑ 且 C↑↑ 且 D↑↑` → two-stage 依据充分；
`D≈0` → 问题在 Composer，Phase 6B 转向 composer 训练（如 canonical-skill-conditioned 程序生成）。

## 5. 分层指标（Task 8，禁止单一端到端准确率掩盖失败位置）

**Canonicalizer**：Boundary Macro F1 / Exact Skill F1 / EXEC_ACTION Recall /
channel robustness（五种 dropout 形态各自的上述指标）

**Resolver**：top-1 selection / top-k recall / NO_CALL precision·recall·F1
（真实/合成分开报）

**Composer**：Parse / Validator / OpSeq / Semantic Recall / Pred/T

**End-to-end**（联合链）：correct capability selected ∧ correct canonical skill
∧ valid TaskIR ∧ semantic action matched（同时报告各合取项的边缘比率，定位最弱环节）

## 6. 数据与合规

- 训练语料：如需重建（利用 adapter 修复后的真 capabilities），建 **corpus v4**，
  禁止原地覆盖 v3.1；external benchmarks 照旧禁入训练。
- CapabilityIR 表可预计算缓存（canonicalizer task-independent 的工程收益）。
- 评估协议沿用 Phase 6A 冻结 manifest；外部仅 diagnostic。
