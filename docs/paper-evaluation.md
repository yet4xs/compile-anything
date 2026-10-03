# Evaluation 章节（§9）— Phase 5D 终版（真实结果）

> **状态：evidence frozen（Phase 5D）。** 所有数字来自仓库冻结产物（见 §10 工件表与
> `experiments/paper_snapshot_v1.json`），不再修改。此前版本中"训练未跑/PENDING/空模板"
> 已全部删除。纪律不变：PENDING 禁止预填，名义成本一律标注 nominal，离线不可得指标
> 标注 UNSCORED-OFFLINE。

## 1. 实验链总览

评测回答四个递进问题：(i) 小模型能否学会把 NL 任务**编译**为合法 TaskIR（编译质量）；
(ii) 该能力是否泛化到从未见过的外部基准家族（结构泛化）；(iii) 合法程序是否还原了
任务**语义**（跨域语义接地）；(iv) 各训练选择贡献几何（2×2 因子消融）。

| 实验 | 内容 | 状态 |
|---|---|---|
| E0 | Qwen2.5-3B zero-shot | DONE |
| E1 / E1-A | 3B QLoRA（28,093 条，corpus v3.1） | DONE |
| E2 | 7B zero-shot 规模对照 | DONE |
| **E3** | **7B LoRA** | **NOT RUN**（如实标注，不以估计值填充） |
| Phase 5C 2×2 | schema 条件化 × 深度课程前端消融（A/S/D/SD） | DONE |
| BFCL 外部开发集 | 4,696 条，冻结 oracle 三级语义判定 | DONE |
| τ³ 外部开发集 | 2,546 条，skill-oracle 降级对齐 | DONE |
| AgentBoard untouched | 351 条一次性离线确认（预注册） | DONE |
| 官方交互环境 AgentBoard | progress/success rate | NOT DONE（UNSCORED-OFFLINE） |

## 2. Internal：神经编译器主结果（T7）

test split n=1,561，三级闸门 + 指令选择指标（`benchmark/neural_compiler_eval.py`）：

| Model | Train | Parse% | Valid% | Exec% | OpSeq% | Skill F1 |
|---|---|---:|---:|---:|---:|---:|
| 3B zero-shot（E0） | 0 | 65.92 | 0.00 | 0.00 | 0.00 | 0.000 |
| 3B QLoRA（E1） | 28,093 | **99.55** | **99.10** | **99.10** | **88.54** | **0.92** |
| 7B zero-shot（E2） | 0 | 68.23 | 0.00 | 0.00 | 0.00 | 0.000 |

解读：

> **参数规模本身没有产生 TaskIR compiler semantics**：7B zero-shot 与 3B zero-shot
> 同样在 validator 闸门前全军覆没（valid 0%），其 68% 的 parse 率只说明模型能模仿
> 表面语法。28k 条经语义审计的高质量监督使 3B 从 surface-format imitation 转变为
> 几乎完全 structural-valid 的编译（valid 99.10%）。不声称"3B 优于 7B"——没有
> 训练后的 7B 对照（E3 NOT RUN）。

分 source（E1）：xLAM 994（valid 98.99%）、Spider 555（99.82%）为绝对主体；
toolbench_static 9 / humaneval 1 / mbpp 2 样本过少，仅作记录。Per-skill F1 主体
技能 0.82–1.00（QUERY_DB 0.978、CALCULATE 0.988、TRANSLATE 1.000）。

## 3. Structure / Semantics 分离（核心表 T8-a）

同一编译器在四个基准家族上的结构与语义指标对照（E1-A 基线口径）：

| Benchmark | Structural metric | Semantic metric | 主观察 |
|---|---:|---:|---|
| Internal（n=1,561） | Valid 99.10% | Skill F1 0.92 | 域内编译已学会 |
| BFCL V4（n=4,696） | Valid 92.4% | Functional E2E 15.50% | 结构迁移，语义弱 |
| τ³（n=2,546） | Valid 98.86% | Grounded recall 0.62% | 严重跨域接地失败 |
| AgentBoard（n=351，untouched） | Valid 91.7% | UNSCORED-OFFLINE | 结构结果第三次复制 |

**这是本文的中心实证发现**：结构合法性（SSA/类型/DAG/validator 不变量）零样本迁移到
三个从未见过的基准家族；语义技能选择严格绑定于训练分布。两者以数量级计的落差
（92~99% vs 0.6~16%）说明它们是**可分离的能力维度**，也证明"validator 通过 ≠ 编译正确"。

定版模型 **E5C-S**（Phase 5C 选出，schema 部署协议）单列：

```text
Internal (matched protocol):  Valid 99.10%   OpSeq 94.11%
BFCL:   Functional | Valid    18.19%（诊断口径）
        Functional E2E        15.86%（主指标，基线 15.50）
τ³:     Semantic Recall        0.58%
        Pred/T                 1.75   （参考 5.83）
AgentBoard (offline): Parse 97.7%   Valid 91.7%
```

## 4. Phase 5C 2×2 前端消融（T8-b）

四组：A（基线）/ S（schema 条件化）/ D（深度课程）/ SD（双因素），统一 matched
推理协议，交互项按 SD−S−D+A 计算（freeze-fix 修正后口径）：

| 因变量 | Schema 主效应 | Depth 主效应 | 交互 |
|---|---:|---:|---:|
| Internal OpSeq（matched） | +1.83pp | −4.77pp | +0.96pp |
| τ³ Pred/T | +0.425 | +0.325 | **−0.65（饱和）** |
| τ³ Semantic Recall | −0.02pp | +0.05pp | +0.15pp（噪声） |
| BFCL Functional E2E | **+0.77pp** | −0.62pp | +0.84pp |
| BFCL Functional\|Valid（诊断） | +1.92pp | −1.04pp | +1.12pp |

### 4.1 Schema conditioning（正结果，幅度小）

```text
Internal OpSeq:            92.76 → 94.11
BFCL Functional E2E:       15.50 → 15.86   （+0.35pp，相对 +2.3%）
BFCL Functional|Valid:     16.83 → 18.19   （仅作诊断）
τ³ Pred/T:                 1.00  → 1.75
```

> Schema-conditioned training provides a small positive improvement in cross-domain
> semantic correctness, preserves internal compiler quality, and strongly changes
> the model's planning-length prior.

注意两点必须保留：(i) E2E +0.35pp 是主口径，**不得只引条件口径 +1.36pp**；
(ii) schema 模型推理时必须携带 capability schema（无 schema 推理时 OpSeq 崩至 44%，
train/eval 协议失配所致，非能力丢失）。此前 capability-at-inference-only 消融
（E1 上 +0.2pp）说明**训练时见过 schema 才有效**。

### 4.2 Depth curriculum（负结果，如实保留）

> Depth reweighting increases generated plan length but does not improve
> semantic correctness.

```text
τ³ Pred/T:                1.00 → 1.65     （长度↑）
τ³ Semantic Recall:       0.62% → 0.55%   （语义不升反微降）
Internal OpSeq:           92.76 → 87.51   （域内代价）
τ³ Valid:                 98.86% → 86.53% （结构鲁棒性代价）
BFCL Functional E2E:      15.50 → 14.47   （语义代价）
```

> Length is trainable; correctness is not obtained by naive depth reweighting.

## 5. 主失败分析：EXEC_ACTION 边界（跨三基准合并）

**主导的跨域失败不是语法或工具绑定，而是 retrieval–action 边界上的语义接地。**
（dominant *observed* failure，不写"唯一语义问题"）

| 基准 | 参考侧 | 模型侧 | 证据 |
|---|---|---|---|
| τ³ | 96%（14,229/14,834）参考动作 oracle 降级为 EXEC_ACTION | E5C 全系产出率 ~0（SD 4,465 个预测中 1 个） | op 分布分析 |
| BFCL | multi_turn/action 类期望 EXEC_ACTION | 模型选 retrieval 类 | 关系判定 INCOMPATIBLE 8,008 主项 |
| AgentBoard tool-operation | 状态变更类工具调用 | EXEC_ACTION 23.9%（17/71），FETCH 仍压制（49） | 首测 op 分布 |

训练语料含 1,649 例 EXEC_ACTION（~3.5%）——非零样本问题，而是**条件映射不迁移**：
触发 EXEC_ACTION 的域内指令模式不覆盖外部基准的措辞。AgentBoard 首测中 env-visible
schema 使 EXEC_ACTION 使用率从 ~0% 升至 23.9%，证明 schema 部分缓解但未解决。

## 6. Binder 上界：失败定位到前端

τ³ oracle 语义计划 → binder → 具体工具的 P/R/F1 = **99.95%**。因此在架构上：

```text
Frontend semantic grounding（技能选择）    瓶颈
Backend binder（技能→工具绑定）           非主要瓶颈
```

> 显式的前端 IR 与后端绑定分层，使失败定位成为可能——这是 compiler 分层架构
> 的直接评测收益。

## 7. AgentBoard untouched 确认（正确口径）

**pre-registered（commit d59d00b，先于任何运行）、untouched、351 条、one-shot、
离线编译协议。** 结果：

| Task | n | Parse% | Valid% | Pred/T | EXEC_ACTION% |
|---|---:|---:|---:|---:|---:|
| tool-query | 60 | 98.3 | 78.3 | 1.32 | 2.5 |
| tool-operation | 40 | 95.0 | 87.5 | 1.77 | 23.9 |
| webshop | 251 | 98.4 | 96.0 | 0.96 | 0.0 |
| **合计** | **351** | **97.7** | **91.7** | 1.12 | — |

- 本节标题与结论一律称 **offline compiler confirmation**：证明的是第三个基准家族上的
  TaskIR compile-side structural generalization。
- **不得写** AgentBoard performance / task success / beats baseline——无交互执行环境，
  official progress rate / success rate = **UNSCORED-OFFLINE**。
- 官方 GPT-4/GPT-3.5 SR 仅作 non-comparable literature reference（文字引用，
  不与我们的数字同图对比）。
- webshop pred/T 0.96（单 SEARCH 近退化解）复现 τ³ 的浅计划先验。

## 8. 其余消融与基线（已执行部分）

- **A2 capability context**（E1 上，500 条分层 BFCL）：parse 97.8→99.2%，semantic
  8.6→8.8% —— 推理时给 schema 无效，训练时见过才有效（§4.1 已并入主结论）。
- **A4 数据质量**（v3→v3.1 参数保全修复）以语义审计报告呈现：suspect 64.6%→1.06%，
  参数保全 16.7%→98.9%（lifter-only 修复，未动 IR/ISA/split）。
- **ReAct/LLMCompiler 式基线**：协议已实现（`src/eval/react_baseline.py`），本轮未作为
  主表对照，如实标注 NOT RUN-in-this-snapshot。
- 名义成本（3.4% energy / 37.9% makespan，5 程序示例）：nominal analytical model，
  **非实测硬件能耗/延迟**，不作 headline，仅在调度章节作架构级分析。

## 9. 未执行项（如实清单）

```text
E3 7B LoRA                          NOT RUN
官方交互环境 AgentBoard（SR/PR）     NOT RUN（UNSCORED-OFFLINE）
真实硬件 cost profiling             NOT RUN
effect v0.2 实现（含 V7）            NOT RUN（设计提案）
LOOP region / 多程序链接            NOT RUN
```

## 10. 工件映射

| 结果 | 工件 |
|---|---|
| T7 internal 主表 | `docs/results-phase5b1-complete.md`; `experiments/phase5b1/` |
| 2×2 主表 + 补测 | `results/phase5c/*.json`; `docs/results-phase5c.md` |
| BFCL 语义判定 | `results/phase5c/bfcl_semantic.json`（冻结 audit 规则） |
| τ³ 归一化语义 | `results/phase5c/tau3_semantic.json` |
| AgentBoard 首测 | `results/agentboard/first_test.json`; `docs/agentboard-preregistration.md` |
| 快照哈希 | `experiments/paper_snapshot_v1.json` |
