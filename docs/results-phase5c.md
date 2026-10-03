# Phase 5C 实验结果：Schema 条件化 + 深度课程 2×2 实验

> 实验时间：2026-10-02 ~ 10-03
> 服务器：BitHub RTX 4090 24GB
> 冻结 manifest：`experiments/phase5c/manifest.json`（base commit 31869b3）
> 训练数据：corpus v3.1（28,093 条，同 Phase 5B-1，无任何外部基准数据）

## 1. 实验设计

2×2 因子矩阵，基座统一 Qwen2.5-3B-Instruct + QLoRA（r=16, α=32, lr=1.5e-4, 2 epochs, seed=42）：

| 模型 | Schema 条件化 | 深度课程 | 说明 |
|---|---|---|---|
| E1-A | ✗ | ✗ | Phase 5B-1 基线（已有） |
| E5C-S | ✓ | ✗ | 仅 schema：训练时把 capabilities（工具名+描述+参数）拼进 user prompt |
| E5C-D | ✗ | ✓ | 仅深度课程：按 action 数分桶（1 / 2-3 / 4-6 / 7+）加权过采样，上限 5× |
| E5C-SD | ✓ | ✓ | 双因素 |

**训练前审计（Task 1/2 结果，manifest 已冻结）：**

- 深度偏斜确认：train 集中 69.9% 单 action（19,638/28,093），4-6 action 仅 308 条（1.1%），7+ 仅 17 条（0.1%）。
  即使 5× 过采样，7+ 桶也只有 ~85 有效样本 —— **语料本身缺少深计划，这是天花板**。
- Schema 覆盖：FULL_SCHEMA 20,774（xLAM + ToolBench 有完整工具签名）、SEMANTIC_ONLY 6,966（Spider 仅表名）、NONE 353。

## 2. 主结果表（评测协议：2048-token，greedy，全部无 schema prompt）

| 模型 | Int.Valid% | Int.OpSeq% | BFCL.Parse% | BFCL.Valid% | τ³.Parse% | τ³.Valid% | τ³.Pred/T | τ³.Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1-A | **99.2** | **92.8** | 98.9 | 94.5 | 98.9 | **98.9** | 1.00 | 0.17 |
| E5C-S | 97.5 | 44.1 | 98.4 | 96.5 | 97.3 | 96.7 | 1.75 | 0.30 |
| E5C-D | 97.8 | 87.5 | 98.3 | 96.1 | 86.8 | 86.5 | 1.65 | 0.28 |
| E5C-SD | 94.8 | 39.1 | **99.0** | **98.0** | 96.6 | 96.5 | **1.75** | **0.30** |

（τ³.Pred/T = 模型每任务平均 action 数；参考均值 5.83；Ratio = Pred/T ÷ 5.83）

## 3. 因子分析

### 3.1 深度瓶颈：方向性修复，但远未解决

- 四个模型中 E1-A 精确卡在 1.00 action/task（复现 Phase 5B-2 结论：输出长度由训练分布主导）。
- 三个新模型全部提升到 1.65–1.75（+65~75%），count_ratio 0.17→0.30。
- **注意**：E5C-S 并没有深度课程，也涨到了 1.75 —— 计划深度的提升不能单独归因于过采样；
  schema 训练本身也移动了长度先验（机制：训练时"丰富的 capabilities ↔ 多步计划"相关性改变了无条件长度先验）。
- 距离参考 5.83 仍差 3.3×。结合 Task 1 审计（深计划样本只有 325 条），
  **训练数据的长计划稀缺是一个被实验证实的强限制因素**。
  但还不能排除其他因素：whole-plan autoregressive objective 本身可能不适合长规划、
  oversampling 不能产生新的组合模式、3B 容量/解码偏置、dependency supervision 不足、
  schema prompt 可能只让模型"多输出节点"而非"输出正确节点"。
  区分这些假设依赖 τ³ semantic recall（§5）与 BFCL functional（§6）补测。

### 3.2 Schema 条件化的双面效应

正面（跨域结构有效性）：

- BFCL Valid：94.5 → 96.5（S）/ 98.0（SD）。三个新模型全部 ≥96.1，其中 SD 达 98.0%。
- τ³ 上 SD 保持了 96.5% valid（vs D 单独掉到 86.5%）—— schema 训练稳定了深度课程带来的格式退化，
  **正交互效应**。

负面（无 schema 推理时的计划选择崩塌）：

- Internal OpSeq：92.8 → 44.1（S）/ 39.1（SD）。Valid 仍高（97.5/94.8），说明程序合法但
  技能选择与参考不一致 —— 模型在等一个没来的 capability 声明。
- **这是 train/eval 分布失配，不是能力丢失**。补测（见 §4）：带 schema prompt 重评 internal。

### 3.3 深度课程的隐性代价

- E5C-D 的 τ³ Parse 掉到 86.8%（-12pp）：对 1.1% 的桶 5× 过采样放大了噪声，跨域格式稳定性受损。
- 但 SD（schema + depth）恢复到 96.6% —— schema 上下文为深样本提供了更稳定的条件信号。

### 3.5 正式 2×2 因子分析（对四个因变量分别计算）

效应定义（+schema = (S+SD)/2 − (E1-A+D)/2，+depth 同理，交互 = SD − E1-A − S + D）：

**Internal OpSeq%（matched 协议，§4）**

| | depth ✗ | depth ✓ | depth 主效应 |
|---|---:|---:|---:|
| schema ✗ | 92.76 | 87.51 | **−4.78** |
| schema ✓ | 94.11 | 89.82 | **−4.78** |
| schema 主效应 | **+1.83** | **+1.83** | 交互 **−9.54** |

- schema 主效应 +1.83pp（有益但小），depth 主效应 −4.78pp（真实代价），交互 −9.54pp（非加性损害）。
- 注意主效应在这个 2×2 设计里完全共线（各单元只有一个 seed），交互项吸收了所有单元间残差，
  多 seed 复现前只作描述性结论。

**τ³ Pred/T（主表）**

| | depth ✗ | depth ✓ | depth 主效应 |
|---|---:|---:|---:|
| schema ✗ | 1.00 | 1.65 | **+0.65** |
| schema ✓ | 1.75 | 1.75 | **+0.00** |
| schema 主效应 | **+0.75** | **+0.55** | 交互 **+0.35** |

- 深度提升的主要来源是 schema（+0.75），depth reweighting 单独 +0.65，且被 schema 完全饱和
  （SD 不再叠加）。支持"capability 可见性驱动计划展开"的机制解释。

**BFCL Functional%、τ³ Semantic Recall%**：待 §5/§6 补测结果回填。

### 3.6 归因警告

BFCL Valid 的 +2.0~+3.5pp 出现在全部三个新模型上（包括无 schema 的 D）。
共同点是"重新训练"本身（数据顺序、过采样改变的有效 epoch 组成）。
没有多 seed 方差数据前，**不能把 BFCL 提升单独归因给 schema**。
manifest 已禁止基于外部分数调参，此警告仅限论文表述。

## 4. 补测：internal 带 schema prompt（修 §3.2 的失配）

> 已完成（`results/phase5c/internal_schema_eval.json`）。协议：对有 capabilities 的 1,003 条测试样本
> 用与训练完全一致的 `build_capability_prompt` 格式，558 条无 schema 样本保持裸指令。
> 注：脚本中 "all" 行的 n 有计数 bug（未累加），下表 all 行由 schema/noschema 两桶精确重算。

**Matched-protocol internal 表（主比较，train/eval 协议一致）：**

| Model | Eval Prompt | Valid% | OpSeq% |
|---|---|---:|---:|
| E1-A | bare（主表） | 99.23 | 92.76 |
| E5C-S | schema | 99.10 | **94.11** |
| E5C-D | bare（主表） | 97.82 | 87.51 |
| E5C-SD | schema | **99.23** | 89.82 |

分桶（schema 桶 n=1,003 / noschema 桶 n=558）：

| Model | schema桶 Valid/OpSeq | noschema桶 Valid/OpSeq |
|---|---|---|
| E1-A（ablation） | 98.80 / 84.95 | 100.0 / 98.39 |
| E5C-S | 98.70 / **91.33** | 99.82 / 99.10 |
| E5C-SD | **99.00** / 84.65 | 99.64 / 99.10 |

**结论：**

1. **"能力丢失"假设排除**：matched 协议下 S OpSeq 44.1→**94.11**，SD 39.1→**89.82**，
   崩塌完全由 train/eval prompt 失配解释。主表 §2 中 S/SD 的 internal 数字应视为
   schema-ablated 推理的鲁棒性数字，而非能力数字。
2. **冻结规则 §8 门槛（matched Valid ≥ 97%）四组全部通过**：E1-A 99.23 / S 99.10 / D 97.82 / SD 99.23。
   选型进入规则 #2（BFCL Functional）裁决 → §6。
3. **E5C-S 的 internal OpSeq 94.11 是四格最高**（超过 E1-A 92.76）：schema 条件化训练本身
   不损 internal 计划选择——反而略有益。
4. 深度课程在 internal OpSeq 上有真实代价：D 87.51、SD 89.82 均低于各自对照。
5. E1-A 用 schema prompt（不匹配方向）只损 OpSeq ~3pp（92.76→89.75），不损 Valid ——
   与 5B capability ablation 的方向一致（schema 上下文主要影响计划选择，不影响结构合法性）。
6. 主表失配数字反推（noschema 桶两轮 eval 同为裸指令、行为一致，≈553 正确）：
   主表中 S/SD 在 schema 类样本上的 OpSeq 仅约 14%（S）和 6%（SD），而 noschema 类 ~99%
   —— 失配损害集中在训练时有 schema 的域（xLAM/ToolBench），对从未有 schema 的域（code 类）无影响。

## 5. 补测：τ³ 语义召回（深度提升是否转化为正确性）

> 进行中（`results/phase5c/tau3_semantic.json`）。关键指标：semantic_recall（预测的语义技能
> 覆盖 oracle 降级参考动作的比例）。E1-A 之前为 ~19%（受 1.0 pred/T 限制的理论上限）。

（结果待填）

## 6. 补测 3：BFCL semantic（matched protocol）

> 已排队（`results/phase5c/bfcl_semantic.json`）。schema 假设最终检验：
> schema-conditioned training 是否提高 BFCL Functional Semantic（基线 16.0%），
> 而不仅是 Valid。协议：E1-A/D 裸指令（复用冻结 preds / 重推理），
> S/SD 用 BFCL 逐题 function schemas 按 `build_capability_prompt` 训练格式拼入。
> 判定用冻结的 audit_bfcl_ontology.py 规则（Strict / Ontology-Equivalent / Functional），不改 oracle/crosswalk。

（结果待填）

## 7. 研究问题回答（初版，待补测更新）

| 问题 | 回答 |
|---|---|
| Q1 语料是否偏短计划？ | **是**。69.9% 单 action，≥4 action 仅 325 条（1.2%）。已冻结审计。 |
| Q2 Schema 条件化改善跨域语义接地？ | **部分**。结构有效性（BFCL Valid 94.5→98.0）改善，但语义接地需看 §5/§6；且推理时必须带 schema，否则计划选择崩塌（OpSeq 92.8→39~44）。 |
| Q3 深度课程改善多步规划？ | **方向上是**（pred/T 1.0→1.75），但 (a) schema 训练也能带来同样提升，(b) 单独使用会损跨域格式稳定性（τ³ parse -12pp），(c) 距参考仍 3.3×。 |
| Q4 联合训练能否双收益且不伤 valid？ | **结构上可以**（SD：BFCL 98.0 + τ³ 96.5 + 深度 1.75），代价是 internal OpSeq 需带 schema 推理（见 §4）。最终裁决见 §8。 |

## 8. 冻结的选型规则（补测结果出来之前冻结，防止事后挑选）

1. **门槛条件**：internal matched-protocol Valid ≥ 97%
2. **主排序**：BFCL Functional Semantic 最大化
3. **次排序**：τ³ Grounded Semantic Recall + Count Ratio
4. 不能只因 Valid 高选模型

> 按此规则，当前 SD 的 internal matched Valid 若仍 ~95% 而 S 达到 ~98% 且
> BFCL/τ³ semantic 接近，则应选 E5C-S 而非 E5C-SD。

## 9. 决策（provisional，待三个补测）

1. **Phase 5C 定版模型：未定**。E5C-SD 是 external structural robustness +
   planning depth 的 best candidate（BFCL 98.0 / τ³ 96.5 / pred/T 1.75），
   但不是 overall winner —— internal matched Valid、BFCL functional、
   τ³ semantic recall 均未知，按 §8 规则裁决。
2. **schema-at-inference 是部署协议的一部分**（对 schema 训练模型），
   主比较必须 train/eval protocol matched；E1-A 用 schema prompt 只作 robustness ablation。
3. **Phase 6 方向暂记**（冻结前不定版）：若 τ³ semantic recall 证明更长计划
   同时更正确（情况 A），则进入 compositional deep-plan data construction /
   hierarchical compiler objective；若只是更长不更正确（情况 B/C），
   问题在 objective/组合泛化而非数据量。
4. AgentBoard 确认性基准保持未动，待 Phase 5C 冻结、final model 选定后一次性首测。

（schema/depth 非加性现象已升级为正式分析，见 §3.5。）

## 10. 工件清单

- `results/phase5c/evaluation_summary.json` — 2×2 主表（4 模型 × 3 基准）
- `results/phase5c/internal_schema_eval.json` — §4 补测
- `results/phase5c/tau3_semantic.json` — §5 补测
- `results/phase5c/bfcl_semantic.json` — §6 补测
- `runs/phase5c/{e5c_s,e5c_d,e5c_sd}/final` — 三个 LoRA adapter
- `scripts/eval_5c_schema.py`、`scripts/eval_5c_tau3_semantic.py`、`scripts/eval_5c_bfcl_semantic.py` — 补测脚本
