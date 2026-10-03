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

四格记号：A=E1-A（无schema无depth）、S=仅schema、D=仅depth、SD=双因素。标准公式：

```text
schema_main = ((S + SD) − (A + D)) / 2
depth_main  = ((D + SD) − (A + S)) / 2
interaction = SD − S − D + A
```

（freeze-fix 修正：初版交互公式用错了加减格排列，本节全部交互值已按上式重算。）

**Internal OpSeq%（matched 协议，§4）**

| | depth ✗ | depth ✓ | depth 主效应 |
|---|---:|---:|---:|
| schema ✗（A / D） | 92.76 | 87.51 | **−4.77** |
| schema ✓（S / SD） | 94.11 | 89.82 | **−4.77** |
| schema 主效应 | **+1.83** | **+1.83** | 交互 **+0.96** |

- schema 主效应 +1.83pp（有益但小），depth 主效应 −4.77pp（真实代价），交互 +0.96pp（轻微次可加）。
- 注意主效应在这个 2×2 设计里完全共线（各单元只有一个 seed），交互项吸收了所有单元间残差，
  多 seed 复现前只作描述性结论。

**τ³ Pred/T（统一协议：E1-A 已按 S/D/SD 协议归一化重跑，§5.2）**

| | depth ✗ | depth ✓ | depth 主效应 |
|---|---:|---:|---:|
| schema ✗（A / D） | 1.00 | 1.65 | **+0.325** |
| schema ✓（S / SD） | 1.75 | 1.75 | **+0.00** |
| schema 主效应 | **+0.425** | **+0.55** | 交互 **−0.65** |

- **典型饱和/负交互（−0.65）**：schema 已把计划长度推高后，depth reweighting 几乎没有额外收益。
  支持机制解释：训练时 capability 可见性与深计划相关，改变了无条件长度先验；
  过采样本身贡献有限。

**τ³ Semantic Recall%（统一协议，§5.2）**

| | depth ✗ | depth ✓ | depth 主效应 |
|---|---:|---:|---:|
| schema ✗（A / D） | 0.62 | 0.55 | −0.05 |
| schema ✓（S / SD） | 0.58 | 0.66 | +0.04 |
| schema 主效应 | −0.02 | **+0.055** | 交互 **+0.15** |

- 全部效应 ≤0.15pp，在 n=14,834 上属噪声量级：**schema 和 depth 对 τ³ 语义召回均无实效**。
  长度可训练，正确性不可 —— 两个因子都只动了"计划多长"，没动"计划对不对"。

**BFCL Functional E2E %（§6，统一分母 full+partial=4,541，主指标）**

| | depth ✗ | depth ✓ | depth 主效应 |
|---|---:|---:|---:|
| schema ✗（A / D） | 15.50 | 14.47 | **−0.62** |
| schema ✓（S / SD） | 15.86 | 15.66 | **−0.49** |
| schema 主效应 | **+0.77** | **+0.60** | 交互 **+0.84** |

**BFCL Functional | Valid %（条件诊断指标，§6）**

| | depth ✗ | depth ✓ | depth 主效应 |
|---|---:|---:|---:|
| schema ✗（A / D） | 16.83 | 15.23 | −1.04 |
| schema ✓（S / SD） | 18.19 | 17.71 | −0.48 |
| schema 主效应 | **+1.92** | **+2.48** | 交互 **+1.12** |

- 因子图景汇总：**schema** 对 internal 计划选择（+1.83pp）、深度（+0.425）、
  BFCL functional（E2E +0.77pp / 条件 +1.92pp）全部正向或中性 —— 唯一无代价因子；
  **depth reweighting** 对深度 +0.325 但对其他因变量全部负向
  （internal OpSeq −4.77pp、τ³ parse −12pp、BFCL functional E2E −0.62pp）——
  收益最窄、代价最宽的因子，且深度收益被 schema 饱和（交互 −0.65）。

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

> 已完成（`results/phase5c/tau3_semantic.json`，重打分脚本 `scripts/rescore_5c_tau3.py`，
> E5C-SD 初版打分因畸形 set-literal 参数触发 parser 原生 TypeError 崩溃，已改宽 except 重打分，
> 推理结果不变）。语义召回 = 预测语义技能命中 oracle 降级参考动作的比例（14,834 个参考动作）。

| Model | Valid% | Pred/T | **Semantic Recall** | 匹配/参考 | **Semantic Precision** | Seq Exact | Multi-Complete |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1-A | 98.94 | 1.77* | 0.62% | 90/14,834 | 1.99% | 0.36% | 0% |
| E5C-S | 96.74 | 1.75 | 0.58% | 81/14,834 | 1.82% | 0.29% | 0% |
| E5C-D | 86.53 | 1.65 | 0.55% | 71/14,834 | 1.69% | 0.27% | 0% |
| E5C-SD | 96.54 | 1.75 | 0.66% | 92/14,834 | 2.06% | 0.04% | 0% |

\* E1-A 此行来自 Phase 5B 的 2048 协议 preds（生成预算更长）；主表 512 预算下 E1-A pred/T=1.00。
S/D/SD 均为 512 预算，与主表一致。

**判定：情况 C（recall/precision 双平坦，甚至微降）——深度提升没有转化为语义正确性。**

1. **四组召回全部在 0.55~0.66% 持平**（最大差 0.11pp ≈ 21 个动作），而 pred/T +65~75%。
   多出来的动作不是正确的动作：precision 1.7~2.1%，即模型生成的每个 action 命中参考的概率仅 ~2%。
2. **multi-action 任务 0% 完全覆盖**（1,000+ 个多动作任务，四组全部为 0）。
3. 与 internal OpSeq 94% 对照：模型在训练域内技能选择良好，但 τ³ 域上语义技能选择几乎完全失败
   —— 0.6% 的召回不是"计划不够长"的问题，是**跨域 skill 选择本身未接地**。
   Under-planning 不是 τ³ 语义召回的约束瓶颈（把 E1-A 的生成预算放开到 1.77 后召回仍是 0.62%）。
4. **对 Phase 6 的直接含义**：合成更多深计划数据**不会**自动修复 τ³ 语义召回
   （S/D/SD 已证明长度可训练、正确性不可）。需要的是跨域 skill 接地机制
   （如 τ³ 域的 capability schema 在推理时的 grounding、或 skill 选择的对齐训练），
   这正是 §6 BFCL functional 要回答的问题。

### 5.1 失败模式解剖：EXEC_ACTION 边界（op 分布分析）

`scripts/analyze_5c_tau3_ops.py`（CPU 分析，已存 docs）：

| | 参考降级分布 | E5C-SD 预测分布 | E1-A 预测分布 |
|---|---|---|---|
| EXEC_ACTION | **14,229（96%）** | **1** | 10 |
| FETCH | 373 | 4,278（96%） | 2,227 |
| QUERY_DB | — | 168 | 10 |
| SEND | 9 | 7 | **2,235** |
| SAVE / SEARCH / 其他 | 232 | 12 | ~50 |

1. **召回失败的机制极其单一**：τ³ oracle 把几乎所有工具调用降级为 EXEC_ACTION，
   而模型把所有工具调用映射为 FETCH/QUERY_DB（SD）或 FETCH/SEND（E1）。
   EXEC_ACTION 产出率 ~0 是召回 0.6% 的直接原因——该指标在 EXEC_ACTION 上近乎二元。
2. **不是零样本问题**：EXEC_ACTION 在训练语料有 1,649 例（~3.5%），但训练中触发它的
   指令模式（域内措辞）不覆盖 τ³ 的表述。schema 训练还消灭了 E1 的 SEND 习惯
   （2,235→7），让预测进一步集中到 FETCH。
3. **与 BFCL 同构**：BFCL multi_turn 类（shell 操作）oracle 同样期望 EXEC_ACTION、
   模型同样选 retrieval 类（见 results-bfcl-ontology-analysis.md 失败模式 2）。
   **两个外部基准在同一个 skill 边界（action-execution）上失败。**
4. **指标设计警示（论文须如实报告）**：τ³ skill 级召回在 96% 参考为单一 skill 时
   近乎退化为"是否使用 EXEC_ACTION"的二元指标；一个只会输出 EXEC_ACTION 的退化解
   也能拿到 ~96% 召回。跨域语义接地的结论应以 BFCL 分级关系判定（§6）为主证据，
   τ³ 数字为辅证并附此警示。

### 5.2 E1-A 协议归一化重跑（freeze-fix Task 3）

> 初版 τ³ 表中 E1-A 用了 Phase 5B 旧 preds（生成预算更长，pred/T=1.77），
> 而 S/D/SD 用 5C 协议（max_new_tokens=512）——不同 inference 不应混入同一因子表。
> `scripts/rerun_e1a_tau3_normalized.py` 以与 S/D/SD 完全一致的协议
> （SYSTEM_PROMPT + 裸指令、input 2048、max_new_tokens 512、greedy、同 oracle）
> 重跑 E1-A，重写 `results/phase5c/tau3_semantic.json`，§3.5 因子表使用归一化值。

**归一化后的 τ³ 统一协议表：**

| Model | Valid% | Pred/T | Recall% | 匹配/参考 | Precision% | Seq% |
|---|---:|---:|---:|---:|---:|---:|
| E1-A（归一化） | 98.86 | **1.00** | 0.62 | 89/14,834 | 3.50 | 0.40 |
| E5C-S | 96.74 | 1.75 | 0.58 | 81/14,834 | 1.82 | 0.29 |
| E5C-D | 86.53 | 1.65 | 0.55 | 71/14,834 | 1.69 | 0.27 |
| E5C-SD | 96.54 | 1.75 | 0.66 | 92/14,834 | 2.06 | 0.04 |

- 归一化 E1-A（recall 0.62%、pred/T 1.00）与旧协议值（0.62%、1.77）在召回上几乎不变
  ——**τ³ 召回对生成预算不敏感，§5 的情况 C 判定在统一协议下稳固**。
- E1-A 的 precision 更高（3.50%）仅因其动作更少（分母小），不代表选择更准。
- 正式因子表（§3.5 τ³ 行）使用本表数值。

## 6. 补测 3：BFCL semantic（matched protocol）

> 已完成（`results/phase5c/bfcl_semantic.json`）。E1-A 复用冻结 preds（裸指令）；
> E5C-D 裸指令重推理；E5C-S/SD 用 BFCL 逐题 function schemas 拼入（`build_capability_prompt`）。
> 判定为冻结的 audit_bfcl_ontology.py 三级规则，oracle/crosswalk 未改。
> comparable = full+partial 且 parse+valid 的样本（各模型 3,959~4,313）。

| Model | Schema训练 | 深度 | Valid% | Strict% | Equivalent% | Functional\|Valid% | **Functional E2E%** |
|---|---|---|---:|---:|---:|---:|---:|
| E1-A | ✗ | ✗ | 92.4 | 10.95 | 14.20 | 16.83 | 15.50 |
| E5C-S | ✓ | ✗ | 87.6 | **11.01** | **15.94** | **18.19** | **15.86** |
| E5C-D | ✗ | ✓ | **95.0** | 8.86 | 12.38 | 15.23 | 14.47 |
| E5C-SD | ✓ | ✓ | 88.8 | 10.26 | 15.24 | 17.71 | 15.66 |

> **Functional E2E** = 命中数 / 全部 full+partial（4,541，统一分母），是端到端成功率，
> 为选型主指标。**Functional|Valid** = 命中数 / 其中 valid 的样本（3,959~4,313），
> 是条件诊断指标——它被 schema 模型较低的 Valid 放大，不能单独作为主结论。
> （freeze-fix：初版把条件值当主指标引用，已改。）

**结论：**

1. **Schema 假设成立但幅度小**：E2E 15.50→15.86（**+0.35pp，相对 +2.3%**）；
   条件口径 16.83→18.19（+1.36pp）。Strict/Eq 同向。
   这是 schema conditioning 改善跨域语义接地的**首个正证据**（此前 capability
   ablation 在 E1 上为 +0.2pp 无效——训练时见过 schema 才有效，符合 5B 的预言）。
   论文表述用：
   > small but consistent improvement in end-to-end BFCL semantic correctness,
   > while substantially changing planning behavior.
   不写成强 semantic breakthrough。
2. **深度课程损害 BFCL functional**：E2E 15.23（−1.03pp），strict 掉到 8.86。
3. **代价披露**：schema 推理使 BFCL Valid 降 ~4-5pp（92.4→87.6/88.8）——
   schema 促使模型生成更多/更长动作，validator 失败率上升。E2E 口径已把这一代价
   计入分母，因此 +0.35pp 是净效应。
4. 按 rep 分列：S 的条件提升主要来自 full 类（545→564）与 partial strict（159→156 持平，
   D 掉到 94）；E2E 排序 S > SD > A > D 与条件排序一致，选型结论不变。

## 7. 研究问题回答（终版）

| 问题 | 回答 |
|---|---|
| Q1 语料是否偏短计划？ | **是**。69.9% 单 action，≥4 action 仅 325 条（1.2%）。已冻结审计。 |
| Q2 Schema 条件化改善跨域语义接地？ | **BFCL 侧小幅一致成立**：E2E Functional 15.50→15.86（+0.35pp，相对 +2.3%），条件口径 +1.36pp —— 首个正证据，且与 5B 预言一致（推理时给 schema 无用、训练时见过才有效）。**τ³ 侧不成立**（召回四组持平）。论文表述 small but consistent，不写 breakthrough。 |
| Q3 深度课程改善多步规划？ | **长度上是**（pred/T +0.65，被 schema 饱和），**正确性上否**（τ³ 召回不动、BFCL functional −1.04pp），结构代价明确（internal OpSeq −4.78pp、τ³ parse −12pp）。**总体为负因子**。 |
| Q4 联合训练能否双收益且不伤 valid？ | matched 协议下 valid 全过 97%；深度与 schema 同享；functional +0.88pp 但低于 S 单独。SD 无优势。 |
| Q5 为什么跨域语义接地失败？ | **EXEC_ACTION 边界**：τ³ 96% 参考动作为 EXEC_ACTION，模型产出率 ~0（训练有 1,649 例但条件映射不迁移）；BFCL multi_turn 同构失败。 |

## 8. 冻结的选型规则（补测结果出来之前冻结，防止事后挑选）

1. **门槛条件**：internal matched-protocol Valid ≥ 97%
2. **主排序**：BFCL Functional Semantic 最大化
3. **次排序**：τ³ Grounded Semantic Recall + Count Ratio
4. 不能只因 Valid 高选模型

> 按此规则，当前 SD 的 internal matched Valid 若仍 ~95% 而 S 达到 ~98% 且
> BFCL/τ³ semantic 接近，则应选 E5C-S 而非 E5C-SD。

## 9. 决策（按 §8 冻结规则裁决，三补测齐备）

**逐条应用冻结规则（freeze-fix 后口径）：**

1. 门槛 internal matched Valid ≥ 97%：E1-A 99.23 ✓ / S 99.10 ✓ / D 97.82 ✓ / SD 99.23 ✓（全过）
2. 主排序 BFCL **Functional E2E**（统一分母 4,541）：**S 15.86** > SD 15.66 > E1-A 15.50 > D 14.47
3. 次排序 τ³ recall（四组无差别）/ pred/T：S 1.75 并列最高

**→ Phase 5C 定版模型：E5C-S（schema-conditioned，部署时带 capability schema 推理）。**

完整定版画像（matched protocol）：

| 维度 | 数值 | 备注 |
|---|---|---|
| internal Valid / OpSeq | 99.10 / **94.11** | OpSeq 四格最高 |
| BFCL Valid / Functional E2E / 条件 | 87.6 / **15.86** / 18.19 | E2E 四格最高（基线 15.50，+0.35pp） |
| τ³ Valid / Pred/T | 96.7 / 1.75 | 深度恢复 75%；召回与基线无差别 |
| 部署要求 | 推理时必须提供 capability schema | 无 schema 时计划选择崩塌（OpSeq 44%） |

后续决定：

1. **Phase 5C 就此冻结**（manifest 附最终结果指针）；depth reweighting 从主线移除
   （负因子），其"长度可训练"的机制发现保留。
2. **Phase 6 靶心（由 Q5 直接决定）**：跨域 skill 接地——EXEC_ACTION 边界的
   条件映射迁移（多样化 action 类指令合成、负例/irrelevance、skill 选择对齐），
   辅以受限的深计划合成（满足组合监督，而非单纯过采样）。
3. **AgentBoard untouched 首测**：用 E5C-S（schema 部署协议），一次性，结果无论好坏照登。

## 10. 工件清单

- `results/phase5c/evaluation_summary.json` — 2×2 主表（4 模型 × 3 基准）
- `results/phase5c/internal_schema_eval.json` — §4 补测
- `results/phase5c/tau3_semantic.json` — §5 补测
- `results/phase5c/bfcl_semantic.json` — §6 补测
- `runs/phase5c/{e5c_s,e5c_d,e5c_sd}/final` — 三个 LoRA adapter
- `scripts/eval_5c_schema.py`、`scripts/eval_5c_tau3_semantic.py`、`scripts/eval_5c_bfcl_semantic.py` — 补测脚本
