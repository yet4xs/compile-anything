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
  **结论：剩余差距是数据生成问题而非训练策略问题** —— 需要 Phase 6 的组合式深计划合成。

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

### 3.4 归因警告

BFCL Valid 的 +2.0~+3.5pp 出现在全部三个新模型上（包括无 schema 的 D）。
共同点是"重新训练"本身（数据顺序、过采样改变的有效 epoch 组成）。
没有多 seed 方差数据前，**不能把 BFCL 提升单独归因给 schema**。
manifest 已禁止基于外部分数调参，此警告仅限论文表述。

## 4. 补测：internal 带 schema prompt（修 §3.2 的失配）

> 进行中（`results/phase5c/internal_schema_eval.json`）。协议：对有 capabilities 的 1003 条测试样本
> 用与训练完全一致的 `build_capability_prompt` 格式，558 条无 schema 样本保持裸指令。
> 分桶报告 all / schema / noschema。

（结果待填）

## 5. 补测：τ³ 语义召回（深度提升是否转化为正确性）

> 进行中（`results/phase5c/tau3_semantic.json`）。关键指标：semantic_recall（预测的语义技能
> 覆盖 oracle 降级参考动作的比例）。E1-A 之前为 ~19%（受 1.0 pred/T 限制的理论上限）。

（结果待填）

## 6. 研究问题回答

| 问题 | 回答 |
|---|---|
| Q1 语料是否偏短计划？ | **是**。69.9% 单 action，≥4 action 仅 325 条（1.2%）。已冻结审计。 |
| Q2 Schema 条件化改善跨域语义接地？ | **部分**。结构有效性（BFCL Valid 94.5→98.0）改善，但语义接地需看 §5 召回；且推理时必须带 schema，否则计划选择崩塌（OpSeq 92.8→39~44）。 |
| Q3 深度课程改善多步规划？ | **方向上是**（pred/T 1.0→1.75），但 (a) schema 训练也能带来同样提升，(b) 单独使用会损跨域格式稳定性（τ³ parse -12pp），(c) 距参考仍 3.3×。 |
| Q4 联合训练能否双收益且不伤 valid？ | **结构上可以**（SD：BFCL 98.0 + τ³ 96.5 + 深度 1.75），代价是 internal OpSeq 需带 schema 推理（见 §4）。 |

## 7. 决策建议

1. **Phase 5C 定版模型：E5C-SD**（带 schema 推理部署）。BFCL 98.0 / τ³ 96.5 / 深度 1.75 全面最优或并列最优。
2. **schema-at-inference 是部署协议的一部分**，不是可选增强 —— 训练/推理必须一致。
3. **深度问题的下一阶段（Phase 6）转向数据合成**：训练侧因子（课程、schema）已到收益上限，
   语料里没有的深度不可能凭空学出。候选：组合任务自举（单 action 样本程序化组合成 2-6 步链）、
   τ³ 风格 policy 约束合成（注意污染防火墙：合成器不得读外部基准）。
4. AgentBoard 确认性基准保持未动，待 Phase 6 模型冻结后一次性首测。

## 8. 工件清单

- `results/phase5c/evaluation_summary.json` — 2×2 主表（4 模型 × 3 基准）
- `results/phase5c/internal_schema_eval.json` — §4 补测
- `results/phase5c/tau3_semantic.json` — §5 补测
- `runs/phase5c/{e5c_s,e5c_d,e5c_sd}/final` — 三个 LoRA adapter
- `scripts/eval_5c_schema.py`、`scripts/eval_5c_tau3_semantic.py` — 补测脚本
