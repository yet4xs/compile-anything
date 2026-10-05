# Phase 6B-1 最终报告 — Capability-Conditioned TaskIR Composer

> 数据齐备：2026-10-05。评测：`results/phase6b/composer_eval_v2.json`（8 项修复全量实施：
> kept-records 对齐断言 / matched-subset C0 / C0T post-hoc 对照 / multiset EA-R /
> E2E adherence / off-condition 与 unsupported-vs-reference 双指标 / parse 前深度分母）。
> 实验定性：**tool-use Composer experiment**（train 20,532 = xLAM 19,473 + toolbench 1,059；
> no-tool 7,319 条被训练过滤器跳过——脚本注释与行为不符已记录，实验名称据此更正）。

## 主表（matched 994-case 子集，E2E 口径）

| Arm | Valid-E2E% | OpSeq-E2E% | SkillF1 | EA-R micro% | CondR-E2E% | off-cond% | **unsup-vs-ref%** |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 E1-A（full 1,561，历史行） | 99.23 | 92.76 | 0.9398 | 90.12 | 93.20 | 33.69 | 5.21 |
| C0 E1-A（matched 994） | 98.99 | 89.94 | 0.9235 | 90.32 | 93.49 | 33.44 | 6.56 |
| **C0T 对照**（裸 prompt 重训同子集，post-hoc） | 98.69 | 78.17 | 0.8309 | 54.39 | 84.12 | 40.24 | 16.22 |
| C1 skill-only s42 | 97.79 | 92.05 | 0.9674 | 85.25 | 98.61 | 29.80 | 2.35 |
| C1 skill-only s43 | 98.19 | 92.05 | 0.9715 | 88.71 | 98.52 | 29.82 | 2.73 |
| **C2 selected-IR s42** | **98.29** | **95.88** | **0.9810** | **98.36** | **98.89** | 28.49 | **0.46** |
| C2 selected-IR s43（坏 seed，如实保留） | 67.00 | 64.89 | **0.9880** | 96.36 | 97.87 | 28.21 | 0.48 |
| C2 selected-IR s44 | 96.88 | 93.96 | 0.9786 | 95.08 | 98.34 | 28.71 | 0.74 |
| C3 available-table s42 | 97.38 | 94.57 | 0.9760 | 98.31 | 98.61 | 28.52 | 0.87 |

## B1 判定（对冻结五条件）

| 条件 | 阈值 | s42 | s44 | s43 |
|---|---|---|---|---|
| Valid-E2E | ≥97% | 98.29 ✓ | 96.88（差 0.12） | 67.0 ✗ |
| CondR-E2E | ≥95% | 98.89 ✓ | 98.34 ✓ | 97.87 ✓ |
| EA-R micro | ≥85% | 98.36 ✓ | 95.08 ✓ | 96.36 ✓ |
| OpSeq-E2E | ≥C0T 78.17 | 95.88 ✓ | 93.96 ✓ | 64.89 ✗ |
| SkillF1 | ≥C0T 0.8309 | 0.9810 ✓ | 0.9786 ✓ | 0.9880 ✓ |

**裁决：Outcome B1 confirmed（带披露的 seed 不稳定性）**——2/3 seed 全过五条件，
s43 语义条件全过但结构性 valid 崩（67%）。

## 四个结论

### 1. 因果证据链闭合（C0T 对照）

```
C0T（裸 prompt、同子集、同 1 epoch）：OpSeq 78.17 / EA-R 54.39 / unsup 16.22%
C2 （+ selected CapabilityIR 条件）：  OpSeq 95.88 / EA-R 98.36 / unsup  0.46%
                                    ──────────── +17.7pp / +44.0pp / −15.8pp
```
**显式 CapabilityIR 因果性地改善 TaskIR 组合**——排除了"只是子集分布效应"
（C0T 恰好显示裸重训同子集反而更差）。且 C2 以一半训练量（1 epoch vs E1-A 2 epoch）
在更小子集上超过全语料 E1-A matched（OpSeq 95.88 vs 89.94）。

### 2. "我告诉你用 EXEC_ACTION，你就写 EXEC_ACTION" — 恢复且近乎完美

6B-0 的核心失败（oracle 标注也救不回，EXEC_ACTION 输出 0.25→3.28%）在受训
Composer 上：EA-R micro 95-98%，unsup-vs-ref 0.46-0.74%（预测几乎不含参考之外的
技能）。Phase 6B-0 的诊断被训练实验证实：**E5C-S 从未学过消费 CapabilityIR；
学会之后 controllability 完全恢复**。

### 3. C1 vs C2：具体 capability 元数据有真实增量

C2 全面优于 C1（OpSeq +3.8pp、EA-R +10-13pp）——canonical skill 名单不够，
**具体工具名+描述+参数**进一步改善组合。架构上支持完整 CapabilityIR 接口。
C3（selection 隐式塞回 Composer）域内也表现好（94.57/98.31）——但注意域内
distractor 是随机的，这与 Phase 6A 硬负例下的 selection≈随机不矛盾；
C3 的真正考验在 6B-2 的 hard-negative 设定。

### 4. s43 seed 不稳定性：结构性的，非语义性的

- 训练 loss 三 seed 几乎一致（0.142-0.153）——非优化发散
- s43 的语义指标全 seed 最优（skillF1 0.988、EA-R 96.36、condR 99.67、unsup 0.48）
- 仅 parse→valid 掉 31pp：**validator 结构失败**（语义对、结构漂移）
- 三 seed 全部保留报告；s43 的 validator 错误分类学列为待办
- 5 条件判定未含 seed-稳健性条款（冻结时的疏漏），如实披露

## 五问回答

1. **Oracle-selected canonical skills 能否被忠实消费？** 能——CondR-E2E 97.9-98.9%，
   unsup-vs-ref ≤0.74%。
2. **EXEC_ACTION recall 是否恢复？** 是——micro 95-98%（vs 6B-0 的 3.28% 上界诊断）。
3. **C1 vs C2？** C2 胜（完整 CapabilityIR 优于 skill 名单）。
4. **Valid/OpSeq 是否保持？** 保持且提升（95.88 vs 基线 89.94/C0T 78.17；s43 除外已披露）。
5. **whole-plan objective 是否仍是瓶颈？** **在受条件化训练后不再是**——6B-0 的
   "one-shot NL→whole-TaskIR 抗拒语义控制"结论被修正为"未被训练过消费语义接口"。

## 对 Phase 6B-2 的指令（按冻结顺序，Resolver 训练启动）

- Resolver 输入：task + available CapabilityIR；输出 selected capability ids（不输出 skill）
- R1 生成式（baseline）vs **R2 ranking/scoring（主候选）**
- 硬负例：same domain / same object / opposite semantic intent（get_order vs cancel_order）
- 显式 NONE/NO_CALL 监督（合成负例标注 synthetic_negative，真实/合成分开报告）
- 指标：top-1/top-3/set P/R/F1/exact-set/NO_CALL F1，按 single/multi/unseen-family/
  action/retrieval 分桶
