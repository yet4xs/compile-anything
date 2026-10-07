# Phase 6B-2.1 — Refusal Gate Freeze（部署验证，无训练）

> 评测：`results/phase6b/refusal_gate_freeze.json`（994 positive + 101 clean synthetic NONE，
> 训练同款 full-table prompt，禁止改 prompt/threshold）。基线 `d51ef97`。

## Task 2 断言结果（先说，因为它影响 6B-2 的数字）

```text
tok("relevant")   = [97573]           单 token
tok("irrelevant") = [404, 97573]      双 token（"ir"+"relevant"）
```

**断言失败**：v2 评测的 next-token 打分不是完整标签似然（对比的是 P(relevant) vs P(ir)）。
按冻结指示：改为真序列似然（sum_t logP(label_t | prefix)）**重评 R2 选择指标**（不重训），
**重打分结果**：序列似然下 R2 mean±std = setF1 0.9947±0.0024 / exact 0.9846±0.0078 /
top1 0.9977±0.0009（首 token 近似 0.9946/0.9842/0.9983）——差异在第三位小数，
排序近似几乎等价，6B-2 选择结论在修正口径下**确认不变**。R1 0.9913/0.9899 不受影响。

## Task 1 门控双侧验证——**灾难性失败（负结果）**

| Seed | 正例误拒绝率 | 正例接受率 | NONE 召回 | NONE 精确 | NONE F1 |
|---|---:|---:|---:|---:|---:|
| s42 | **99.3%** | 0.7% | 1.000 | 0.093 | 0.170 |
| s43 | 77.6% | 22.4% | 0.802 | 0.095 | 0.170 |
| s44 | **97.6%** | 2.4% | 0.960 | 0.091 | 0.166 |

**显式 NONE 门控在正例侧几乎全拒绝**。6B-2 报告的 explicit NONE accuracy 0.90±0.13
只测了负例侧——部署检查正是为此设计，结论：**该门控不可部署**。

### 根因：格式捷径（format shortcut）

R2 训练数据中，NONE 监督（1,152 条）**只以 full-table prompt 格式**出现；pairwise
relevant/irrelevant 监督只以单 capability 格式出现。模型学到的是：

```text
full-table prompt  ⇒  输出 NONE（格式条件反射，非语义拒绝）
pairwise prompt    ⇒  relevant/irrelevant
```

正例侧给 full-table prompt ⇒ 99% 说 NONE。这不是拒绝能力，是**监督格式混淆**。

## 汇总：两个拒绝机制均不可部署

| 机制 | 负例侧 | 正例侧 | 判定 |
|---|---|---|---|
| pairwise-empty（score>0 全空） | 有 false call（6B-2 F1=0） | — | ✗ |
| 显式 full-table NONE 门控 | 0.80-1.00 召回 | **误拒绝 78-99%** | ✗ |

**Phase 6B-2 的 refusal 结论修正**：R2 的"explicit NONE 0.90"是格式捷径产物，
不构成语义拒绝能力。修正表述：

> Neither refusal mechanism is deployable as trained: pairwise-empty never
> triggers on foreign tables, and the explicit gate refuses virtually all
> positive tasks due to a supervision-format shortcut (NONE supervision was
> only ever presented in full-table format).

## 对 6B-3 的直接影响（按冻结顺序执行）

6B-2.1 **未通过**部署验证 → 6B-3 的 Learned Resolver 链条**不接 gate**：
- I3/I4 使用 pairwise score>0 选择（序列似然版），gate 缺席如实记录
- 无 refusal 的 modular 前端在正任务域内完整可测（本次集成目标）
- refusal 修复属于监督设计（格式平衡的 NONE：pairwise 格式也出 NONE 负例），
  需新训练 → **Phase 6C 候选项**，不阻塞 6B-3 正任务集成

## Deployment 措辞冻结（Task 3）

> R1 and R2 both solve capability selection on the current candidate
> distribution; R2 is selected for modular integration because it
> additionally supports explicitly supervised refusal.（该 refusal 优势
> 现已被 gate 检查降级为"可被监督但当前实现有格式捷径"）

主链 s42（预定义 primary）/ 复制 s44，不因结果选 seed。
