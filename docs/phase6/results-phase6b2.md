# Phase 6B-2 最终报告 — Task-Conditioned Capability Resolution

> 数据齐备：2026-10-06。评测：`results/phase6b/resolver_eval_v2.json`（matched common 15-候选集，
> 确定性构造；teacher-forced `logP(relevant)−logP(irrelevant)` 打分，score>0 选择，零阈值调节）。
> 数据审计：`results/phase6b/resolver_data_audit.json`。协议：`experiments/phase6b/resolver_manifest.json`（训练前冻结）。

## 数据审计（评测前）

- **R1 gold 可见率 100%**（train/test 均无隐藏 gold，无需重建）
- 有效训练分布（实际 50k）：relevant 13,676 / irrelevant 35,172 / NONE 1,152（pos:neg=1:2.57，NONE 2.30%——构造意图 10% 被截断稀释，如实记录）
- 合成 NONE 碰撞率 33%（150 中 49）；**clean 101 条为 NONE 主指标**；全部标注 synthetic refusal diagnostic
- **same_object_opposite_intent 负例 n=0**：v4 distractor 池不含同对象反意图工具对——headline 切片不可测（数据缺口，如实披露）

## 主表（matched 15-候选集）

| Model | Set P | Set R | **Set F1** | Exact Set | Top-1 | Top-3 | Pairwise-Empty NONE F1 | **Explicit NONE Acc** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 generative s42 | 0.9914 | 0.9914 | 0.9913 | **0.9899** | 0.9909 | 0.992 | 0.0 | **0.0** |
| R2 scorer s42 | 0.9960 | 0.9995 | 0.9970 | 0.9909 | 0.999 | 1.0 | 0.0 | 0.9901 |
| R2 scorer s43 | 0.9871 | 1.0 | 0.9914 | 0.9738 | 0.998 | 1.0 | 0.0 | 0.7129 |
| R2 scorer s44 | 0.9945 | 0.998 | 0.9953 | 0.9879 | 0.998 | 1.0 | 0.0 | 1.0 |
| **R2 mean±std** | — | — | **0.9946±0.0023** | 0.9842±0.0075 | 0.9983±0.0005 | 1.0 | 0.0 | 0.9010±0.1274 |

## 硬负例表（FPR = 被误选为 relevant 的比例）

| Negative Type | n | FPR（s42/s43/s44） | Mean relevance score |
|---|---:|---|---:|
| same_object_opposite_intent | **0** | **不可测（数据缺口）** | — |
| same_family_wrong_object | 3,147 | 0.0% / 0.13% / 0.0% | −13.7 / −10.9 / −13.1 |
| same_domain_similar_name | 28 | 0% / 0% / 0% | −13.1 / −9.7 / −12.1 |
| random_cross_domain | 10,553 | 0.08% / 0.21% / 0.09% | −14.1 / −10.2 / −12.9 |

## 六问回答

1. **Resolver 远超 0.20 了吗？** 是——0.99+（set F1），且两个 objective 都达到：
   Phase 6A 的 Probe C 随机水平被 corpus v4 的 selected-id 监督彻底解决。
   **限定**：matched 候选集负例以 random 为主（71%），难度低于 6A 硬负例设定。
2. **Ranking 优于 generative 吗？** 选择能力打平（0.9946 vs 0.9913）；**拒绝能力是本质分离**：
   explicit NONE 0.90±0.13 vs **0.0**（R1 无此监督则不涌现，与 6A 结论一致且现在有了正面对照）。
   冻结三条件按字面未全过（exact 均值被 s43 拖至 0.9842<0.9899；pairwise-NONE 平局）——
   如实记录；机制层结论：**R2 的优势在 refusal，不在 selection**。
3. **Multi-capability 可学？** 是（multi slice F1 0.996-0.998，n=170；单/多无显著差）。
4. **硬负例仍是主要失败？** 可测三类 FPR≈0；**最难类型（同对象反意图）在数据中不存在**——
   无法回答，Phase 6C 需 deliberately-constructed 对映工具负例池（get_X↔cancel_X 类）。
5. **显式 NO_CALL 产生有效拒绝？** 分裂结论：**显式生成机制有效**（R2 0.99-1.0）；
   **pairwise-empty 部署机制全体失败**（F1=0：外域表上总有候选 score>0）。
   部署指令：refusal 必须经显式 full-table NONE 门控，不能靠逐对阈值空集。
6. **Unseen family 泛化？** n=7 过小不 headline（F1 1.0）；v4 split 非家族留出，
   真正的家族 OOD 证据仍在 Phase 6A 的 142-family 测试。

## 过程修复存档（不掩盖）

- R1 parser 大小写 bug（o.upper() vs c\d+）导致首评全零——修复后 0.9913
- 补丁 heredoc 转义事故（\b 变 \x08 退格符）致 R1 top1/top3 首次重评为 0——字符类正则替换修复
- scoring OOM（全序列 logits）→ num_logits_to_keep=1
- R2 训练 OOM（batch16 logits 9.9GB）→ batch8/accum2

## 结论与 6B-3 前置条件

```text
Phase 6A: selection ≈ 0.20（随机）        → Phase 6B-2: 0.99+（选定 id 监督 + 真实候选表）
拒绝：无监督不涌现（6A: 0.0）             → 显式监督后 R2 0.90±0.13（R1 对照仍 0.0）
```

三段式前端（Canonicalizer 0.917 / Resolver 0.9946 / Composer 0.981-0.9643）的三个组件
现在各自有受控证据。**6B-3（集成）的全部前置满足**，但集成时必须：
(i) Composer 用 C2_s42/s44（s43 结构不稳已披露）；(ii) refusal 用显式 NONE 门控；
(iii) oracle 分解表（6B 设计 §13）量化各层误差贡献。
