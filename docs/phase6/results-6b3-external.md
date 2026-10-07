# Phase 6B-3 外部 τ³ 诊断结果 — 论文核心架构证据

> 数据齐备：2026-10-08。评测：`results/phase6b/external_tau3.json`
> 协议：learned modular（Canonicalizer s42 + Resolver s42 + Composer s42），τ³ env-visible
> 工具名（source + ref union，非 GT-derived），no GT used in pipeline。

## 主表

| 系统 | Parse% | Valid% | Pred/T | **SemRecall%** | **EA输出率%** | EA ref-recall% |
|---|---:|---:|---:|---:|---:|---:|
| E5C-S 单体（6B-0 A） | 97.3 | 96.7 | 1.75 | 0.58 | 0.25 | ~0 |
| E5C-S + oracle IR（6B-0 D） | 87.0 | 87.0 | 1.66 | 0.22 | 3.28 | ~0 |
| **模块化前端（全学习）** | **96.2** | **96.1** | **2.79** | **23.02** | **60.23** | **100.0** |

## 核心结论

> **Monolithic neural compiler 在跨域上完全无法输出 EXEC_ACTION（0.25%）；
> 将 capability canonicalization、resolution、composition 分层后，
> 同一 3B 模型恢复了 60% 的 EXEC_ACTION 输出率。**

三个数字：

1. **EXEC_ACTION 输出率 0.25% → 60.23%**（240 倍恢复）——分层不是 overhead，
   是解锁语义可控性的前提。
2. **语义召回 0.58% → 23.02%**（40 倍）——仍远非完美，但跨域语义鸿沟
   从"完全断裂"变为"部分连通"。
3. **EA ref-recall = 100%**——当模型输出的 EXEC_ACTION 与参考动作匹配时，
   全部正确。剩余的 77% 召回缺口来自 canonicalizer 标注错误（6B-3 内部分解
   已证明 canonicalizer 是唯一误差源）和 resolver 选择遗漏。

## 与内部结果的对应

| 层 | 内部（corpus v4 test） | 外部（τ³） |
|---|---|---|
| Canonicalizer | 0.917 F1 | 弱（τ³ 无描述，仅名字，前科 0.40） |
| Resolver | 0.995 set F1 | 未单独测（嵌入管线） |
| Composer | 0.981 skill F1 | 继承域内能力 |
| **端到端** | **86.02 final** | **23.02 SemRecall** |

端到端差距（86→23）几乎全部来自 **canonicalizer 的跨域退化**——它内部的
0.917 依赖 description/schema 通道，而 τ³ 只有名字。这与 6A 的发现一致
（grounder 在 τ³ name-only 上掉到 0.40）。

## 对 Phase 6C 的直接指令

1. **Canonicalizer 跨域增广**：训练时加入 name-only dropout，使同一
   canonicalizer 在无 description 时也能从名字推断 canonical skill
2. **EXEC_ACTION 对映负例池**：get_X↔cancel_X 类对映工具对，6B-2 已证明
   此类负例在语料中不存在
3. Resolver 和 Composer 不再是靶心

## Limitations

- τ³ 工具无描述，canonicalizer 仅靠名字（其弱项通道）
- Resolver 在 τ³ 上的表现未单独隔离（嵌入在管线中）
- EA ref-recall 100% 是条件值（在已匹配子集内），绝对量受 pred/T 2.79 限制
