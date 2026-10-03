# Phase 5B-2 外部 Benchmark 评测计划

> 基线：Phase 5B-1 完成，E1 模型冻结
> 冻结：训练数据 / TaskIR / Skill ISA / external benchmark suite / eval adapters

## 执行状态

| Task | Benchmark | 样本数 | 状态 | 预计时间 |
|---|---|---|---|---|
| 1 | E1 模型 manifest | — | ✅ | — |
| 2 | External inference pipeline | — | ✅ 内联于 Task 3 脚本 | — |
| 3 | **BFCL V4** | 4,696 | 🔄 运行中 | ~2-3h |
| 4 | τ³-bench | 2,546 | 待跑 | ~1.5h |
| 5 | AgentBoard (tool-query/op/webshop) | 351 | 待跑 | ~20min |
| 6 | RTL-Repo (test) | 1,174 | 待跑 | ~1h |
| 7 | 统一论文表格 | — | 待生成 | — |
| 8 | Error analysis | — | 待生成 | — |

## 报告格式（Task 7 输出）

| Benchmark | Samples | Parse% | Valid% | Exec% | Official Metric | Notes |
|---|---|---|---|---|---|---|
| BFCL V4 | 4,696 | ? | ? | ? | ? | 按 full/partial/none 分列 |
| τ³-bench | 2,546 | ? | ? | ? | — | 按域分列 |
| AgentBoard | 351 | ? | ? | ? | — | 3 类任务 |
| RTL-Repo | 1,174 | ? | ? | — | — | 无 testbench，不声称 functional correctness |
| BIRD mini | 500 | ? | ? | ? | — | SQL domain |

## Error Analysis 分类（Task 8）

| 错误类别 | BFCL | τ³ | AgentBoard | RTL |
|---|---|---|---|---|
| Wrong skill | ? | ? | ? | ? |
| Missing dependency | ? | ? | ? | ? |
| Wrong argument | ? | ? | ? | ? |
| Unsupported ISA | ? | ? | ? | ? |
| Effect violation | — | ? | ? | — |
| Multi-turn state missing | ? | — | ? | — |
| Memory (ISA gap) | ? | — | — | — |

## 附加实验：No-Validator Ablation

对比 E1 输出直接执行 vs E1 → Validator → Runtime：
- 证明 validator 是必要组件（过滤非法程序）
- 量化 validator 拦截了多少潜在运行时错误
