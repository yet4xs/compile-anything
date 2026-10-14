# Phase 8D-1A: Executable Dynamic Dataflow Corpus — Readiness Report

**Date**: 2026-10-14 | **8D-0.2 Gate**: PASS (`bf7b960`)

## 语料统计(data/runtime_dataflow_v1/)

| 指标 | 值 |
|------|-----|
| 生成任务 | 195 |
| 去重后 | **97** |
| 严格验证通过 | **195/195 (100%)** |
| 运行时执行成功 | **195/195 (100%)** |
| 失败数 | **0** |

### 切分
train 71 / dev 13 / test 13

### 拓扑覆盖(9 种独立拓扑)

| 拓扑 | 生成数 | 说明 |
|------|--------|------|
| linear_3 | 30 | lookup → account → transactions |
| linear_4 | 25 | + tx_status |
| linear_5 | 20 | + credit_score |
| linear_6 | 15 | + loan_approval |
| fork | 20 | 同一 lookup → 两个独立下游 |
| join | 20 | 两个上游字段 → 一个消费者 |
| shop_chain | 25 | search → details → order |
| read_mutate | 20 | 读链 + 状态修改 + 通知 |
| fan_out_in | 20 | fan-out → fan-in + mutation |

### 深度分布
3步(43) / 4步(22) / 5步(20) / 6步(12)

### 动态参数统计
每任务 runtime_ref 数:2(22)/3(33)/4(20)/5(10)/6(12)——**所有任务都有运行时值传递**

## 关键特征

1. **动态参数不出现在指令文本中**:指令只说"Look up user Alice, then retrieve their account details"——user_id 在运行时才能获得
2. **参数来源全部是前驱工具输出**:通过 EXTRACT_FIELD + bind() 从实际执行结果获取
3. **每个任务同时生成 M0(TaskIR)和 M1(Skeleton)标签**:监督一致
4. **9 种独立拓扑**:不再是"改变用户名的同构测试"

## 严格门槛核对(Task 5)

| # | 条件 | 状态 |
|---|------|------|
| 1 | 合法任务 Strict Validator Pass | ✅ 195/195 |
| 2 | 非法字段/类型任务被拒绝 | ✅ 23/23 负面测试 |
| 3 | 执行前预检阻止不支持副作用 | ✅ mutation-before-guard 测试 |
| 4 | Runtime 参数值与 Oracle Trace 一致 | ✅ |
| 5 | Text/JSON/AST Roundtrip | ✅ |
| 6 | v0.1 回归通过 | ✅ 100/100 |
| 7 | Lowering 不将 runtime_ref 退化为字面量/order | ✅ |
| 8 | 每个工具输入参数有明确 literal 或 runtime_ref 来源 | ✅ |

## 结论

**允许启动 M0/M1 小规模可行性训练。**
语料规模(97 unique)足够做 pilot signal 检测;若 M1 在动态参数绑定上显示优势，可扩大生成(增加工具域和任务模板)。
