# τ³ Benchmark Integrity Audit

**Phase**: 9B-2.1 | **Date**: 2026-10-15 | **CPU-only**

## 核心发现

### Gold Literal Observability (Task 2)

| 来源 | 数量 | 说明 |
|------|------|------|
| USER_TEXT | 123 | 值在用户指令中可见 |
| runtime_ref | 233 | 标注为运行时依赖 |
| **ENV_STATE_OR_UNKNOWN** | **245** | **值不在用户指令中,模型无法从可见信息推导** |

**关键问题**: 全部 53 个任务都包含至少一个"hidden environment value" — 参数值模型不可能知道(来自 DB/环境状态)。

| 严重度 | 数量 | 含义 |
|--------|------|------|
| **HIGH** | 37 | 第一步就使用模型不可知的值 |
| MEDIUM | 157 | 后续步骤,可能来自工具但未验证 |
| LOW | 51 | 可能从上下文推导 |

### Runtime Dependency Verification (Task 3)

| 状态 | 数量 | 含义 |
|------|------|------|
| **VERIFIED** | 68 | 字段在生产者 output_schema 中声明 |
| **UNKNOWN** | 165 | 回显字段(需要真实运行时验证) |
| REFUTED | 0 | 无明确错误标签 |

**UNKNOWN 原因**: 大部分是 "echo field" — 工具的输出字段名与输入字段名相同(如 get_product_details 的 product_id),我们无法确定输出值是否等于输入值,除非实际执行工具。

### Hidden Environment Values 的影响

以 `tau3_retail_91` 为例:
- Gold 第一步: `return_delivered_order_items(order_id="#W7553978", item_ids=[...], payment_method_id="...")`
- 这些 ID 值不在用户指令中,也不来自任何前驱工具
- 模型无法合法推导出这些值

**这意味着**: M1 在此任务上的 0% 可能是"正确地无法编造环境不可知值",而不是"规划失败"。

## 结论

| 指标 | 值 |
|------|-----|
| 53 任务中 clean(无 REFUTED + 无 HIGH hidden) | **35** |
| 53 任务中含 HIGH hidden env | 18 |
| 全部 runtime deps 中 VERIFIED | 68 |
| 全部 runtime deps 中 UNKNOWN | 165 |

## 对 Phase 9B-2 结果的重新解读

之前报告的 "全模型 0%" 需要修正:
1. **M1 的 0% 不能全部归因于跨域泛化不足** — 部分 Gold 标签包含模型不可知的值
2. **Oracle 14/14 只是静态编译通过** — 未使用真实 τ³ 工具执行验证
3. **Benchmark 需要 provenance 审计后才能作为论文证据**

## 建议

1. 使用 35 个 clean 任务重新评估
2. 对 UNKNOWN 依赖进行 τ³ 真实 replay 验证(Task 4)
3. 只有 provenance 验证通过的子集才能进入论文
