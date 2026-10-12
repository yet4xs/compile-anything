# Semantic Plan Skeleton (Phase 8B-0)

## 定位

神经网络只决定 **做什么**(哪些能力、什么技能语义、依赖关系、必需参数);
编译器决定 **怎么表达与执行**(SSA、类型、验证、调度)。Skeleton 是
Natural Language → Semantic Task Graph 的中间表示,TaskIR 由确定性
Lowering 生成——错误的语义计划不会被静默修改成另一个合法计划。

## 结构

```json
{
  "task_id": "airline-44",
  "nodes": [
    {"id": "p0", "capability_id": "get_reservation_details",
     "skill": "FETCH", "depends_on": [],
     "arguments": {"reservation_id": "Q69X3R"}},
    {"id": "p1", "capability_id": "update_reservation_flights",
     "skill": "EXEC_ACTION", "depends_on": ["p0"],
     "arguments": {"reservation_id": "Q69X3R", "cabin": "economy"}}
  ]
}
```

节点字段:
- `capability_id` — 必须引用运行时可见的 CapabilityIR(注册表中的工具)
- `skill` — Skill ISA 操作(EXEC_ACTION 保留工具身份;FETCH/SEARCH 为语义技能)
- `depends_on` — 数据/顺序依赖(骨架层显式)
- `arguments` — 必需参数;**缺失即 unresolved,不得编造**

## 不变量(校验)

1. capability_id ∈ 运行时注册表(非 discoverable、agent 侧)
2. skill 与 oracle/注册表语义兼容(EXEC_ACTION 恒可;FETCH 仅 read_only 类)
3. depends_on 无环、只引用已定义节点
4. 有副作用(effect_class ∉ read_only)的节点默认串行(保守顺序)

## Lowering 契约(src/compiler/skeleton_lowering.py)

- 分配 SSA id(%p0, %p1, ...)、插入数据引用(依赖边 → 操作数/after)
- 保留 capability_id 与 effect_class(以 params.tool / 注释形式)
- Skill ISA 参数与类型校验;失败 → unresolved 报告,不静默改计划
- 产物必须通过正式 Parser + Validator
