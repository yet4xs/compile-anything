# Phase 8C-1: Synthetic Chain Semantic Audit + Runtime Dataflow Feasibility

## Task 3 结果:合成链语义分类

| 类别 | 数量 | 占比 | 说明 |
|------|------|------|------|
| **explicit-value** | 604 | 67.2% | 所有参数值在指令文本中明写 |
| **copied-literal** | 295 | 32.8% | 参数值来自上游参数但也在指令中 |
| **genuine-runtime** | **0** | **0%** | 参数值仅可从前驱工具输出获得 |
| inconsistent | 0 | 0% | — |

**结论:合成链中 0% 具有真正的运行时数据传递。** 全部 899 条的参数值都在自然语言指令中明写(生成器设计如此),因此:
1. 两个模型都能 100% 提取参数(达到天花板)
2. `depends_on` 边只起排序作用,不代表真实的值消费
3. 不能用于检验分层在数据流建模上的优势

## Task 4:真实数据流最小测试集(设计)

### 需要什么

```
指令:"查询用户 Alice 的账户,找到最近一笔交易,查询该交易状态。"

p0: lookup_user(name="Alice")           → 输出 {user_id: "U123"}
p1: get_account(user_id=p0.user_id)      → user_id 不在指令中!
p2: get_latest_txn(account_id=p1.account_id) → account_id 不在指令中!
p3: get_txn_status(tx_id=p2.tx_id)       → tx_id 不在指令中!
```

### TaskIR v0.1 表示缺口

当前 TaskIR 的操作数引用是**节点级**的(`%p0`),不支持**字段级**引用(`%p0.user_id`)。要表达 `get_account(user_id=%p0.user_id)`,需要:
- **字段选择原语**(如 `FIELD(%p0, "user_id")`)
- 或**类型系统扩展**(输出类型为 Dict/Record 时允许字段访问)
- 或**运行时绑定**(延迟到执行时从上游结果中提取)

### 可行路径

1. **扩展 Skill ISA**:增加 `EXTRACT_FIELD(ref, field_name)` 原语,将上游输出字段作为下游参数
2. **CapabilityIR 输出 Schema**:每个工具声明输出字段名和类型,Lowering 时可自动插入字段提取
3. **Mock Runtime**:提供确定性返回值,使链可端到端执行验证

### 建议的最小实验(不训练)

构造 20-50 条含 `EXTRACT_FIELD` 的链,验证:
- Skeleton → Lowering → TaskIR(带字段引用)→ Validator 通过
- Mock Runtime 执行:前驱输出 → 字段提取 → 下游输入 → 最终结果
- 与无字段引用的版本对比,证明扩展的表达能力

## Task 5:用修正评测器(v3.1)重算的结果

见 `results/phase8c/semantic_eval_v31.json`。

## Q1-Q5 裁决

**Q1:Phase 8C 是否真的使用 v4 数据训练?**
**是。** 服务器脚本 DATA=v4,训练日志 n_samples=15,480 与 v4 一致。但 v4 路径修改未提交到 git(已修正)。

**Q2:M1 工具身份收益来自模型选择还是编译器约束?**
**两者都有。** 编译器保证当 Skeleton 引用合法 ID 时 Lowering 不丢失身份(compiler guarantee);但 M1 的 99.5% 是模型在 Skeleton 格式下**选择正确工具**的准确率——这高于 M0 直接生成 TaskIR 时的 93.6%,说明 Skeleton 表示对模型选择工具有正面影响。但**不能归因于编译器保证**,因为编译器不保证模型选对工具。

**Q3:合成深任务包含多少真正的数据值依赖?**
**0。** 全部 899 条链的参数值在指令中明写。depends_on 边是结构性的(排序),不代表值消费。

**Q4:现有 TaskIR 是否可以表达前驱工具结果中的字段引用?**
**不能(有表示缺口)。** 需要增加字段级引用原语或类型系统扩展。

**Q5:修正后的 Strict Joint 是否仍显示 M1 不如 M0?**
**是(在 unseen-family 上)。** 参数提取差异是主要原因(与 8B-1C 结论一致)。
