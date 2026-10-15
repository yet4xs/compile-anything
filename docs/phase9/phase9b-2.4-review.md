# Phase 9B-2.4 — Runtime Gold Correctness & Compiler Integration

**Baseline**: 78c7828 | **全部 CPU,官方 τ³ 模拟器,无 GPU,无历史重写**

## P0 — Security

- `clean_git_history.sh` 加固:程序性门控(`CREDENTIAL_ROTATED=yes` +
  `FORCE_PUSH_AUTHORIZED=yes` 才能运行)、秘密经 `mktemp`(0600,EXIT trap
  清理)、重写前 `git bundle` 备份、**只重写/强推 main**(去掉
  `--all/--tags`)。维持草案不可执行状态。
- `ssh_config.py`:TOFU 自动信任已移除——未知主机密钥仅在匹配
  `BITAHUB_EXPECTED_FINGERPRINT`(带外获取)时接受,否则拒连并打印所见
  指纹。
- 事件状态: **Credential Rotation Unconfirmed**(所有者未确认轮换)。
  GPU/服务器任务继续封锁。

## Task 1 — Gold Execution Audit

生产 E2E(`taskir_gold_execution_v2.json`)逐步审计,每步独立判定
工具/参数/结构化输出/DB hash:

| 任务 | 编译 | 验证 | 执行 | 参数+工具逐步一致 | 终态 hash |
|------|------|------|------|------------------|-----------|
| tau3_retail_0 | ✓ | ✓ | ✓ 5 calls | ✓ | ✓ |
| tau3_retail_1 | ✓ | ✓ | ✓ 5 calls | ✓ | ✓ |
| tau3_retail_25 | ✓ | ✓ | ✓ 6 calls | ✓ | ✓ |
| tau3_retail_62 | ✓ | ✓ | ✓ 5 calls | ✓ | ✓ |
| tau3_retail_65 | ✓ | ✓ | ✓ 3 calls | ✓ | ✓ |

- **make_tool_call 等价**: 5/5 任务底层 `tools.tools[name]` 直调与官方
  `env.make_tool_call` 输出与 DB hash 完全一致——直调路径无行为偏差。
- **LITERAL 启发式审计**: 5 条风险标记全部是 zip 码(纯数字触发保守
  标记),逐条核实均逐字存在于用户文本——分类正确,标记机制按设计
  工作(标记 → 人工确认)。

## Task 2 — Verified Registry(`verified_registry_manifest.json`)

30 工具,全部来源官方:输入 = 工具函数签名(inspect);输出 = 返回类型
注解递归展开 pydantic 模型(Struct/List[Struct]/Dict/裸标量 scalar_root);
effect = toolkit 官方 metadata(**仅 mutates 与否**——可逆性不在官方
来源中,不伪造,统一标 mutating)。与任何模型预测无关。
Model/Gold/Validator/Lowering/Runtime/Scorer 全部经 `load_verified_schemas`
适配同一版本。

## Task 3 — Graph Scorer Hardening

`gold_binding_scorer.py`:
- 同签名等价实例组内**回溯枚举映射**(组 ≤6),以一对一绑定匹配数选优;
  贪心仅作 >6 组回退并显式标记 `bijection_method`
- Gold 边一对一消耗(重复认领被拒,专项测试)
- **三指标分离**: `value_match`(调用双射)/ `exact_binding_match`
  (绑定精确,含无多余)/ `strict_binding_exact`(= 前两者 + gold 全定源);
  end-state 由执行器单独报告
- 对抗测试: 相同工具+相同字面量+不同绑定的双实例,预测合法重排后
  仍正确配对(`bijection_method=backtracking`);同 gold 边双认领被拒

既有 21 条可解析预测 vs v1 gold(加固后): 双射 0 / 精确 0/121 /
root-scalar 等价 1(`strict_binding_scorer_v2.json`)。此数为**诊断
基线**——受历史截断、gold 未定源参数、模型接口限制三重影响,不作
正式跨域结论(与 9B-2.3 立场一致)。

## Task 4 — Typed Access 生产集成(完成)

- **ISA**: MAKE_LIST intrinsic(variadic,无副作用)
- **Lowering**: runtime_ref 路径直达 EXTRACT_FIELD(field="orders[0].order_id");
  runtime_ref_multi → 逐元素 EXTRACT_FIELD + MAKE_LIST → bind
- **Validator**: 描述符 schema 下的类型化路径检查——静态可判定 / Any
  边界(警告 DYNAMIC_CHECK_BOUNDARY,非错误)/ 静态反驳(FIELD_NOT_FOUND,
  编译期 fail-closed);root-scalar(field="")仅对 scalar_root 生产者合法;
  ARITY 检查修复 variadic 比较
- **Runtime**: 官方环境 SSA 解释器(EXTRACT_FIELD=typed path 解活体对象,
  MAKE_LIST,EXEC_ACTION=官方工具);mock 侧回归不受影响
- **回归**: 37 项测试全过;旧 flat schema 用例向后兼容;畸形路径
  (`orders[0]id` 等)在编译期被拒
- `orders[i]` 类路径在静态证明下的动态边界(retail_25/62/65 各 1 条
  root-scalar 警告)如实记录

## Task 5 — Gold Dataset Integrity

- 5 条 strict 候选全部通过**生产 TaskIR 端到端**(编译→验证→官方执行→
  逐步审计),`admitted_v2` = 5/5
- **Group-aware 分组**: (domain, acting_user, 工具序列模板) 分组确认
  retail_0/1 同组(同一模板、同 zip 查找用户流)——**不得跨划分**。
  5 任务共 4 组,样本量仍不足以冻结测试划分(维持 ≥15 目标)
- 14 条探索分析任务继续排除;无强制消歧
- 生产端到端通过数(严格口径): **5/53**

## 交付物

- `results/phase9/taskir_gold_execution_v2.json`(生产 E2E + 审计)
- `results/phase9/verified_registry_manifest.json`(官方来源 registry)
- `results/phase9/strict_binding_scorer_v2.json`(加固 scorer 诊断)
- `src/eval/gold_binding_scorer.py`(回溯双射)+ 8 项测试
- `src/validator/validator.py`(类型化路径)+ `src/isa/registry.py`
  (MAKE_LIST)+ `src/compiler/skeleton_lowering.py`(multi-ref lowering)
- `tests/test_taskir_typed_integration.py`(7 项)
- `scripts/build_verified_registry.py` / `execute_taskir_official.py`
- `scripts/clean_git_history.sh` 加固 + `scripts/ssh_config.py` 指纹强制

## 报告五问

1. **生产 TaskIR 端到端通过多少任务**: 5/5 候选全部通过(编译、类型
   验证、官方环境执行、逐步参数/工具/终态审计)。
2. **Gold/预测 Schema 一致性**: 成立——单一 verified registry
   (官方注解来源)经适配器供全部组件;预测侧评分也已切换。
3. **Scorer 重复调用与拓扑重排**: 正确——等价实例经依赖约束回溯区分,
   合法重排不影响(专项测试);gold 边一对一。
4. **安全处置状态**: 仓库侧完成(凭据移除、指纹强制、清理脚本门控);
   **凭据轮换未确认**(所有者未回复)→ GPU/服务器继续封锁。
5. **Phase 9B-3 GPU GO/NO-GO**: **NO-GO 维持**——硬阻塞仅剩安全
   (轮换未确认);研究侧生产管线已就绪,严格 gold 规模(5 任务 4 组)
   仍低于测试冻结阈值(≥15),需扩充 AMBIGUOUS 裁决或引入新任务源。
