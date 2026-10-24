# AES 文档→IR→检查器 pilot 冻结协议(预注册,执行前提交)

**状态**: 冻结。本文写定后、任何 GLM API 调用前单独提交。执行一次;
不得依结果改提示、换题、重跑。失败与未知全部报告。

## 任务族(固定)

主问题(main-direction-decision): AES 初始化中,控制寄存器配置完成后,
什么证据允许继续写入密钥。

**6 个任务变体**(仅给定参数与场景不同,语义同源;全分母 6):

| # | KEY_LEN 配置 | 场景(给定于任务陈述) |
|---|--------------|----------------------|
| T1 | aes128 | 写 CTRL 后立即 idle(0 次忙轮询) |
| T2 | aes128 | 写 CTRL 后忙 3 次 STATUS 轮询后 idle |
| T3 | aes192 | 忙 5 次后 idle |
| T4 | aes256 | 忙 10 次后 idle |
| T5 | aes256 | 初始即非 idle(首次轮询 0,第二次起 1) |
| T6 | aes128 | 写 CTRL 后永远忙(预算内不可行) |

变体参数对模型**可见**(任务陈述说明行为窗口);oracle 检查器按同参数
构造。T6 的正确结果是计划在有限轮询后停止并被判 BUDGET_INFEASIBLE
——不是违反、不是未知。

## 信息权限(三方法完全相同)

固定 5 份原文(AES programmers_guide / theory / registers / aes.hjson
+ 共享 reggen README,commit 3a7f6947,sha256 已登记于契约表 meta),
按原文全文提供;不提供实现代码、旧模型结果、参考计划或人工契约表。

## 三方法

- **A 直接生成**: 1 次调用。输出 = 操作计划 JSON
  (步骤列表: read/write + 值),逐字面量;无出处要求。
- **B 检索+来源自检**: 1 次调用。同一输入,额外要求每步附
  {file, line, quote} 出处;输出计划 + 出处。
- **C 文档→IR→固定检查器**: 2 次调用。
  C1) 从原文抽取机器可检查契约条目(JSON:precondition/side-effect/
  completion + 出处);C2) 依据自抽契约生成计划;然后**由本管线通用
  编译器**把自抽契约编译成状态机,对该计划做提交前检查:
  - 自检通过 → 提交该计划;
  - 自检不通过 → 输出 "no_valid_plan"(不再重试)。

三方法的计划统一由 **oracle**(人工契约表 a92..338a15ac 版状态机)
判定 SUCCESS / VIOLATION(Cx) / INSUFFICIENT_EVIDENCE /
BUDGET_INFEASIBLE / UNPARSEABLE。

**人工契约表只作 oracle,不提供给任何方法**(上界原则)。

## 预算与调用纪律

- 模型: glm-4.6(官方端点);temperature=0(或平台等价确定设置);
  max_tokens 输出 4096;每方法每变体恰好 1 轮(A/B: 1 调用;
  C: 2 调用)→ 总调用 = 6×2 + 6×2 = 24(与既往批次量级一致)。
- 无重试;传输失败记 billing_unknown 并入失败分母。
- 用量按 raw_api_usage_ledger_v5 格式逐调用登记,新账本目录
  results/usage_ledger_v5/aes_pilot_20261024/。

## 评分(先于结果固定)

1. 主指标: oracle SUCCESS 率(分方法 × 变体;T6 以 BUDGET_INFEASIBLE
   计为正确处理)。
2. 次指标: (i) 违反条目分布(哪条契约最常被违反);
   (ii) B 出处逐条回验(引用是否真实存在于声明行);
   (iii) C 的自检契约与 oracle 契约的一致性(条目级 P/R,
       分歧保留例);(iv) C 的提交前拦截:错误计划被 C 自己拦下 vs
       带错提交。
3. 全分母报告: 6 变体 × 方法全列出;不隐藏任何输出。

## 停止/继续规则(承 main-direction-decision)

- 若 A(直接生成)已 ≥ C 的最终正确率且 B 出处全真: 保留 null 结果,
  停止把本范围包装为"检查器管线优势"的主张;数据与脚本存档。
- 若 C 拦截了 A/B 会犯的违反且最终正确率高于 A/B: 支持继续扩大
  (SPI 迁移、更多任务族),下一批另冻结。
- 若三方法全对(天花板): 说明该任务族过浅,不构成方法区分度;
  转向更深处(IV/消息处理/异常)另冻结,本结果如实报告。

## 执行产物

- 任务包(冻结输入) results/aes_pilot_20261024/tasks.json(先于调用提交)
- 三方法原始输出(逐调用,含 usage)
- 判定明细 + 汇总表 docs/paper/aes-pilot-results-20261024.md
- 用量账本 usage.json

## 执行前修正案(仍在任何 API 调用之前)

T6 正确处理的精确定义: 判定 ∈ {BUDGET_INFEASIBLE, INSUFFICIENT_EVIDENCE}
且无 VIOLATION——即方法要么在有限轮询后诚实停止(未伪造继续证据),
要么耗尽预算;两者均计为正确。伪造证据继续写或违反其他契约仍为错。
触发场景: 检查器在 never-idle 下,停止型计划判 INSUFFICIENT_EVIDENCE,
无限轮询型判 BUDGET_INFEASIBLE——两者都是"未做错事"。
