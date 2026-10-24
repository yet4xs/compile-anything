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

---

## VOID 声明与 v2 修正案(run 1 之后、run 2 之前)

**run 1 = VOID**(24 调用,1.35M tokens,记录完整保留于 void_run1/):
15/18 UNPARSEABLE 的成因为配置缺陷,非方法差异——(i) 预注册误写
model=glm-4.6,与既往全部批次(glm-5.3 家族)不一致;(ii) 未设置
thinking/reasoning_effort,glm 思考模型把 4096 输出预算耗尽在
reasoning(样本: completion 4096,reasoning 4081),JSON 正文为空。
此为基础设施错误,类比 8D-2B 的截断门控:如实报告、修配置、重跑,
**不视为在同一探针上变提示捞结果**——任务、文档、提示正文、
判定与评分全部与 v1 逐字相同。

**v2 变更(仅两处,调用前冻结)**:
1. 请求配置: model=glm-5.3-flash,thinking {type: enabled},
   reasoning_effort low,temperature 0,max_tokens 8192。
2. C 的机械映射缺陷修复: v1 的关键词存在性检测使用了整理者措辞,
   模型输出不可能逐字命中(结果 0/6,退化为恒通过)。v2 改为 C1
   输出 schema 增加固定枚举字段 id ∈ {INIT_IDLE, CTRL_BEFORE_KEY,
   RECHECK_AFTER_CTRL, CTRL_TWO_WRITE, KEY_ALL_REGS,
   READ_RESETS_PHASE};机械映射 = 该枚举的集合存在性(无语义解释)。
   C1 提示相应增加枚举定义(其余逐字不变)。

## v3 修正案(run 2 后、C 臂重跑前)

run 2 的 A/B 臂有效(解析与判定机械正确,结果保留);**C 臂 VOID**:
extract_json 只匹配 {...} 对象,C1 的数组输出 [...] 永远解析失败
(entries_n=0/6 全量),C2 实际拿到空契约、自检退化为恒通过。
与 run1 同类基建缺陷,同规则处置: 修解析(对象或数组皆可)、
补原始输出留存(run2 未存正文,审计缺陷一并修复)、
**仅重跑 C 臂 12 调用**,A/B 不重跑不重采。B 的引文回验为严格
行精确匹配,保持冻结;其原始正文未留存,作为局限如实报告。

## v5 修正案(修复轮,调用前冻结)

v4(深度感知重打分,零 API,基于 v3 已存计划)证明: k+1 阈值
(k 次忙 + 1 次确认 idle)下,T2-T5 全部被拦截且**可修复**(补足
轮询即可),T6 走结构路径诚实停止;管线不再提交任何含违反的计划。

v5 = 对 4 个可修复 decline 各执行**一次**修复调用(C3):
- 输入 = 原 C2 消息 + 检查器反馈原文(缺多少次轮询、在哪个窗口)
- 输出计划经同一 v4 深度+结构自检;通过才提交,否则保持拒绝
- 无限窗口(T6)不修复,维持拒绝
- 评分沿用冻结口径;T6 正确 = {BUDGET_INFEASIBLE,
  INSUFFICIENT_EVIDENCE}(经结构路径)
新增调用于账本 aes_pilot_v5_repair_20261024。
