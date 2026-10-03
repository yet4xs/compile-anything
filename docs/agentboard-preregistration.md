# AgentBoard Untouched 首测 — 协议预注册

> 注册时间：2026-10-03（**在任何 AgentBoard 推理运行之前**）
> 基线 commit：`659061e`（Phase 5C freeze）+ freeze-fix 统计口径修正
> 模型：**E5C-S**（Phase 5C 定版，`runs/phase5c/e5c_s/final`）

## 1. 测试集（一次性，全部）

| 任务 | n | 层 |
|---|---:|---|
| tool-query（academia 20 / movie 20 / weather 20） | 60 | A function calling |
| tool-operation（todo 20 / sheet 20） | 40 | B stateful agent |
| webshop | 251 | B stateful agent |
| **合计** | **351** | |

数据：`data/external_benchmarks/agentboard/data/{tool-query,tool-operation,webshop}/test.jsonl`（sha 已冻结于 eval_suites）。

## 2. 输入构造（capability schema 的唯一合法来源）

E5C-S 部署协议要求推理时带 capability schema。本测试 schema 构造规则：

**允许（env-visible，by construction）：**
- tool-query / tool-operation：使用 AgentBoard 环境运行时展示给 agent 的动作空间定义
  `agentboard/prompts/Raw/{todo,sheet,academia,movie,weather}_raw.json` 的
  `tool_set_message`（name/description/parameters），仅取该任务 `additional_info.tool`
  对应工具的定义，经 `format_capabilities`（≤15 条）拼入 prompt。
- tool-operation 的 `additional_info.init_config`（初始环境状态，运行时可见）拼入 prompt。
- webshop：WebShop 标准动作空间（search[keywords] / click[element] / buy[now]），
  该动作空间是 WebShop 环境对 agent 的公开接口定义。

**禁止（GT 泄漏）：**
- `subgoals`（ground truth 分解）、`additional_info.answer`、`additional_info.product_id`
  及任何 grading 字段、baseline_results、未来轨迹 —— 一律不得进入 prompt 或 schema。
- 不得因结果调整任何 prompt/schema/解码参数。

## 3. 推理协议（与 Phase 5C 定版协议一致）

- prompt：SYSTEM_PROMPT + `build_capability_prompt(goal, capabilities)`
- input max_length 2048，max_new_tokens 512，greedy，left padding
- 输出解析：`extract_taskir_text` → `parse_text` → V1-V6 validate

## 4. 预注册指标（跑之前定死）

1. **结构层**：Parse%、Valid%（全 351 条）
2. **计划层**：Pred/T（每任务 action 数）、EXEC_ACTION 使用率
   （Phase 5C 教训的外推检验：tool-operation 应触发 EXEC_ACTION）
3. **答案层（tool-query/tool-operation 100 条）**：TaskIR → 环境工具调用的
   ground-truth 对照 —— 对可静态判定的（如日期查询）计算 answer-match；
   不可静态判定的标注 UNSCORED-OFFLINE，不猜测
4. **webshop 251 条**：离线无环境，仅结构层 + 计划层；progress rate 需交互环境，
   如实报告为 limitation（官方 env 依赖浏览器集群，超出本轮范围）
5. 参考对照（只读）：`baseline_results/` 的 GPT-4/GPT-3.5 等官方基线数字

## 5. 结果处理规则

- 一次性运行，结果无论好坏照登，不回调 E5C-S，不修改任何评测组件
- AgentBoard 从此转为 seen（不再是 untouched）
- 论文以本次结果为 AgentBoard 终值
