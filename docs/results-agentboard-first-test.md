# AgentBoard Untouched 首测结果（E5C-S）

> 测试时间：2026-10-03，预注册协议 `docs/agentboard-preregistration.md`（commit `d59d00b`，
> **先于任何 AgentBoard 推理注册**）。
> 模型：E5C-S（Phase 5C 定版，schema 部署协议）。
> 一次性运行，无任何结果驱动修改。AgentBoard 自此转为 seen。

## 主结果（351 条，离线编译协议）

| Task | n | Parse% | Valid% | Pred/T | EXEC_ACTION% | 主产出来源 |
|---|---:|---:|---:|---:|---:|---|
| tool-query | 60 | 98.33 | 78.33 | 1.32 | 2.5 | FETCH 60 / SEARCH 13 |
| tool-operation | 40 | 95.00 | 87.50 | 1.77 | **23.9** | FETCH 49 / **EXEC_ACTION 17** |
| webshop | 251 | 98.41 | 96.02 | 0.96 | 0.0 | SEARCH 242（近退化解） |
| **合计** | **351** | **97.7** | **91.7** | 1.12 | — | — |

## 官方基线（只读对照；交互环境协议，与本次离线协议**不可直接比较**）

| Model | tool-query SR | tool-operation SR | webshop SR |
|---|---:|---:|---:|
| GPT-4 | 0.683 | 0.600 | 0.390 |
| GPT-3.5-turbo | 0.450 | 0.075 | 0.351 |
| Claude 2 | 0.483 | 0.275 | 0.379 |

（SR = success rate，来自 AgentBoard 官方 baseline_results；我们的离线协议不产出 SR，
此表仅提供文献锚点。）

## 三个发现

### 1. 结构泛化在第三个基准家族上复制（零污染）

AgentBoard 从未进入训练/评测（contamination firewall 全程有效），首次接触即
parse 95.0~98.4%、valid 78.3~96.0%。TaskIR 结构规则（SSA/类型/DAG/validator）的
跨域泛化至此在 BFCL（94.5%）、τ³（98.9%）、AgentBoard（91.7%）三个家族上成立。

### 2. EXEC_ACTION 边界发现外推确认——且 schema 部分缓解

tool-operation（需要状态变更类工具调用）上 EXEC_ACTION 使用率 **23.9%**：
- 远高于 τ³ 裸 prompt 的 ~0% —— **env-visible capability schema 使部分 EXEC_ACTION
  接地成为可能**，与 Phase 5C"训练+推理都带 schema 才有效"的结论一致；
- 但仍被 FETCH 压制（49 vs 17）—— Phase 6 靶心（EXEC_ACTION 边界的跨域接地）
  在新基准上再次确认。
- tool-query（只读查询）以 FETCH/SEARCH 为主 —— 语义方向正确（只读任务不需要
  EXEC_ACTION），这是 EXEC_ACTION 区分度存在的正面证据。

### 3. webshop 暴露长程规划短板（与 τ³ 一致）

pred/T 0.96、几乎全部为单 SEARCH —— 购物任务需要多步（搜索→点击→比较→购买），
模型仍输出单步计划。与 τ³ pred/T 1.75（vs 参考 5.83）同一现象：
**浅计划先验来自训练分布，不因 benchmark 改变**。

## Limitations（预注册中已声明）

- 离线编译协议不运行交互环境：progress rate / success rate 不可得，
  不能与官方基线数值直接比较；
- 答案层（answer-match）因无环境执行器而 UNSCORED-OFFLINE；
- 结论限于结构与计划层。

## 工件

- `results/agentboard/first_test.json` — 指标 + 基线对照
- `results/agentboard/first_test_preds.jsonl` — 351 条 TaskIR 预测
- `scripts/agentboard_first_test.py` — 预注册协议实现
