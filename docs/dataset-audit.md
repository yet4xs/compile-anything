# Dataset Audit — real benchmark coverage (Phase 5A)

> 数据：真实 benchmark（下载 provenance 见 `data/raw/metadata.json`），
> 管线：adapter → lifter → validator → simulator。
> 拒绝分析基于 `data/reports/dataset_coverage.{json,md}`（真实样本
> 9499 条）。

## 0. 真实数据清单

| dataset | 真实样本 | 来源 | license | 备注 |
|---|---|---|---|---|
| xlam | 60000 | ModelScope `LLM-Research/xlam-function-calling-60k`（Salesforce/xLAM 镜像） | CC-BY-4.0 | function-calling 主集 |
| spider | 8034 (7000 train + 1034 dev) | taoyds/spider 官方仓 | CC-BY-SA-4.0 | 文本→SQL |
| toolbench_static | 2356 | ModelScope `AI-ModelScope/ToolBench-Static` | 见上游 | ToolBench 静态评测集（ReAct 真值） |
| mbpp | 974 | GitHub 镜像（原 Google MBPP cleaned） | CC-BY-4.0 | Python 函数 |
| verilogeval | 312 (2 tracks) | NVlabs/verilog-eval 官方仓 | 见上游 | HDLBits 派生 RTL |
| humaneval | 164 | openai/human-eval 官方仓 | MIT | Python 函数 |
| toolbench | 15 | OpenBMB/ToolBench 官方仓 data_example | Apache-2.0(代码) | G1/G2/G3 答案轨迹 |
| **合计** | **71855** | | | 端到端通过 70449 |

数据获取说明：HF 直连在本环境不可达且 ToolBench/xLAM 原仓为 token 门控
（hf-mirror.com 同样绕不开门控）；**ModelScope 国内镜像**提供了这两个
数据集的无门槛副本（downloader 新增 `modelscope` plan 类型）。仍不可得：
BIRD（上游表单门槛 + ModelScope 副本损坏）、API-Bank（仓库迁移）、
AgentBench（HF LMUData）、ToolBench 全量 1.8GB 轨迹（单文件过大，暂缓）。

## 1. 哪些 benchmark 可以被 TaskIR 表达？

| dataset | lift 覆盖 | valid | 执行 | 结论 |
|---|---|---|---|---|
| xlam | **100%** (60000/60000) | 100% | 100% | 完全可表达（语义化 toolmap + EXTRACT 桥接） |
| spider | **100%** | 100% | 100% | 完全可表达（保守 lowering：复杂 SQL 留在 QUERY_DB 内） |
| toolbench_static | **100%** (2356/2356) | 100% | 100% | 完全可表达（ReAct 真值解析） |
| verilogeval | **100%** | 100% | 100% | 完全可表达（LOAD→EXTRACT→SEARCH→CODEGEN+VERIFY 骨架） |
| toolbench | **100%** | 100% | 100% | 完全可表达 |
| humaneval | **7.3%** (12/164) | 100% | 100% | 大部分不可表达 |
| mbpp | **3.3%** (32/974) | 100% | 100% | 大部分不可表达 |

关键观察：**tool-use/SQL/RTL 类任务已大规模可用（6 万级）；代码算法类
任务是表达力黑洞**。

## 2. 哪些失败？（拒绝直方图，n=9499）

| 拒绝原因 | 样本数 | 归类 |
|---|---|---|
| LOOP / 递归 | **524**（mbpp 407+32, humaneval 85） | **ISA 缺口**（v0.1 无循环） |
| 局部变量赋值后返回（`res = ...; return res`） | **137** | **lifter 能力**（赋值=SSA def，可修，不是 ISA 缺口） |
| 字符串操作（re 模块、str/int 转换、''.join、字符串表达式） | **~60** | **ISA 缺口**（无 STRING_OP） |
| 纯算术/比较/元组/下标表达式返回 | ~130 | 部分 CALCULATE 可覆盖，部分是自由表达式 |
| 其它（list() 包装、属性调用等） | 余量 | lifter 能力为主 |

## 3. 缺少哪些 Skill ISA？

按证据量排序：

1. **LOOP（region + 受控回边）**——单项解释 524 个拒绝，是 code 类任务
   的决定性缺口。spec v0.2 草案已有（taskir-spec §4.5）。
2. **STRING_OP 家族**（SPLIT/CONCAT/REPLACE/REGEX_MATCH）——直接证据
   ~60，潜在大得多（LOOP 解锁后字符串任务会暴露更多）。
3. **GROUP / TABLE_SCAN**（SQL 分解用）——当前不影响可表达性（SQL 整体
   在 QUERY_DB 内执行，coverage 100%），影响的是**算子可见性**（优化
   器无法在 IR 层做关系代数优化）。TABLE_SCAN 是 Table→List 的关键
   解锁。
4. EDA 工具链（LINT/SYNTH/SIMULATE/FORMAL_CHECK）——不影响表达，影响
   RTL workload 的成本真实性（missing-skills.md #4 不变）。

## 4. 是否需要新增 LOOP / JOIN / GROUP / STRING_OP / TABLE_SCAN？

| 候选 | 结论 | 理由 |
|---|---|---|
| LOOP | **必须加，第一优先** | 524 个真实拒绝的唯一主因；不加则 code 域永久 ~5% 覆盖 |
| STRING_OP | **加** | ~60 直接证据 + LOOP 后的连锁需求；低成本 python 类指令 |
| GROUP | 加（第二优先） | 表达力已由 QUERY_DB 兜底；加它是为了优化器可见性，不是正确性 |
| TABLE_SCAN | 加（与 GROUP 同批） | Table→List 解锁，让 FILTER/SORT/AGG 作用于 SQL 结果 |
| JOIN | **不需要新增** | 列表 JOIN 已存在；SQL 跨表 JOIN 语义不同，v0.2 决策：留在 QUERY_DB（与 GROUP 同理由），若做关系分解再引入 REL_JOIN |

## 5. 对 v0.2 的量化预期

- 修 lifter 的"赋值=SSA"（纯前端工作，不动 ISA）：code 覆盖率
  3.3%/7.3% → 估计 **~17%/~15%**（+137 样本）
- + LOOP region：→ 估计 **~70%**（解释全部 loop/recursion 拒绝的
  大部分；剩余为复合表达式）
- + STRING_OP：→ 估计 **~80%+**
- 顺序建议：**lifter 赋值支持（1 天）→ LOOP（v0.2 核心）→ STRING_OP
  → SQL 关系算子**。LOOP 是其中唯一的语义级变更（validator V3 回边
  豁免 + runtime 版本化 memo 已有 superseded 基础）。

## 6. 方法论声明

- 没有手工生成数据冒充真实 benchmark；synthetic 仅以
  `source=synthetic:*` 标签并入 corpus（reviewer 目标清单中包含
  "Synthetic 10k"）。
- 覆盖率统计对象 = 真实样本；拒绝原因由 lifter 的机器可读 reason
  直方图产生，无人工归类。
- `data/raw/metadata.json` 记录每文件 sha256 与下载时间，可复现审计。
