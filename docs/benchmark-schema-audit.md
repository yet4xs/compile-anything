# Benchmark Schema Audit — external evaluation suite (audited from real files, 2026-09-29)

> 全部字段基于本地已下载数据逐文件检查，非论文转述。官方 metric 与
> TaskIR 双指标原则见每节 "Metrics"。适配器实现：`src/eval/adapters/`。

## P0

### BFCL V4（4,696 case + 1 id-list 文档 = 4,697 raw 行）

| 项 | 实测 |
|---|---|
| input fields | JSONL：`{id, question: [[turn 消息]], function: [函数定义]}`；multi_turn 类另有 `initial_config / involved_classes / path / excluded_function`；memory 类有 `scenario` |
| available tools | `function` 字段（multi_turn/web_search/memory 经 `involved_classes` 引用外部类函数，不在记录内） |
| ground truth | `possible_answer/*.json`（JSONL）：single 类 `[{func: {param: [vals]}}]`；multi_turn 为逐轮 call-string 列表；web_search 为最终答案 + 子问题来源 |
| environment/state | multi_turn 的 `initial_config`（文件系统等）；web_search 需真实搜索 |
| official metric | function-call correctness（AST + 参数匹配；官方 evaluator = `third_party/gorilla@6ea57973`） |
| multi-turn / stateful | multi_turn 类是（4 子类 800 用例）；其余否 |
| offline replay | 是（除 web_search 100 例需环境） |

**特殊**：`BFCL_v4_format_sensitivity.json` 不是用例文件，是
`{category: [case_ids]}` 的 ID 清单（适配器把成员标进 metadata）。
**Oracle representability（Task 13）**：full 1,931 / partial 2,610 /
none 155；partial 主因 `unknown_tool_semantics`（2,330 个函数名超出
语义规则）与 `multi_turn_state`（800），none 全部是 memory 类
（需 memory ops，ISA 缺口）。

### τ³-bench（2,546 tasks / 4 domains）

| 项 | 实测 |
|---|---|
| input fields | `tasks.json`: `{id, description, user_scenario{persona, instructions}, initial_state(常 null), evaluation_criteria{actions, communicate_info, nl_assertions, reward_basis}, annotations}` |
| available tools | retail 有 `tools.md`；airline/telecom/banking 的工具定义在仓库代码（指针记录，不伪造 schema） |
| ground truth | `evaluation_criteria.actions`（14,834 个期望动作 `{action_id, name, arguments}`）+ NL assertions |
| environment/state | 域 `db.json`/`db.toml`（sha256 入 manifest）；动态用户模拟器 |
| official metric | task success（DB state + communicated info + NL assertions 组合 reward） |
| multi-turn / stateful | 是（核心特性）；**offline replay 需 user simulator**（部分可离线：actions 断言） |
| effect 分类 | read_only 468 / reversible_state 6,920 / irreversible_world 5,606 / other 1,840；2,443 任务多动作事务 → `docs/tau3-effect-audit.md` |

### AgentBoard — tool-query(60) / tool-operation(40) / webshop(251)

| 项 | 实测 |
|---|---|
| input fields | 统一 `{task, id, goal, subgoals, difficulty, additional_info}` |
| ground truth | webshop：`additional_info.product_id` + 属性约束；tool-*：subgoals 序列 |
| environment/state | 需 AgentBoard 交互环境（不可纯离线 replay） |
| official metric | progress rate（子目标完成度）+ success rate |
| stateful | tool-operation/webshop 是（多步交互）；tool-query 偏检索 |

（webarena 245 条已适配为 P1；jericho/babyai/pddl/scienceworld 本轮不接。）

### RTL-Repo（test 1,174 / train 2,924）

| 项 | 实测 |
|---|---|
| input fields | parquet：`{repo_name, file_path, next_line, context[{path,snippet}], created_at, all_code, cropped_code, level}` |
| 任务本质 | **行级 next-line 补全 + 跨文件 repo context**（非模块级生成） |
| ground truth | `next_line`（精确行） |
| executable testbench | **无** —— 官方 metric 是 exact-match / syntax-match pass@1，不是功能正确性（如实记录，不伪造） |
| official metric | pass@1（exact match / syntax match） |
| stateful / env | 否；offline replay 是 |

## P1

### BIRD mini-dev（sqlite/pg/mysql × 500）
`{question_id, db_id, question, SQL, evidence, difficulty}`；官方 metric
= execution accuracy（EX，需数据库环境）；offline replay 需 DB 快照。

### ToolBench full（124,345）
`{id, tools, conversations[{from, value}]}`（ShareGPT 风格；assistant 为
ReAct 文本，工具响应在 `tool` 轮）；官方 metric = pass^1 / win rate，
**需真实 API replay**（离线只能评编译侧指标）。

### AgentBench（paper-era v0.1@873b35a / v0.2@ed013ff；当前 main 为 FC 版）
原版数据经 LMUData/HF 分发（本环境不可达）——版本引用已冻结，数据
获取留待 HF 可达环境；不阻塞。

### WebShop（AgentBoard webshop/test.jsonl 251 条可用）
完整环境（1.18M 商品 + 12,087 指令）经 Google Drive 分发，本网络不可
达（PARTIAL，attempts 记录在 external MANIFEST）。

## Metrics 原则（Task 8）

每个 benchmark 双指标：**官方 metric**（上表）+ **Compile Anything 指标**
（parse rate → validator pass → execution success → semantic skill
accuracy → generic-action rate，`benchmark/neural_compiler_eval.py` 已实
现）。评测 manifest：`experiments/external_eval/manifest.json`。

## 三层套件（Task 11）

| Suite | 成员 | 论文问题 |
|---|---|---|
| A function calling | BFCL V4、ToolBench full、AgentBoard tool-query | 能否正确选 semantic skill/function |
| B stateful agent | τ³、AgentBoard tool-operation、AgentBoard webshop | TaskIR+runtime 在多步有状态副作用任务上是否成立 |
| C domain transfer | BIRD mini、RTL-Repo | 同一 compiler abstraction 跨 SQL/RTL 域 |

冻结 ID：`data/external_benchmarks/eval_suites/*.json`（只含 case_id/
category/hash；suite sha256 入 manifest）。**训练后不可挑选测试样本。**
