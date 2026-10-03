# 系统设计与方法（System Design and Methodology）

本节阐述 Compile Anything 的整体设计与实现方法。核心论点是：自然语言任务应当被视为**编译目标**而非提示工程的产物——工具链中只有前端（神经编译器）是习得的，验证、调度、执行与成本核算全部是确定性组件。系统总览如下：

```
自然语言任务 ─▶ 神经编译器 ─▶ TaskIR ─▶ Validator ─▶ Scheduler ─▶ Runtime ─▶ Trace + Cost
  (NL task)    (Qwen 3B/7B)   (SSA IR)   (V1–V6)    (list sched)  guard/VERIFY/retry   工作负载核算
                   ▲           └────────────── 确定性区域（可静态验证、可静态核算）──────────────┘
                   │                                                             │
  31,215 条审计语料 ◀── 71,855 条真实基准样本（adapter→lifter→validator→simulator）    ▼
                                                              70B 单发 baseline 对照
```

设计遵循六条可溯源至经典编译器与体系结构类比的原则：(P1) 单一 canonical IR，带书面规范与版本号；(P2) 先验证后执行，验证是数据闸门而非事后过滤；(P3) 分支以谓词而非跳转表达（guard/SELECT），保持 IR 扁平以适配小模型发射；(P4) 验证与重试为一等 IR 结构（有界投机执行）；(P5) 成本模型自 ISA 第一天起存在（每指令五维）；(P6) 严格分层——语义层永不出现工具名，executor 绑定是独立的第二层 lowering。以下各小节依次展开。

## 3.1 TaskIR：可验证的 SSA 任务中间表示

TaskIR 扮演传统编译器中 LLVM IR 的角色，与 Skill ISA（目标指令集，§3.2）、Executor（功能单元，§3.5）构成严格三层。程序由三部分组成：`inputs`（`@` 前缀的外部输入）、`nodes`（按规范拓扑序排列的扁平 SSA 节点表）与 `output`（终端值）。每个节点恰好定义一个 SSA 值（节点 id `%v` 即值名），不存在 void 节点——副作用操作的输出类型为 `Any`；`inputs`、`after`、`guard.cond`、`retry.on` 中的全部引用必须先定义后使用，构成经典 def-use 约束。选择扁平 SSA 而非带跳转的结构，动机来自实证：前沿 LLM 虽能解析 LLVM IR 语法，却在控制流重建、循环与指令级执行推理上系统性失败 [Jiang et al., ICML'25]，而发射密度是训练侧小型前端模型的主要失败模式 [WorkflowLLM]；扁平结构显著压缩了发射负担。

**类型系统**分三层：(i) 原子类型 `Str/Int/Float/Bool/Json/Table/Any`；(ii) 参数化类型 `List[T]/Set[T]/Map[K,V]`；(iii) 领域类型（开放集合，如 `Flight/Entity`，由 skill 签名或显式 `output_type` 引入，仅做名义匹配）。兼容规则刻意保守：`Any` 与任何类型双向兼容；参数化类型递归比较（`List[Any]` 与 `List[Flight]` 兼容）；其余情况要求 canonical 化后字面相等。v0.1 不做跨节点类型推断，类型来源限定为 ISA 签名的 `output_rule`（`FIXED:t`、`SAME_AS:i`、`ITEM_OF:i`）与显式收窄，使验证可在常数时间内完成。

**控制流不使用跳转**，而以谓词执行表达分支（类比 GPU/VLIW predication），并以 `SELECT` 完成数据流汇聚：

```
%c   = VERIFY(%ans)                                ; -> Bool，验证结论
%ok  = GENERATE(%ans, @task)  guard(%c == true)    ; 主路径（谓词执行）
%fb  = GENERATE(@task)        guard(%c == false)   ; 备路径
%out = SELECT(%c, %ok, %fb)                        ; 分支汇聚（数据流 φ）
```

谓词传播是语义的一部分：被跳过的条件使其 guard 的节点亦被跳过。验证与重试是一等 IR 结构：`VERIFY` 产出 `Bool`；`retry(on=%v)` 在该 VERIFY 判假时使本节点之值及其全部传递下游的记忆失效并重执行（rollback-lite），直到通过或耗尽 `max_attempts`——其运行时语义类比乱序处理器的投机执行与 squash/replay，而验证器静态保证 `on` 引用的 VERIFY 传递依赖于本节点（否则重试无意义）。与 [LLMCompiler] 失败时依赖 LLM 重规划（不可校验的全量重编译）和 [CoRE] 让 LLM 常驻解释热路径不同，TaskIR 将失败恢复编译进程序本身。

**序列化双形式**：JSON 为 canonical 机器格式（唯一合法输入），文本形式面向调试与训练目标，二者经 printer↔parser 往返保障一致；前向兼容约定使 `params/hints/meta` 保持开放字典、新增字段对旧验证器安全降级为警告。这一约定使 IR 可以随基准扩展演化而不破坏既有语料的可复现性。

**与 LLVM IR 的对应关系**：

| LLVM 体系概念 | TaskIR 体系对应 | 说明 |
|---|---|---|
| LLVM IR | TaskIR | 任务语义层：数据流 + 控制流 |
| SSA 寄存器 `%v` | 节点 id `%v` | 一节点一值，无 void |
| global `@g` | `program.inputs` 中的 `@` 输入 | 外部输入，声明类型 |
| 指令 | 节点 op（semantic skill / 控制指令） | 语义引用，不含工具名 |
| `select` / φ | `SELECT(cond, a, b)` | 数据流分支汇聚 |
| 条件跳转 `br` | `guard` 谓词执行 | 无跳转，保持扁平 |
| MemorySSA | effect token 链（§3.6） | 副作用排序 |
| `opt` / `llc` 调度 | list scheduler + 成本模型（§3.5） | 消费 IR 优化目标 |
| target ISA | Skill ISA | 能力层指令集 |
| 硬件功能单元 | Executor（lm/python/api/db） | 异构执行资源 |
| `verifier` pass | V1–V6 静态验证（§3.3） | 编译期合法性闸门 |

## 3.2 Skill ISA：带成本模型的技能指令集

Skill ISA 是"目标机指令集"，描述系统能做什么，共 **31 条 skill 指令 + 2 条控制指令**（`VERIFY`/`SELECT`，IR 级内建），按功能分为六类：

| 类别 | 数量 | 代表指令 | 资源类 |
|---|---:|---|---|
| io | 2 | `LOAD` `SAVE` | python |
| retrieval | 3 | `SEARCH`（域参数化）`FETCH` `QUERY_DB` | api / db |
| transform | 7 | `FILTER` `TRANSFORM` `SORT` `JOIN` `MERGE` … | python |
| compute | 10 | `ARGMIN` `SUM` `CALCULATE` `COMPARE` `CONVERT` … | python / api |
| lm | 7 | `GENERATE` `SUMMARIZE` `CLASSIFY` `CODEGEN` `PLAN` … | lm（2B 级） |
| action | 2 | `SEND` `EXEC_ACTION`（兜底） | api |

每条指令的 SkillSpec 携带**五维名义成本模型**：`latency_ms`（python 类 1–20ms、api 类 40–150ms、lm 类 200–900ms）、`tokens_in/out`（lm 类指令，如 `SUMMARIZE` 2000/300）、`flops`（lm = (in+out)×2·N_params，2B 执行器 4×10⁹/token）、`energy_j`（lm ≈ 0.02 J/token，python ≈ 0.005 J）、`memory_mb`（2B 执行器 ≈ 4600MB，python ≈ 30MB）。成本自第一版起即为 ISA 的一等属性，使模拟器无需执行即可核算工作负载、调度器以 `min: latency + λ·energy` 类目标做优化；所有常数为名义值，profiling 接口为后续替换点。

**两层 lowering** 将任务语义与具体实现解耦：

```
具体工具调用（如 google_flight_api.search）
   ↓ 第一层 lowering：lifter，tool → semantic skill
TaskIR 节点 SEARCH(domain="flight")
   ↓ 第二层 lowering：scheduler/runtime，skill → executor 绑定
执行单元（tool / small LM / python / db）
```

工具名永不进入 TaskIR 语义部分（仅记录于 `meta.provenance`），一个 skill 可绑定多个 executor，绑定决策归属调度器。这一分离使 IR 可跨基准移植、优化器面对语义而非厂商字符串，也是与 [LLMCompiler]（计划停留在工具名空间的无类型 JSON DAG）和 [Parrot]（数据流布线依赖开发者手工注解）的根本区别。

## 3.3 Validator：编译期验证闸门

验证器在程序进入数据集或执行之前强制六类静态不变量：

| 编号 | 不变量 | 检查内容 |
|---|---|---|
| V1 | 结构 | 字段齐全、id 匹配 `^[%@][\w.-]+$` 且唯一、op 非空 |
| V2 | def-use | `inputs/after/guard.cond/retry.on` 引用均已定义且先定义后使用 |
| V3 | DAG | data + after + guard 边无环（retry 边为时间回边，不参与环判定） |
| V4 | 类型 | 输入与 ISA 签名兼容、`output_type` 与推断兼容、`guard.cond` 为 Bool、SELECT 两分支类型兼容 |
| V5 | skill 可用性 | `op` ∈ Skill ISA registry ∪ {VERIFY, SELECT} |
| V6 | 控制流 | arity、`retry.on` 指向传递依赖本节点的 VERIFY、`max_attempts ≥ 1`、`output` 已定义 |

另有警告级检查（死节点、未使用输入、版本不匹配）。验证器同时充当数据质量闸门：任何记录进入语料库前必须通过 V1–V6，这也是后文三级评测（语法→验证→执行）的第二道闸门——"验证即闸门"取代了以往 agent 系统中"先跑起来再说"的事后过滤。

验证器的正确性以语义 fuzz 检验：构造器合法生成的 **10,000 个随机程序全部被接受（零误拒、零漏收）**，在随机 guard 与故障注入下运行时仅出现**已定义**的失败路径（consume-skipped / chosen-branch-skipped / retry 耗尽，共 1,719 例），六条运行时不变量 IV1–IV6（DAG 顺序、attempts 单调、确定性、输出非空、成本非负、Bool 产 Bool）全程成立。失败样本自动落盘以供回归，fuzzer 本身纳入版本管理，使验证语义成为可被持续检验的契约而非文档承诺。

## 3.4 神经编译器训练

**数据管线**：从 7 个公开数据集实拉 **71,855 条真实样本**（xLAM 60,000、Spider 8,034、ToolBench-Static 2,356、MBPP 974、VerilogEval 312、HumanEval 164、ToolBench 样例 15，sha256 溯源），经 adapter（仅字段映射）→ lifter → validator → 模拟器全管线转换，零 synthetic 冒充。按质量分层：Tier A 15,755 / B 15,460 / C 39,546，A+B 共 **31,215 条**进入 SFT 语料（切分 28,093/1,561/1,561），MinHash-LSH 近重复去重 + 组感知切分保证跨切分精确与近重复泄漏均为 0；训练目标取 `plan_target` 视图（纯 lowering 产物，无 policy 尾巴），`execution_target` 另存供运行时实验。质量分层的依据之一是映射可审计性：lifter 的 `map_tool()` 显式返回映射种类（exact/heuristic/fallback）与置信度，审查发现未识别工具的 fallback 曾占调用 55.2%——若不加区分即入语料，监督信号将被兜底映射污染；因此语义映射覆盖率与 lift/验证/执行覆盖率分离上报，fallback 主导的记录归入 Tier C 仅用于消融。

**语义审计（六层确定性验证）**：训练前对每条记录按 `sample_id` 回链原始真值，逐条执行六层确定性检查——(1) 动作覆盖：IR 动作节点数等于原始工具调用数；(2) 顺序一致：lowering 溯源序与 IR 链序一致；(3) 语义映射：skill 与 lowering 映射种类（exact/heuristic/fallback）及置信度一致；(4) 参数保真：原始标量参数在节点 `params` 中存活；(5) 指令一致：与基准自身措辞的鸿沟（如 "average number of" vs `avg(num_employees)`）标注为 GROUND_TRUTH_AMBIGUOUS 而非误报；(6) 溯源链完整：`meta.provenance` 与原始记录对齐。首轮审计暴露 ARGUMENT_LOSS 19,528 条（suspect 64.6%），根因是 lifter 参数映射丢弃；仅修复 lifter（参数保全率 16.7%→98.9%）、不动 IR/ISA/切分，得到 v3.1 语料：**verified 98.94% / suspect 1.06%**（Tier A 0.254%、Tier B 1.882%），残余 suspect 331 条以长参数串截断边缘为主。与 [WorkflowLLM] 的 LLM 评审和 [OrchDAG] 的合成 DAG 训练相比，该闸门完全可复现、无模型介入。

**训练配置**：基座 Qwen2.5-3B/7B-Instruct，LoRA/QLoRA 微调（peft 0.13.2、trl 0.12.2、bitsandbytes 0.44.1、torch 2.5.1，依赖全部 pin）；实验以冻结 manifest（commit、语料三切分 SHA256、seed、tier 计数）与 preflight 闸门（版本核对、bf16/4bit 加载、chat template、tokenize/forward/单步优化器往返）保障可复现。token 审计（真实 tokenizer）显示 total p50=371 / p99=856，超 2048 仅 0.17%，故取 2048 定长。评测采用三级闸门（parse→validate→execute）加 skill F1、图编辑相似度、generic-action 率与 **seen/unseen composition**——后者用于判别"编译"与"背模板"。

## 3.5 调度器与成本模型

调度采用**资源约束 list scheduling**：优先级取节点到任一汇聚点的延迟加权最长路径（critical-path 优先），就绪集合中取最高优先者派发，平局按程序序；节点开始时刻为 `max(依赖最早完成时刻, 本资源类执行器池空闲时刻)`。执行器按资源类（`lm/python/api/db`）组织独立池（默认每类 1 个），并发上限按类配置。调度器在依赖/after/guard 边构成的 DAG 上工作，并预留 effect 边接入点（§3.6）。跨资源类并行带来真收益：rtl_debug 案例中 `EXTRACT`（python）与 `SEARCH`（api）并行，makespan 1070ms 优于顺序和 1098ms。基准协议统一从 TaskIR JSON 产出 {latency、makespan、critical path、energy、memory、lm/api 调用数、schedule} 全轴指标，其中 critical path 按构造排除 retry 边（保证静态可调度下界），失败率加权的期望延迟分析作为独立的概率性扩展另论。

成本核算以 **70B 单发直答 baseline** 为参照（名义、摊销：约 2500ms、1800 tokens、约 900J、约 140GB——8×H100 级部署）。在示例基准上，编译后流水线的名义均值能耗仅为 baseline 的 **3.4%**，均值 makespan 为 **37.9%**；工作负载刻画显示语料程序节点数均值 4.79、43.6% 含并行结构、47.1% 含 retry，说明"小执行器流水线 vs 大模型单发"的权衡可在编译期静态核算。与 [Halo]、[SAGA]、[Teola] 在 serving 栈内部优化（隐含节点皆 LM 调用）不同，本调度器消费类型化程序 IR，并把 Python、API、数据库等非 LM 执行器视为带独立并发上限与五维成本的一等调度资源；名义常量在文中一律如实标注，实测 profiling 为替换接口。

## 3.6 效果系统：AI 任务的 MemorySSA

纯数据流 IR 无法表达副作用顺序：两个无数据依赖的 `SEND` 可被任何并行调度器合法交换——"先扣款、后发确认邮件"不可表达。为此我们提出类 MemorySSA 的 **effect token** 设计：每个副作用节点消费并产出一个 token，同类动作首尾相接成链：

```
world 链:  %e0 ──SEND──▶ %e1 ──SEND──▶ %e2     同类全序 = 调度硬屏障
state 链:  %e0s ──SAVE──▶ %e1s                  与 world 链相互独立、可并行
纯计算:    SEARCH / FILTER / ARGMIN / GENERATE…  零 effect 开销（约九成节点）
```

效应分两类：`world`（`SEND`、`EXEC_ACTION` 保守归入，外部不可逆动作）与 `state`（`SAVE` 等本地状态写入），两类为独立链、链间可并行；纯 skill 完全不付表示代价。在表示方案上我们对比了三个候选：专用字段（节点携带 `effect_in/out`）、显式链节点（每个动作额外一个 `EFFECT` 节点，程序体积翻倍且需两跳才生效）、隐式约定（验证器按 action 类自动串行，顺序成为不可推理的黑盒约定）。选定专用字段方案：数据流保持单值输出，token 有显式名字可做 def-use，序列化自然，且 optimizer 与调度器可以直接在 IR 上推理顺序约束。token 是**受约束的 SSA 值**：有唯一定义点、可被引用、免费进入 V2 def-use 检查，但禁入 `SELECT`/算子输入/`guard.cond`。effect 边进入 DAG（参与环检测与 critical path 计算，因同类串行是真实成本）。运行时语义与 IR 联合定义：guarded action 被跳过时 token 直通（链不断裂）；`world` 类动作默认禁止 verify-retry（不可逆动作重放等于重复扣款），幂等动作须显式 `hints.idempotent` 豁免；回滚 checkpoint 必须包含 effect 链位置，回滚边界不得跨越已推进的 `world` 链。

**τ³-bench 实证**：对 2,546 个事务型任务（2,443 个期望多动作序列）的 **14,834 个真值动作**做效应分类：reversible_state 6,920（46.6%）、irreversible_world 5,606（37.8%）、read_only 468（3.2%）、other 1,840（12.4%，如实披露）。该分布验证了 world/state 分类并非臆造：read-only 可由纯 skill 表达，reversible 对应 `state` 链加回滚语义，irreversible 恰好落入 "world 类禁 verify-retry" 的安全规则。同时暴露真实缺口：**跨类事务**（一个事务同时含 world 与 state 动作、要求全有或全无回滚）无法由当前每类独立线性链表达，已作为事务性 effect region 记录为 v0.2 设计输入，而非事后修补。

---

*本节数字均来自仓库冻结产物：成本常量见 `src/isa/registry.py`，fuzz 结果见 `scripts/fuzz_taskir.py` 运行记录，语料统计见 `data/compiler_corpus_v3_1/stats.json`，语义审计见 `data/reports/semantic_label_audit.md`，τ³ 分类见 `docs/tau3-effect-audit.md`。名义成本一律标注 nominal，未实测的实验不报告估计值。*
