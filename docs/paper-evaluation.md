# Evaluation 章节规划（§9）

> 对应 `docs/paper-outline.md` §9（E6–E10 行）。数据与代码依据：
> `experiments/external_eval/manifest.json`、`docs/benchmark-schema-audit.md`、
> `docs/external-benchmark-status.md`、`docs/training.md`、
> `benchmark/neural_compiler_eval.py`、`docs/semantic-label-audit.md`。
> 纪律：训练未跑，本文件所有结果表为空模板；**PENDING 单元格禁止以估计值填充**。

## 1. 评测体系设计

评测章需要回答三个递进的问题：其一，学习到的编译器前端能否把自然语言任务
**编译**为合法且可执行的 TaskIR 程序（编译质量）；其二，编译得到的程序在真实
基准上是否还原了任务语义，而非仅在内部测试集上拟合（外部效度）；其三，监督
数据的质量闸门与各种训练选择各自贡献了多少（归因与消融）。为此，评测体系由
**三层外部套件、双指标、全量冻结**三个要素构成。

**三层套件**按任务形态而非数据来源划分，分别对应论文的三条核心主张：A 层
考察函数调用形态下的语义技能选择能力，这是编译器前端最基本的"指令选择"问题；
B 层考察多步有状态、带副作用的交互任务，检验 TaskIR 与运行时（guard/VERIFY/
retry 回滚）在副作用排序与状态维护上是否成立，这是与纯规划系统拉开差距的
地方；C 层考察同一套编译器抽象（同一 Skill ISA、同一 validator、同一评测
协议）跨 SQL 与 RTL 两个专业域的迁移能力，回应"Compile Anything"的通用性
主张。三层共同构成从"会调用工具"到"能编译任意域任务"的递进论证。

| Suite | 成员（冻结 case 数） | 论文问题 | 官方 metric |
|---|---|---|---|
| A function calling | BFCL V4（4,696）、ToolBench full（124,345）、AgentBoard tool-query（60） | 能否正确选择 semantic skill / function | AST 函数调用正确率；pass^1 |
| B stateful agent | τ³-bench（2,546，airline/retail/telecom/banking 四域）、AgentBoard tool-operation（40）、AgentBoard webshop（251） | TaskIR+runtime 在多步有状态副作用任务上是否成立 | task success（组合 reward）；progress rate |
| C domain transfer | BIRD mini-dev（500）、RTL-Repo test（1,174） | 同一 compiler abstraction 跨 SQL/RTL 域 | execution accuracy（EX）；pass@1（exact/syntax match） |

**13 个基准分组冻结**：按"基准×任务域"粒度共 13 组——BFCL V4、ToolBench
full、BIRD mini-dev、RTL-Repo、AgentBench、WebShop 六个基准仓，加 τ³ 四域
（airline/retail/telecom/banking）、AgentBoard 三任务（tool-query/
tool-operation/webshop）。其中 11 组已完成 case-ID 级冻结：`data/
external_benchmarks/eval_suites/*.json` 共 8 个套件文件，**合计 133,612 条
case**，套件与适配器的 sha256 均写入 `experiments/external_eval/manifest.json`
（另含 BFCL 的 oracle 可表达性审计：full 1,931 / partial 2,610 / none 155，
用于解释编译侧指标的上界）。AgentBench 与 WebShop 全量数据受网络门槛限制，
版本引用与快照哈希已冻结，数据获取留待可达环境，不阻塞主线。冻结即承诺：
**训练开始后不得挑选、删减或重抽测试样本**。训练数据（corpus v3.1）与外部
评测数据由 `src/dataset/external_guard.py` 在所有 corpus builder 入口强制
隔离，`tests/test_external_benchmark_guard.py` 持续守护，杜绝评测集泄漏进
监督数据。

**双指标原则**：每个基准同时上报两类指标。（i）**官方 metric**（见上表），
使用基准自带的 evaluator 与 ground truth，保证与已有工作可比、可复核；对
τ³ 这类需用户模拟器的基准，优先采用可离线 replay 的动作断言子集并如实标注
覆盖比例。（ii）**Compile Anything 指标**：parse rate（文本能否解析为
Module）→ validator pass（V1–V6 静态合法性）→ execution success（模拟器
能否完整执行）三级闸门，再加 op-seq exact、skill F1、图编辑相似度
（approx. GED，下称 GES）、generic-action rate（EXEC_ACTION 直调占预测
算子的比例，即"语义化失败率"）。前者回答"任务做得对不对"，后者回答"前端
是否产出了合法、可执行、语义化的程序"；二者分离上报，是因为官方 metric
无法区分"前端编译得好但执行环境弱"与"前端根本没编出合法程序"，而这一区分
恰是编译器论文的本体问题。对必须真实环境 replay 的基准（ToolBench 真实
API、webshop 环境），离线阶段至少上报编译侧指标并明确标注环境缺口。

## 2. 实验矩阵（E0–E3）

训练语料为 corpus v3.1：Tier A 15,755 条 + Tier B 15,460 条，共 **31,215**
条（train 28,093 / val 1,561 / test 1,561，MinHash-LSH 去重 + group-aware
切分，跨 split 完全与近重复均为零）。E1/E3 实际消费 train split 28,093 条；
监督目标为 `plan_target`（纯 lowering 视图，不含 policy 尾巴），capability
context 默认 OFF，序列长度 2048（token 审计 p99=856，超长仅 0.17%），
seed 42。冻结包、依赖 pin 与 preflight 闸门见 `experiments/phase5b1/
manifest.json` 与 `docs/training.md`；正式训练前先以 200 条 verified-only
样本做 sanity overfit，验证训练管线本身。

| 编号 | 模型 | 训练方式 | 样本 |
|---|---|---|---|
| E0 | Qwen2.5-3B-Instruct | zero-shot 基线 | 0 |
| E1 | Qwen2.5-3B | QLoRA（4bit NF4） | 31,215（train 28,093） |
| E2 | Qwen2.5-7B-Instruct | zero-shot 规模对照 | 0 |
| E3 | Qwen2.5-7B | LoRA（bf16）/QLoRA | 31,215（train 28,093） |

四个实验共用同一评测协议与指标链（`benchmark/neural_compiler_eval.py`
已实现）：**parse rate → validator pass → execution success → op-seq
exact → skill F1（micro + per-skill）→ GES → generic-action rate →
seen/unseen composition**，并按 source 分解（xlam/spider/toolbench_static/
verilogeval/humaneval/mbpp）。指标链的每一级隔离一类失败模式：解析失败是
语法问题，validator 失败是语义构造问题，执行失败是运行时契约问题；op-seq
与 skill F1 度量指令选择的准确率，GES 度量图结构的近似质量，generic-action
rate 度量语义化降级的程度。**seen/unseen composition**（reference
op-sequence 未在 train 中出现；test 集含 44 条）则是判别"真编译 vs 背模板"
的核心指标：若模型只在见过的算子组合上准确，则说明学到的是模板检索而非
编译能力。E1 之后设 Go/No-Go 闸门（`scripts/collect_phase5b1_results.py`
自动判定）：parse/valid/F1/seen 相对 E0 提升、execute 不降、unseen 不
塌方；若 seen 高企而 unseen 近零，判定 template memorization suspected，
逐例 dump 至 `unseen_cases.json` 供人工检查。

## 3. 消融实验设计（Phase 5B-2）

消融的目的不是堆实验数量，而是把"数据质量闸门是必要成本"这一主张从断言
变成证据。四项消融各自只改动一个变量，其余配置（模型、步数、split、seed）
与 E1 或 E3 完全一致：

- **A1 Tier C 加入 vs 不加**：Tier C 为 39,546 条含 fallback 映射的样本
  （修复前 fallback 占全部调用的 55.2%，语义映射覆盖率仅 36.2%）。假设：
  加入 C 可提升长尾 recall，但 fallback 监督会推高 generic-action rate、
  稀释显式语义技能的学习——以 generic-action rate 与 skill F1 的此消彼长
  为判据。
- **A2 capability context ON vs OFF**：指令歧义审计显示 exact 组歧义率仅
  0.85%、且歧义组的 capability 上下文完全相同，提示上下文对消歧帮助有限；
  本消融实测其对技能选择（skill F1）与解析率的边际贡献，决定该输入特征
  去留。
- **A3 execution_target vs plan_target**：execution_target 允许
  GENERATE/VERIFY 的 policy 尾巴进入监督目标。假设：policy 尾巴引入的
  长尾序列会损害 lowering 的纯净性，表现为 op-seq exact 下降与序列长度
  上浮；若不显著，则可在后续轮次安全启用更完整的执行视图。
- **A4 参数保全修复前 vs 后**：用 v3（参数保全率 mean 16.7%，suspect
  64.6%）与 v3.1（98.9%，1.06%）两版语料同配置重训。这是对"lifter 级
  修复、不动 IR/ISA/split"的对照实验，直接证明数据质量闸门是
  load-bearing 的（对应 outline E8），也是对 reviewer"你们的数据修复只是
  工程细节"质疑的最强回应。

## 4. 对比基线

| 基线 | 设置 | 回答的问题 |
|---|---|---|
| LLMCompiler 式规划器 | 冻结 LLM + few-shot，输出 JSON DAG（`$`-变量） | 无 IR 契约的 prompt 规划能否通过三级闸门（对应 outline E7b） |
| ReAct | 冻结 LLM 顺序执行（toolbench_static 有 ReAct 真值可对齐） | 顺序调用 vs 编译并行的延迟/调用次数差（outline E7a 之序列形态） |
| Qwen base zero-shot | 即 E0/E2 | 训练本身的贡献 |
| 70B single-shot | 直答，不编译 | 成本参照系（名义 ~2500 ms / ~900 J / ~140 GB；已有编排侧 energy 3.4%、makespan 37.9% 的 5 程序名义结果，待全量 sweep） |

基线设计遵循两条原则。**协议公平**：所有方法收到相同的任务输入与工具/
能力描述，产出统一送入同一套三级闸门与编译侧指标，LLMCompiler 与 ReAct
的原始输出格式（JSON DAG、ReAct 文本轨迹）不因格式不同而受罚——官方
metric 一列保证它们在各自原生形态下被公平评分。**对比有界**：与
LLMCompiler 的对比聚焦编译质量与调用次数（其论文的 3.7× 加速引用为标杆
而非靶子）；70B 直答仅作成本参照，不声称质量可比，名义成本一律标注
nominal。ReAct 的真值对齐点选在 toolbench_static（2,356 条含 ReAct 轨迹
静态集），可同时报官方指标与逐轮命中率。

## 5. 结果表格模板（训练后回填，禁止预填）

**T7 主结果：E0–E3 × 编译质量指标链**（test split，n=1,561）

| 实验 | parse% | valid% | exec% | op-seq% | GES | skill-P | skill-R | skill-F1 | generic-act% | seen n | seen op-seq% | unseen n | unseen op-seq% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 3B zero-shot | | | | | | | | | | | | | |
| E1 3B QLoRA | | | | | | | | | | | | | |
| E2 7B zero-shot | | | | | | | | | | | | | |
| E3 7B LoRA | | | | | | | | | | | | | |

**T7-b per-source 分解**（E3）

| source | n | parse% | valid% | exec% | op-seq% | skill-F1 | generic-act% | unseen op-seq% |
|---|---|---|---|---|---|---|---|---|
| xlam | | | | | | | | |
| spider | | | | | | | | |
| toolbench_static | | | | | | | | |
| verilogeval | | | | | | | | |
| humaneval | | | | | | | | |
| mbpp | | | | | | | | |

**T7-c per-skill P/R/F1**（31 skills + VERIFY/SELECT，E3，support 列回填）

| skill | precision | recall | F1 | support |
|---|---|---|---|---|
| （逐 skill 行，回填时生成） | | | | |

**T8 外部三层套件双指标**（每 suite 一行；官方列空缺处需环境，如实标注）

| suite | layer | n | 官方 metric | parse% | valid% | exec% | skill-F1 | generic-act% |
|---|---|---:|---|---|---|---|---|---|
| BFCL V4 | A | 4,696 | | | | | | |
| ToolBench full | A | 124,345 | | | | | | |
| AgentBoard tool-query | A | 60 | | | | | | |
| τ³-bench | B | 2,546 | | | | | | |
| AgentBoard tool-operation | B | 40 | | | | | | |
| AgentBoard webshop | B | 251 | | | | | | |
| BIRD mini-dev | C | 500 | | | | | | |
| RTL-Repo | C | 1,174 | | | | | | |

**TA 消融结果**

| 消融 | 对照配置 | parse% | valid% | exec% | op-seq% | skill-F1 | generic-act% | unseen op-seq% |
|---|---|---|---|---|---|---|---|---|
| A1 | A+B vs A+B+Tier C | | | | | | | |
| A2 | ctx OFF vs ON | | | | | | | |
| A3 | plan_target vs execution_target | | | | | | | |
| A4 | v3 vs v3.1（参数保全） | | | | | | | |

**TB 基线对比（质量 + 名义成本）**

| 方法 | 官方 metric（主 suite） | skill-F1 | valid% | LM/API 调用数 | 名义 latency | 名义 energy | 名义 memory |
|---|---|---|---|---|---|---|---|
| Ours（E3） | | | | | | | |
| LLMCompiler 式 | | | | | | | |
| ReAct | | | | | | | |
| Qwen zero-shot | | | | | | | |
| 70B single-shot | — | — | — | | | | |

## 6. 写作纪律

(i) 所有数据规模与版本引用 sha256 manifest，不引用记忆中的数字；(ii) 名义
成本一律标注 nominal，不与实测混排；(iii) 官方 metric 因环境不可离线 replay
而空缺处如实标注，不以编译侧指标替代或外推；(iv) 表格数字一律来自
`scripts/run_neural_compiler.py`、`scripts/collect_phase5b1_results.py` 与
外部评测适配器的落盘产物，禁止手抄改动；(v) 每张表在正文有一句对应的解读
句式预设（如 E1 vs E0 的差值即"训练的贡献"、E3 vs E2 的差值即"规模之上
再训练的增益"），避免罗列数字而不论证。
