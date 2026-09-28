# Compile Anything

> **A Neural Compiler and Runtime Architecture for General AI Tasks.**
> Phase 1: "LLVM for AI tasks" 的最小闭环 —— NL task → TaskIR → validator →
> runtime simulator → execution trace + cost report。
> 这不是 Agent Framework，是一条 AI 任务编译器 + 异构执行架构的研究原型。

```
Human Task ──▶ Neural Compiler ──▶ TaskIR ──▶ Validator ──▶ Runtime Simulator ──▶ Trace + Cost
              (Phase 1: 手工/lift;   (LLVM IR)  (verify pass)   (mock executors)     (workload)
               Phase 2: Qwen2B SFT)
```

## 论文结构映射

| 论文章节 | 对应组件 |
|---|---|
| §3 Neural Compiler | `src/compiler/`（prompt 格式、SFT 数据）+ `src/lifter/`（数据前端） |
| §4 TaskIR | `spec/taskir-spec.md` + `src/ir/` + `src/validator/` |
| §4.5 Skill ISA | `spec/skill-isa.md` + `src/isa/`（31 skills + 2 控制指令，含 latency/FLOPs/energy/memory 成本） |
| §5 Runtime Architecture | `src/runtime/`（依赖驱动执行、VERIFY、retry 回滚、成本核算）；scheduler/optimizer 为 v0.2 接口 |
| §6 Evaluation | `data/reports/`（cost report + dataset statistics）+ `benchmark/`（规划中） |

## 目录

```
compile-anything/
├── spec/
│   ├── taskir-spec.md        # TaskIR v0.1 规范（SSA/类型/控制流/序列化/不变量）
│   └── skill-isa.md          # Skill ISA v0.1（31 条 skill + 2 条控制指令 + 成本模型）
├── src/
│   ├── ir/                   # TaskIR 数据结构 + JSON canonical + 文本 printer
│   ├── isa/                  # Skill registry（签名+成本：latency/flops/energy/memory）
│   ├── validator/            # V1–V6 静态验证 + 死代码警告（数据质量闸门）
│   ├── lifter/               # xLAM → TaskIR（tool → semantic skill 两层 lowering）
│   ├── compiler/             # Neural Compiler 训练侧：prompt 格式 + SFT 数据构建
│   ├── runtime/              # 依赖驱动模拟器：guard / VERIFY / retry 回滚 / trace
│   ├── cost/                 # 成本聚合 + 报告（名义成本，含 70B baseline 对照）
│   ├── optimizer/            # placeholder（仅 pass manager 接口，v0.2）
│   └── scheduler/            # placeholder（executor 绑定与调度，v0.2）
├── benchmark/                # 评测规划（xlam/toolbench/rtl，见其 README）
├── data/
│   ├── raw/                  # xLAM-schema 样例（合成，见其 README）
│   ├── taskir/               # examples/3 + synthetic/1000 + xlam/100（全部过 validator）
│   ├── train/                # (NL task → TaskIR) 训练对 jsonl，给 Qwen2B Compiler
│   └── reports/              # cost report + dataset statistics
├── scripts/                  # demo / 生成器 / 管线 / 统计 / QC 闸门
└── tests/                    # 34 个 unittest
```

## 快速开始

```bash
# 端到端 demo（NL → TaskIR → validate → simulate → cost report）
python scripts/demo_flight.py

# 生成/消费数据
python scripts/gen_synthetic.py -n 1000 --seed 42     # 1000 条 synthetic TaskIR
python scripts/gen_xlam_sample.py -n 100              # xLAM-schema 样例
python scripts/run_xlam_pipeline.py                   # lift + validate + 训练对
python scripts/build_sft.py                           # Qwen2B Compiler SFT 数据集

# 统计与 QC
python scripts/dataset_stats.py                       # workload characteristics
python scripts/validate_all.py                        # 数据质量闸门（非零退出=有非法）

# 测试
python -m unittest discover -s tests
```

## Phase 1 验收对照

| # | 验收项 | 状态 | 位置 |
|---|---|---|---|
| 1 | TaskIR spec | ✅ | `spec/taskir-spec.md`（v0.1）|
| 2 | 可运行 validator | ✅ | `src/validator/`，V1–V6 + 警告 |
| 3 | simulator | ✅ | `src/runtime/`，guard/verify/retry 回滚/trace/成本 |
| 4 | 1000 条 synthetic TaskIR | ✅ | `data/taskir/synthetic/`（100% valid）|
| 5 | 100 条 xLAM 转换 TaskIR | ✅ | `data/taskir/xlam/`（schema 样例，见 raw/README）|
| 6 | cost report | ✅ | `data/reports/cost_report_flight.md`（latency/FLOPs/energy/memory + executor + critical path） |

Dataset statistics（写入 `data/reports/dataset_stats.md`）：1105 programs，
nodes mean 4.79，depth mean 4.35，width mean 1.44，43.6% 含并行结构，
47.1% 含 retry。

**Neural Compiler SFT 数据**：`python scripts/build_sft.py` 从训练对构建
chat 格式数据集（`data/train/sft/{train,val}.jsonl`，当前 593/31 条——
按 NL 任务文本去重后的唯一任务数；换入真实 xLAM 后规模随之增长）。

## 核心设计决定（Phase 1）

1. **两层表示**：TaskIR 平台无关（语义/数据流/控制流）；Skill ISA 是目标机
   指令集。工具名永不进入 TaskIR 语义部分（只留 `meta.provenance`）。
2. **分支不用跳转**：guard 谓词执行 + SELECT 汇聚（类比 GPU predication），
   保持扁平 SSA，利于小模型生成训练数据。
3. **验证/重试是一等 IR 结构**：VERIFY 产 Bool；`retry(on=%v)` 触发运行时
   回滚（失效下游记忆并重执行）。
4. **validator 是数据闸门**：所有进入 `data/taskir/` 的记录必须通过 V1–V6。
5. **成本模型从第一天就有**：名义 FLOPs/tokens/latency + 70B 单次直答
   baseline，为 "small executors + compilation vs 大模型" 的对照实验铺路。

## Phase 2：语义压力验证（stress validation）

不堆 feature，验证 IR 语义本身：

- `tests/test_control_semantics.py` — guard/VERIFY/SELECT 谓词执行、嵌套
  guard、非法 guard（V4 拒绝）、SELECT 选中分支被跳过/谓词传播
- `tests/test_retry_semantics.py` — error retry 事件序、VERIFY 回滚的
  memo 失效（A→B→C 链全量重执行）、非法 retry 构造拒绝
- `scripts/fuzz_taskir.py` — semantic fuzzer：10000 随机程序，
  builder 合法构造 ⇒ validator 必须接受；runtime 只允许**已定义**失败；
  IV1–IV6 不变量（DAG 顺序/attempts 单调/确定性/输出非空/成本非负/
  Bool 产 Bool）。失败自动落盘 `data/fuzz_failures/`
- `docs/validator-audit.md` — V1–V6 审计 + M1 effect ordering（最重要
  真实缺口）/M2 resource legality/M3 cost consistency
- `docs/runtime-review.md` — memo≡SSA cache？回滚覆盖完整性、critical
  path 与 retry 边、本轮发现并修复的 3 个语义缺陷

fuzz 结果：**10000/10000 valid，0 failures**（1719 例落入已定义失败路径：
consume-skipped / chosen-branch-skipped / retry 耗尽——随机 guard + 故障
注入下的预期行为）。

## Phase 3：Effect System（设计）+ Scheduling Foundation（实现）

把 TaskIR 从"描述任务的数据流"推向"可调度执行的程序表示"：

- `docs/effect-system-proposal.md` — **设计先行，未动 IR 代码**。Effect
  token（类 MemorySSA）：action 节点 `effect_in/out` 成链、纯 skill 零开销；
  回答 Q1（token 是受约束的 SSA value）/Q2（effect 边进 DAG 与 critical
  path）/Q3（pure skill 无 token）；含运行时语义（guarded-skip 直通、
  world 类禁 verify-retry、rollback 不得跨越已推进的 effect 链）
- `docs/validator-v7-proposal.md` — V7 检查集（EFFECT_CLASS/UNDEF/DUP/
  FORK/GAP/CYCLE/RETRY/TYPE）+ 0.1 警告→0.2 错误的迁移计划
- `src/optimizer/scheduler/list_scheduler.py` — **第一版真 scheduler**：
  资源约束 list scheduling（critical-path 优先），依赖/after/guard 边 +
  按资源类的 executor pool；`_edges()` 预留 effect 边接入点
- `tests/test_scheduler_cost.py` — 方案 A（70B 单发）vs 方案 B
  （SEARCH→FILTER→VERIFY→GENERATE 流水线）的 latency/energy/memory 全轴
  对照 + 并行/串行/效应排序调度正确性（10 用例）
- `benchmark/taskir_runtime_benchmark.py` — 基准协议：TaskIR json →
  {latency, makespan, critical path, energy, memory, lm/api calls,
  schedule}，输出 `data/reports/benchmark_results.{json,md}`

当前基准（examples，池=1/类）：mean energy = 70B baseline 的 **3.4%**，
mean makespan = 37.9%；rtl_debug 的 makespan(1070ms) < 顺序和(1098ms)
来自跨资源类真并行（EXTRACT∥SEARCH）。

## Phase 4：Benchmark Lifting + Compiler Corpus

回答"TaskIR 是不是合格的 AI machine language"——开源 benchmark 形态数据
→ TaskIR → validator → runtime 的**真实前端**闭环：

- `src/ir/parser.py` — **TaskIR 文本形式解析器**（printer↔parser roundtrip
  保障；Phase 5 模型输出文本 IR 的入口闸门）
- `src/lifter/benchmark/` — 统一 lifter 框架（`can_handle`/`lift`）+
  四个实现：toolbench（tool→semantic skill，工具名禁入语义层）、
  humaneval（ast 模式抽取：sorted→SORT、min/max→ARGMIN/MIN、sum→SUM、
  filter→FILTER、map→TRANSFORM、a+b→JOIN）、spider（SQL 保守 lowering，
  GROUP/HAVING/JOIN 不硬映射）、rtl（EDA 调试任务）
- `scripts/build_compiler_corpus.py` — 10000 样本 corpus
  （toolbench 4000 / code 3000 / sql 2000 / rtl 1000，schema-faithful
  synthetic，`--raw-dir` 可换真实数据）：**总覆盖率 95.02%**，
  code 83.4%（损失全部来自 IR 无 LOOP，见 missing-skills.md #1）
- `data/schema/compiler_sample.json` — 统一 corpus record schema
  （source/input/raw_trace/taskir/taskir_text/validation/execution）
- `benchmark/compiler_eval.py` — 三级评测闸门（**不用 exact match**）：
  syntax accuracy → validator pass → semantic execution；全 corpus
  **100% / 100% / 100%**；`--predictions` 即 Phase 5 模型接口
- `docs/dataset-design.md` + `docs/missing-skills.md` — 数据设计与 ISA
  缺口清单（LOOP 最高优先、SQL 关系算子、EDA 工具链、字符串、粒度）

评测中抓到并修复：simulator 的 @global 程序输入此前无运行时值（返回
类型名），导致 code 类程序 1/3 无法执行——现按声明类型合成 mock 值。

## Phase 5A：真实 Benchmark 数据生产线

**真实数据 → adapter → TaskIR lifter → validator → simulator → corpus**，
零 synthetic 冒充（synthetic 仅以 `source=synthetic:*` 标签并入）：

- `src/dataset/` — registry（9 个 DatasetSpec：url/license/task_type/计划
  + 不可达原因如实记录）、统一 raw-sample schema（原数据保留在
  `raw_payload`）、stdlib 下载器（sha256 + `data/raw/metadata.json`，
  原始数据 gitignore）
- `src/dataset/adapters/` — tooluse/code/sql/rtl 四类（load+normalize），
  只做字段映射，lifting 逻辑不复制
- `scripts/download_datasets.py` — 实拉 **7 个真实数据集（71855 样本）**：
  xLAM 60000（ModelScope 国内镜像，HF 原仓 token 门控）、Spider 8034、
  ToolBench-Static 2356（ModelScope）、MBPP 974、VerilogEval 312、
  HumanEval 164、ToolBench 样例 15；BIRD/API-Bank/AgentBench/ToolBench
  全量因门槛或体积不可达，已注册（downloader 支持 file/github-dir/
  modelscope 三种 plan）
- `scripts/analyze_dataset_coverage.py` — 覆盖率/拒绝直方图/图统计 →
  `data/reports/dataset_coverage.{json,md}`
- `scripts/build_real_corpus.py` — **compiler_corpus_v2**：真实 70761 +
  标注 synthetic 9027（train 71810 / val 3989 / test 3989）
- `docs/dataset-audit.md` — 四问回答（真实数字）

**核心发现（真实覆盖率，非 synthetic 估计）**：tool-use/SQL/RTL 全部
100%（xLAM 6 万条全过）；**真实 HumanEval 仅 7.3%、MBPP 仅 3.3%**——
LOOP/递归 524 个拒绝是主因（ISA 缺口第一优先），局部变量赋值返回 137
个是 lifter 能力问题（赋值=SSA def，可修）。v0.2 ISA 优先级被真实
数据重排：LOOP > STRING_OP > GROUP/TABLE_SCAN；JOIN 无需新增。

## Phase 5B-0：Corpus 质量关（quality gate）

诊断三个"指标虚高"来源并修复——fallback 冒充语义、policy 尾巴当监督
信号、随机切分泄漏：

- **Task 1** `toolmap.map_tool()` 返回 `mapping_kind`（exact/heuristic/
  fallback）+ `matched_rule` + `confidence`；provenance 只进
  `meta.provenance.lowering`，不污染语义体（有测试断言）
- **Task 2** `scripts/audit_training_corpus.py` →
  `data/reports/training_corpus_audit.{json,md}`：**fallback 占调用
  55.2%**（EXEC_ACTION 中 56,588 unknown-fallback vs 2,382
  intentional）；xLAM 语义覆盖（零 fallback）仅 **36.2%**（21,736 条）；
  新指标 `semantic_mapping_coverage` 与 lift/validator/execution 覆盖
  分离上报
- **Task 3** 双 view：`plan_target`（纯 lowering，无 GENERATE/VERIFY——
  第一轮 SFT 目标）与 `execution_target`（允许 policy 尾巴）；同一
  trajectory 两个 view 均 validator 通过（有测试）
- **Task 4** `src/dataset/dedup.py`（MinHash-LSH 近重复 + 同 op-seq
  并查集 + group-aware 切分）：**cross-split exact = 0，near = 0**
- **Task 5** instruction-only 歧义审计：exact 组歧义率 **0.85%**、
  近重复族 **3.72%**，且歧义组 capability 上下文全部相同（上下文无
  助于消歧）——单指令输入对 xLAM 基本充分
- **Task 6** quality tiers：**A 15,755 / B 15,460 / C 39,546**（真实
  数据），A+B 进 `data/compiler_corpus_v3/{train,val,test}.jsonl`
  （28,093/1,561/1,561），C 另存 `tier_c.jsonl` 仅供 ablation
- **Task 7** 训练 harness（未训练）：`src/compiler/train/`（dataset/
  train_lora/infer + qwen3b/qwen7b 配置，LoRA+QLoRA，模型路径全参数化）
  + `scripts/prepare_sft.py` + `scripts/run_neural_compiler.py` +
  `benchmark/neural_compiler_eval.py`（parse→validator→execution 三级
  闸门、op-seq/技能 P-R/图编辑相似度/generic-action 率、**seen vs
  unseen composition**）+ `docs/training.md`（ModelScope 权重下载 +
  服务器 runbook）

顺带修复（oracle 自检发现）：多行 description 的 printer/parser
roundtrip 漏洞（HumanEval docstring/ToolBench 换行指令曾导致两个源
plan_target 不可解析）；eval 报告结构 bug。

## Phase 5B-1：可复现 baseline 训练包（本机零训练，零伪造数字）

冻结 + 服务器包，本机完成所有 GPU-free 工作：

- **冻结**：`experiments/phase5b1/manifest.json`（commit、corpus v3
  train/val/test SHA256、tier 计数、TaskIR/prompt 哈希、seed、依赖 pin）；
  `tests/test_phase5b1_freeze.py` 守护冻结文件（改动即 fail）
- **依赖锁定**：`requirements-training.txt`（torch 2.5.1 / transformers
  4.46.3 / peft 0.13.2 / trl 0.12.2 / bnb 0.44.1 …，pip --dry-run resolve
  验证）；`train_lora.py` 适配 TRL 0.12 `SFTConfig` API + 0.13+ 改名
  shim、`--max-steps` sanity、auto-resume、log_history 落盘
- **preflight**：`scripts/training_preflight.py`（版本核对/CUDA+bf16/4bit
  加载/chat template/LoRA target 存在/10 条 tokenize/forward/1 步
  optimizer/checkpoint 往返；失败禁止开训；`--static` 模式本机已全绿）
- **权重**：ModelScope 国内源下载中（3B/7B），`weights/MANIFEST.json`
  记录文件清单与 sha256；`weights/` 已 gitignore
- **token 审计**：`scripts/audit_token_lengths.py`（近似模式已跑：
  total p50=381 / p99=846，>2048 仅 0.04% → 2048 定长合理）
- **实验编排**：`scripts/run_phase5b1.sh`（preflight/e0/e1-sanity/e1/e2/
  e3/eval-all）+ `scripts/collect_phase5b1_results.py`（E0-E3 表 +
  per-source + E0-vs-E1 Go/No-Go 闸门 + template-memorization 判定；
  未跑实验标 PENDING，不造数）
- **eval 增强**：skill F1（micro + per-skill）、`--dump-unseen` 逐例
  dump（oracle 自检 44 条 unseen 全字段）

## 下一步（Phase 5B-1：Qwen Neural Compiler LoRA 实验）

- 训练数据：`data/compiler_corpus_v3/`（A+B，plan_target 目标，
  28,093/1,561/1,561，zero leakage）——runbook 见 `docs/training.md`
- 3B sanity（QLoRA）→ 7B 主实验（LoRA/QLoRA）；`--capability-context`
  做 view A/B 对照（审计显示 xLAM 上提升有限，留作消融）
- 评测走 `scripts/run_neural_compiler.py`：三级闸门 + seen/unseen
  composition（判别"编译 vs 背模板"的核心指标）
- 效果系统 v0.2 实施（docs/effect-system-proposal.md 落地顺序）
- code 域覆盖率前置修复：lifter"赋值=SSA"（预计 HE/MBPP 3-7% → ~15%）
  与 LOOP region（docs/missing-skills.md #1）
