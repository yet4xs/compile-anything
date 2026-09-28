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

## 下一步（Phase 2 候选）

- 真实 xLAM / ToolBench 数据接入（`run_xlam_pipeline.py --input`）
- optimizer passes（fusion / DCE / 并行化）与 scheduler 绑定
- TaskIR 文本格式 parser（compiler 前端推理输出 → 验证）
- Qwen2B Compiler SFT：用 `data/train/*.jsonl`（task → taskir_text）
