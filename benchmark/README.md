# benchmark/ — 评测计划（v0.2+）

论文 evaluation 的基准集与 harness 将放在这里。规划：

| 目录 | 内容 | 用途 |
|---|---|---|
| `xlam/` | 真实 Salesforce/xLAM 测试划分 | Neural Compiler 编译正确率（生成 TaskIR 过 validator + simulator 可执行率） |
| `toolbench/` | ToolBench 任务 | 跨数据集泛化（lifter 与编译质量） |
| `rtl/` | EDA/RTL 调试任务（Verilog assertion 修复等） | DAC 相关 workload；Skill ISA 需补 EDA skill（LINT、SYNTH、SIMULATE，候选 executor：yosys/verilator） |

评测指标（对应 simulator 输出）：

- 编译成功率：NL → valid TaskIR 比例（validator 判定）
- 执行成功率：TaskIR → simulator/runtime 完成比例（含 retry 后）
- 效率：latency（sequential vs critical path）、FLOPs、energy、peak memory
- 与 70B 单次直答 baseline 的质量/成本对照（人工或 LLM-judge）

EDA skill 扩展草案（进 Skill ISA v0.2）：

```
LINT(rtl) -> Report            [python/yosys]
SYNTH(rtl, target) -> Netlist  [tool:yosys]
SIMULATE(netlist, vectors) -> Waveform [tool:verilator]
FORMAL_CHECK(prop, rtl) -> Bool [tool:sby]
```
