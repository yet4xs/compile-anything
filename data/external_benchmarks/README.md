# External Benchmarks — EVALUATION-ONLY

> **这个目录下的所有数据是外部评测基准（evaluation-only），永远不能进入
> 训练语料。** 防火墙：`src/dataset/external_guard.py` 在所有 corpus
> builder 入口（`iter_corpus_records` / `run_pipeline` /
> `build_real_corpus.py` / `build_compiler_corpus.py` / `prepare_sft.py`）
> 强制拦截本目录路径（RuntimeError）；Phase 5B 禁止使用
> `--allow-external-training`。

| 目录 | 内容 | 状态 |
|---|---|---|
| `bfcl_v4/` | BFCL V4 全 20 类（simple/parallel/multiple/irrelevance/multi-turn×4/memory/web_search/format_sensitivity/live×7 + java/js/python）+ possible_answer | READY |
| `tau3_bench/` | τ³-bench 四域任务定义/DB/策略/工具快照（airline/retail/telecom/banking_knowledge）；voice 音频留在 third_party@commit | READY |
| `agentboard/` | 官方 data.tar.gz（hf-mirror），解压后 9 类任务 test.jsonl | READY |
| `webshop/` | 官方仓 + setup.sh 分析（数据经 Google Drive 分发，本网络不可达——见 MANIFEST attempts）；任务定义经 AgentBoard webshop/test.jsonl | PARTIAL |
| `agentbench/` | third_party/THUDM@commit（含 v0.1/v0.2 论文版本引用；数据原版经 LMUData/HF） | READY (repo) |
| `rtl_repo/` | HF ahmedallam/RTL-Repo parquet（train+test，镜像下载） | READY |
| `bird/mini_dev/` | birdsql/bird_mini_dev 三方言 JSON（镜像下载）；full BIRD 表单门槛，不可得 | READY (mini-dev) |
| `toolbench_full/` | swift/ToolBench 全量 1.82GB jsonl（ModelScope） | READY |
| `toolquery/` | **无独立副本** — canonical source 是 AgentBoard 的 tool-query/test.jsonl（见 MANIFEST hash） | READY via AgentBoard |

Provenance/完整性：每目录 sha256、上游 commit/镜像、逐条校验计数见
`MANIFEST.json` 与 `status.json`；汇总表在
`docs/external-benchmark-status.md`。原始大数据 gitignore，仓库只提交
元数据。本轮停在 download → verify → freeze：**没有 lifter、没有并入
任何 corpus、没有跑模型**（adaptation protocol 由下一轮逐基准评审）。
