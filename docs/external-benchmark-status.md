# External Benchmark Status

Evaluation-only (see data/external_benchmarks/README.md; contamination firewall in src/dataset/external_guard.py).

| benchmark | source | version/commit | status | samples/tasks | size | license | notes |
|---|---|---|---|---:|---:|---|---|
| bfcl_v4 | ShishirPatil/gorilla | 6ea57973c7 | **READY** | 4697 | 10.6MB | Apache-2.0 (repo) |  |
| tau3_bench | codesque16/tau3-bench | f0a0173e31 | **READY** | 2546 | 148.0MB | see repo (MIT-style) | voice audio + user_simulator remain in third_party/tau3-bench@commit (pointer); snapshot = 4 domains tasks/db/policies/tools |
| agentboard | hkust-nlp/AgentBoard | bb7255e2da | **READY** | 1012 | 7438.5MB | see repo (MIT) |  |
| webshop | princeton-nlp/WebShop | 64fa2a5c15 | **PARTIAL** | - | 0.0MB | see repo (MIT) | google drive unreachable from this network; items_shuffle/items_ins_v2/items_human_ins not downloadable |
| agentbench | THUDM/AgentBench | - | **READY** | - | 0.0MB | see repo (MIT) |  |
| rtl_repo | ahmedallam/RTL-Repo (HF dataset) | - | **READY** | 4098 | 73.2MB | cc-by-4.0 (per dataset card) |  |
| bird | birdsql/bird_mini_dev (HF dataset) | - | **READY** | 1500 | 0.9MB | see dataset card |  |
| toolbench_full | OpenBMB/ToolBench (full corpus) | - | **READY** | 124345 | 1823.1MB | see upstream ToolBench |  |
| toolquery |  | - | **READY (via AgentBoard)** | - | 0.0MB |  | no separate download; AgentBoard is the canonical source |

## Totals

- raw size: **9.49 GB** (gitignored; only metadata committed)
- total tasks/samples: **132600**

TRAINING DATA (data/compiler_corpus_v3, Tier A+B) and EXTERNAL EVALUATION DATA (this directory) are strictly separated; tests/test_external_benchmark_guard.py enforces it.
