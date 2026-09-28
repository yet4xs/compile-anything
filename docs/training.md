# Neural Compiler Training (Phase 5B-1 runbook)

> 本目录只准备代码与数据，**不在本环境训练**（无 GPU/依赖）。整个
> `compile-anything/` 目录拷到服务器即可运行。
> **冻结基线**：`experiments/phase5b1/manifest.json`（corpus SHA256 +
> toolmap/spec/prompt 哈希）；`tests/test_phase5b1_freeze.py` 守护——
> 实验期间改动冻结文件会直接 fail 测试。发现问题记录 issue，留到 5B-2。

## 0. 一键流程（服务器）

```bash
pip install -r requirements-training.txt     # 版本已锁定 + resolve 验证
bash scripts/run_phase5b1.sh preflight       # 失败禁止开训
bash scripts/run_phase5b1.sh e0              # 3B zero-shot baseline
bash scripts/run_phase5b1.sh e1-sanity       # 3B QLoRA 50 步 sanity
bash scripts/run_phase5b1.sh e1              # 3B QLoRA 全量（自动 resume）
#   → 检查 Go/No-Go：python scripts/collect_phase5b1_results.py
bash scripts/run_phase5b1.sh e2              # 7B zero-shot（3B 过闸后）
bash scripts/run_phase5b1.sh e3              # 7B 主实验
bash scripts/run_phase5b1.sh eval-all        # 汇总表 data/reports/phase5b1_results.md
python scripts/make_phase5b1_manifest.py --runtime   # 回填 CUDA/torch 信息
```

TRL 版本说明：锁定 `trl==0.12.2`（数据参数在 `SFTConfig`）；
`train_lora.make_sft_args_class()` 带 0.13+（`max_seq_length→max_length`
改名）兼容 shim；preflight 会核对已装版本与 pin 的一致性。

## 1. 权重（已用 ModelScope 国内源下载，禁止提交 Git）

- `weights/Qwen2.5-3B-Instruct/`（E0/E1）
- `weights/Qwen2.5-7B-Instruct/`（E2/E3，48GB 卡 bf16 LoRA / 24GB 卡
  `--quantize 4bit` QLoRA batch_size=1）
- `python scripts/verify_weights.py [--full-hash]` → `weights/MANIFEST.json`
  （文件清单/大小/关键文件 sha256）

```bash
# 重下（如需）
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir weights/Qwen2.5-3B-Instruct
modelscope download --model Qwen/Qwen2.5-7B-Instruct  --local_dir weights/Qwen2.5-7B-Instruct
```

## 2. 数据（冻结）

- `data/compiler_corpus_v3/{train,val,test}.jsonl` = Tier A+B
  （28,093/1,561/1,561），**目标 = plan_target**，capability context OFF
- token 审计（近似，`scripts/audit_token_lengths.py`，服务器可用
  `--tokenizer` 精确化）：total p50=381 / p99=846，>2048 仅 0.04% ——
  2048 定长合理，本轮 frozen

## 3. 实验矩阵与闸门

| 实验 | 模型 | 训练 | 说明 |
|---|---|---|---|
| E0 | 3B zero-shot | 0 | 基线下界 |
| E1 | 3B QLoRA | 28,093 | sanity 50 步 → 全量 |
| E2 | 7B zero-shot | 0 | 规模对照 |
| E3 | 7B LoRA/QLoRA | 28,093 | 主实验 |

E1 后的 Go/No-Go（`collect_phase5b1_results.py` 自动判定）：parse/valid/
F1/seen 提升、execute 不降、unseen 不塌方；seen 高 + unseen≈0 触发
**template memorization suspected**（unseen 逐例 dump 在
`runs/phase5b1/*/unseen_cases.json`，test 集含 44 条 unseen composition）。

## 4. 本轮禁止（留给 5B-2 消融）

Tier C 训练 / capability context / execution_target 目标 / 扩 toolmap /
LOOP / STRING_OP / 换 split / RL / optimizer-aware training。

## 5. 显存参考

| 配置 | 卡 | 方式 |
|---|---|---|
| 3B QLoRA | 24GB (4090) | 4bit NF4 + bf16 compute（默认） |
| 7B LoRA | 48GB (A6000/L40) | bf16 LoRA |
| 7B QLoRA | 24GB (4090) | `--quantize 4bit`，batch 1 + grad accum |

## 6. 复现数据管线（重建 corpus v3，实验期间禁止）

```bash
python scripts/download_datasets.py
python scripts/build_real_corpus.py
python scripts/audit_training_corpus.py
```
