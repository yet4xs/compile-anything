# Neural Compiler Training (Phase 5B-1 runbook)

> 本目录只准备代码与数据，**不在本环境训练**（无 GPU/依赖）。整个
> `compile-anything/` 目录拷到服务器即可运行。

## 1. 权重下载（国内环境建议 ModelScope）

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./weights/Qwen2.5-3B-Instruct
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./weights/Qwen2.5-7B-Instruct
```

依赖：`pip install transformers peft accelerate bitsandbytes trl datasets
pyyaml`（CUDA 环境）。

## 2. 数据（已在仓库生成）

- `data/compiler_corpus_v3/{train,val,test}.jsonl` — Tier A+B（第一轮
  SFT 只用这些），group-aware 切分，exact leakage = 0
- `data/compiler_corpus_v3/tier_c.jsonl` — Tier C（fallback/synthetic），
  仅用于 ablation/augmentation
- 每条含 `plan_target`（纯 lowering，**第一轮训练目标**）、
  `execution_target`（含 policy tail，备用）、`capabilities`（语义能力
  上下文，输入 view B 用）

```bash
# 生成 chat 格式（可选，train_lora.py 也会现场构建）
python scripts/prepare_sft.py                        # plan_target, A+B
python scripts/prepare_sft.py --capability-context   # 输入 view B
```

## 3. 训练

```bash
# 3B sanity（24GB 卡默认 QLoRA 4bit；48GB 卡 --quantize none）
python -m src.compiler.train.train_lora \
    --config src/compiler/train/config/qwen3b.yaml \
    --model ./weights/Qwen2.5-3B-Instruct \
    --train data/compiler_corpus_v3/train.jsonl \
    --val   data/compiler_corpus_v3/val.jsonl \
    --out runs/qwen3b-lora

# 7B 主实验（48GB 卡 LoRA；24GB 用 4bit QLoRA）
python -m src.compiler.train.train_lora \
    --config src/compiler/train/config/qwen7b.yaml \
    --model ./weights/Qwen2.5-7B-Instruct \
    --train data/compiler_corpus_v3/train.jsonl \
    --val   data/compiler_corpus_v3/val.jsonl \
    --out runs/qwen7b-lora
```

关键开关：`--target-field plan_target|execution_target`、
`--capability-context`（view A/B 对照）、`--tiers A,B`。

## 4. 推理 + 评测（禁止只报 loss）

```bash
python scripts/run_neural_compiler.py --model runs/qwen7b-lora/final \
    [--capability-context] [--limit 500]
```

指标（`benchmark/neural_compiler_eval.py` →
`data/reports/neural_compiler_eval.{json,md}`）：
parse rate → validator pass → execution success；
op-sequence exact、per-skill precision/recall、graph edit similarity、
generic-action rate；分 source；**seen vs unseen semantic composition**
（unseen = 参考算子序列未在 train 出现——背模板 vs 真编译的判别指标）。

## 5. 显存参考

| 配置 | 卡 | 方式 |
|---|---|---|
| Qwen2.5-3B | 24GB (4090) | QLoRA 4bit（qwen3b.yaml 默认） |
| Qwen2.5-7B | 48GB (A6000/L40) | LoRA bf16（qwen7b.yaml 默认） |
| Qwen2.5-7B | 24GB (4090) | `--quantize 4bit` QLoRA，batch_size 降 1 |

## 6. 复现数据管线（如需重建 corpus v3）

```bash
python scripts/download_datasets.py                # 7 个真实数据集
python scripts/build_real_corpus.py                # v3: tiers + group split
python scripts/audit_training_corpus.py            # 质量审计
```
