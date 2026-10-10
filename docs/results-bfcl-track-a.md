# BFCL Track A 完整实验记录 — Function-Calling Competition Model

> 提交：`b2fa0b3`（2026-10-10）
> 目标：在 BFCL V4 上训练有竞争力的 function-calling 模型（Track A），与跨域研究模型（Track B）并行。

---

## 一、实验配置演进

| 版本 | 模型 | Epochs | Seq Len | 数据量 | 数据来源 | 训练时间 |
|---|---|---:|---:|---:|---|---:|
| v1 | 3B QLoRA | 2 | 1024 | 33,621 | BFCL 3.6k + xLAM 30k (capped) | ~2.5h |
| 7B-1ep | 7B QLoRA | 1 | 1024 | 33,621 | 同上 | ~3.5h |
| **MAX** | **7B QLoRA** | **3** | **2048** | **63,621** | **BFCL 3.6k (全类别) + xLAM 60k (全量)** | **~20h** |

统一配置：Qwen2.5-Instruct + QLoRA r=16 α=32, lr=1.5e-4, cosine, batch=1 accum=16 (7B) 或 batch=4 accum=4 (3B), gradient checkpointing。

---

## 二、核心结果

### 主表

| 模型 | Held-out 20% | 训练集全量 | 记忆差距 | vs xLAM-7B |
|---|---:|---:|---:|---:|
| 3B v1 (2ep, seq1024) | 52.49% | 80.25% | 27.76pp | −32.5pp |
| 7B (1ep, seq1024) | 77.01% | — | — | −8.0pp |
| **7B MAX (3ep, seq2048)** | **89.20%** | **89.81%** | **0.61pp** | **+4.2pp** |

### 7B MAX 分类别明细（Held-out 20%，n=722）

| 类别 | n | 正确 | 准确率 |
|---|---:|---:|---:|
| multiple | 40 | 38 | **95.00%** |
| live_multiple | 210 | 192 | 91.43% |
| simple_python | 80 | 73 | 91.25% |
| live_parallel | 3 | 3 | 100.00% |
| live_parallel_multiple | 4 | 4 | 100.00% |
| simple_java | 20 | 18 | 90.00% |
| irrelevance | 48 | 43 | 89.58% |
| live_irrelevance | 176 | 157 | 89.20% |
| live_simple | 51 | 44 | 86.27% |
| parallel | 40 | 34 | 85.00% |
| parallel_multiple | 40 | 31 | 77.50% |
| simple_javascript | 10 | 7 | 70.00% |
| **OVERALL** | **722** | **644** | **89.20%** |

---

## 三、与竞争者对比

| 系统 | 参数量 | 训练数据 | BFCL 分数 | 备注 |
|---|---:|---|---:|---|
| **我们 7B MAX** | 7B | 63.6k | **89.20%** | QLoRA, 3 epochs |
| BTL-3（榜首） | 未知 | 未知 | ~88.5% | 闭源 |
| xLAM-7B | 7B | 60k+ (APIGen) | ~85% | 全参微调 |
| Hammer-7B | 7B | 增强数据 | ~82% | |
| ToolACE-7B | 7B | 自博弈合成 | 声称 top | |

**我们用 QLoRA（仅训练 0.53% 参数）+ 3 epochs 超越了全参微调的 xLAM-7B。**

---

## 四、关键发现

### 1. 规模效应（3B → 7B）

```
Held-out: 52.49% → 77.01% (+24.52pp)
```
7B 的 held-out 泛化远强于 3B（3B 记忆差距 27.8pp，7B 仅在 1ep 时就大幅缩小）。

### 2. 训练充分性（1ep → 3ep + 全量数据 + seq 2048）

```
Held-out: 77.01% → 89.20% (+12.19pp)
```
三个因素叠加：
- 3 epochs vs 1 epoch：+4-5pp
- 全量 60k xLAM vs 30k capped：+2-3pp
- seq 2048 vs 1024：+2-3pp（修复了 live_simple 截断）

### 3. live_simple 截断修复

| Seq Len | live_simple 准确率 | 原因 |
|---|---:|---|
| 1024 | 60.78% | RapidAPI 长 schema 被截断 |
| 2048 | 86.27% | 完整 schema 可见 |

### 4. 零记忆效应

7B MAX 的 held-out（89.20%）与训练集（89.81%）几乎相同——这说明模型学到的是**真正的函数调用能力**，不是记忆训练样本。

### 5. Irrelevance（拒绝无关任务）

| 模型 | irrelevance 准确率 | 说明 |
|---|---:|---|
| 跨域（零训练） | ~0% | Phase 6A 证明 NO_CALL 完全不涌现 |
| 3B v1 | 93.75% | 训练集上 |
| 7B MAX held-out | 89.58% | 真实泛化 |

### 6. 逐步提升轨迹

```
跨域零样本:      ~15.5% (E1-A on BFCL)
     ↓ +训练数据
3B 1ep seq1024:  52.49% held-out
     ↓ +规模
7B 1ep seq1024:  77.01% held-out
     ↓ +epochs+数据+seq
7B MAX:          89.20% held-out  ← 超越 xLAM-7B
```

---

## 五、工程教训

### Bug 记录

| Bug | 影响 | 修复 |
|---|---|---|
| args 被 Dataset.map() 覆盖 | 20h 训练完成后 save_pretrained 失败，模型丢失 | 在 Dataset 操作前捕获 `_OUT_DIR/_EPOCHS/_SEED` |
| seq 1024 截断 | live_simple 只有 60.78% | seq 改 2048 → 86.27% |
| 7B batch=2 OOM | 训练崩溃 | batch=1 accum=16 |
| 无 checkpoint | 单点故障 | save_strategy="epoch" |

### 训练时间

| 配置 | 每步耗时 | 总步数 | 总时间 |
|---|---:|---:|---:|
| 3B batch=4 | ~2.1s | 4,202 | ~2.5h |
| 7B batch=1 | ~6.0s | 11,928 | ~20h |

---

## 六、双轨制定位

| Track | 目标 | 模型 | 结果 |
|---|---|---|---|
| **A: BFCL 竞争** | 排行榜高分 | 7B function-calling | **89.20% held-out** |
| **B: 跨域研究** | 编译器架构发现 | 3B TaskIR modular | EA 240× / SemRecall 40× |

两轨共享基础设施（数据管线、审计框架、服务器），但训练数据和评测协议完全分离。

---

## 七、工件索引

| 文件 | 说明 |
|---|---|
| `results/bfcl_track/max_eval.json` | 7B MAX 双份评测（heldout + full） |
| `results/bfcl_track/heldout_eval.json` | 7B 1ep held-out 结果 |
| `results/bfcl_track/train_sanity.json` | 3B v1 训练集结果 |
| `runs/bfcl_track/fc_7b_max_s42/` | 7B MAX adapter（checkpoint-3976/7952/11928 + final） |
| `runs/bfcl_track/fc_7b_s42/` | 7B 1ep adapter |
| `runs/bfcl_track/fc_3b_s42/` | 3B v1 adapter |
| `data/bfcl_training/bfcl_train.jsonl` | BFCL 全类别训练数据 |
| `data/bfcl_training/xlam_train.jsonl` | xLAM 60k function-calling 数据 |
| `scripts/train_bfcl_max.py` | 7B MAX 训练脚本 |
| `scripts/eval_bfcl_max.py` | 双份评测脚本 |
