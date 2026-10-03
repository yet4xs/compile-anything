"""Phase 5C training script — supports 2×2 experiment matrix.

Modes:
  E5C-S:  schema ON,  depth-balanced OFF
  E5C-D:  schema OFF, depth-balanced ON
  E5C-SD: schema ON,  depth-balanced ON

Usage:
  python scripts/train_phase5c.py --mode S   # schema only
  python scripts/train_phase5c.py --mode D   # depth only
  python scripts/train_phase5c.py --mode SD  # both
"""
import json, sys, os, time, argparse, random, base64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["S", "D", "SD"])
parser.add_argument("--corpus", default="data/compiler_corpus_v3_1/train.jsonl")
parser.add_argument("--val", default="data/compiler_corpus_v3_1/val.jsonl")
parser.add_argument("--out", default="")
parser.add_argument("--max-oversample", type=float, default=5.0)
args = parser.parse_args()

SCHEMA_ON = "S" in args.mode
DEPTH_ON = "D" in args.mode

if not args.out:
    args.out = f"runs/phase5c/e5c_{args.mode.lower()}"

print(f"Phase 5C Training — Mode: E5C-{args.mode}", flush=True)
print(f"  Schema conditioning: {'ON' if SCHEMA_ON else 'OFF'}", flush=True)
print(f"  Depth curriculum:    {'ON' if DEPTH_ON else 'OFF'}", flush=True)
print(f"  Output: {args.out}", flush=True)

# ── Load training data ──
from src.compiler.prompt_format import SYSTEM_PROMPT
from src.compiler.capability_format import build_capability_prompt
from src.compiler.train.dataset import iter_corpus_records

records = list(iter_corpus_records([args.corpus]))
print(f"Training samples: {len(records)}", flush=True)

# ── Compute action counts for depth bucketing ──
POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

def count_actions(plan_json):
    nodes = plan_json.get("program", {}).get("nodes", [])
    return len([n for n in nodes if n.get("op","") not in POLICY_OPS])

# Load raw records for plan_json and capabilities
raw_records = []
with open(args.corpus) as f:
    for line in f:
        raw_records.append(json.loads(line))

for r in raw_records:
    r["_action_count"] = count_actions(r.get("plan_json", {}))

from collections import Counter, defaultdict
depth_dist = Counter()
for r in raw_records:
    c = r["_action_count"]
    if c == 1: depth_dist["1"] += 1
    elif c <= 3: depth_dist["2-3"] += 1
    elif c <= 6: depth_dist["4-6"] += 1
    else: depth_dist["7+"] += 1

print(f"\nRaw depth distribution: {dict(depth_dist)}", flush=True)

# ── Depth-balanced sampling (Task 5) ──
if DEPTH_ON:
    total = len(raw_records)
    n_buckets = len([b for b in depth_dist if depth_dist[b] > 0])
    target_per_bucket = total / n_buckets

    weights = {}
    effective = {}
    for bucket, count in depth_dist.items():
        if count == 0:
            weights[bucket] = 0
            effective[bucket] = 0
            continue
        w = min(target_per_bucket / count, args.max_oversample)
        weights[bucket] = w
        effective[bucket] = int(count * w)

    print(f"\nDepth curriculum weights (max {args.max_oversample}x):", flush=True)
    for bucket in ("1", "2-3", "4-6", "7+"):
        print(f"  {bucket:5s}: raw={depth_dist.get(bucket,0):5d} weight={weights.get(bucket,0):.2f} effective={effective.get(bucket,0):5d}", flush=True)

    # Build weighted training list (with repetition for oversampling)
    balanced_records = []
    for r in raw_records:
        c = r["_action_count"]
        bucket = "1" if c == 1 else "2-3" if c <= 3 else "4-6" if c <= 6 else "7+"
        w = weights.get(bucket, 1.0)
        reps = int(w) + (1 if random.random() < (w - int(w)) else 0)
        balanced_records.extend([r] * reps)

    random.shuffle(balanced_records)
    print(f"  Effective training samples: {len(balanced_records)} (from {len(raw_records)} unique)", flush=True)

    # Re-apply balanced list
    balanced_dist = Counter()
    for r in balanced_records:
        c = r["_action_count"]
        bucket = "1" if c == 1 else "2-3" if c <= 3 else "4-6" if c <= 6 else "7+"
        balanced_dist[bucket] += 1
    print(f"  Balanced distribution: {dict(balanced_dist)}", flush=True)
else:
    balanced_records = raw_records
    print(f"\nNo depth curriculum — using original distribution", flush=True)

# ── Build SFT records ──
sft_records = []
for r in balanced_records:
    instruction = r.get("instruction", "")
    capabilities = r.get("capabilities") or []

    if SCHEMA_ON and capabilities:
        user_text = build_capability_prompt(instruction, capabilities)
    else:
        user_text = instruction

    target = r.get("plan_target", "")
    if not target:
        continue

    sft_records.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": target},
        ],
        "source": r.get("source", ""),
        "id": r.get("id", ""),
        "quality_tier": r.get("quality_tier", ""),
    })

print(f"\nSFT records: {len(sft_records)}", flush=True)
print(f"  Schema-conditioned: {SCHEMA_ON}", flush=True)
print(f"  Depth-balanced: {DEPTH_ON}", flush=True)

# ── Train ──
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading model...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    quantization_config=__import__("transformers").BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16) if True else None,
    device_map="auto")

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                   target_modules=["q_proj","k_proj","v_proj","o_proj",
                                   "gate_proj","up_proj","down_proj"],
                   task_type="CAUSAL_LM")
model = get_peft_model(prepare_model_for_kbit_training(base), lora)
model.print_trainable_parameters()

def to_text(example):
    return {"text": tok.apply_chat_template(example["messages"], tokenize=False)}

ds = Dataset.from_list(sft_records).map(to_text)

os.makedirs(args.out, exist_ok=True)

# Same hyperparameters as E1 (fair comparison)
config = SFTConfig(
    output_dir=args.out,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1.5e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=20,
    eval_strategy="no",
    save_steps=500,
    save_total_limit=2,
    report_to=[],
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    seed=42,
    max_seq_length=2048,
    dataset_text_field="text",
)

trainer = SFTTrainer(model=model, args=config, train_dataset=ds,
                    processing_class=tok)
trainer.train()
trainer.save_model(f"{args.out}/final")
tok.save_pretrained(f"{args.out}/final")

# Save log history
with open(f"{args.out}/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

# Save run config
with open(f"{args.out}/run_config.json", "w") as f:
    json.dump({
        "mode": f"E5C-{args.mode}",
        "schema_conditioning": SCHEMA_ON,
        "depth_curriculum": DEPTH_ON,
        "raw_samples": len(raw_records),
        "effective_samples": len(balanced_records),
        "sft_records": len(sft_records),
        "argv": vars(args),
    }, f, indent=2)

print(f"\nSaved to {args.out}/final", flush=True)
print("DONE", flush=True)
