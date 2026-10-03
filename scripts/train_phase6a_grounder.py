"""Phase 6A Task 12: train the skill grounder (Qwen2.5-3B + QLoRA).

Configs:
  g1    : G1-tier samples only (primary, seeds 42/43/44)
  g1g2  : G1+G2 (secondary ablation, seed 42)

Training prompt mix: 40% Probe A (schema-only) + 40% Probe B (task+schema)
+ 20% Probe C (selection among 5 candidates, output 'capability_id=<n>;
skill=<LABEL>'). Output for A/B is the bare label.
"""
import json, os, sys, random, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["g1", "g1g2"], default="g1")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=float, default=1.0)
parser.add_argument("--out", default="")
args = parser.parse_args()

if not args.out:
    args.out = f"runs/phase6/grounder_{args.config}_s{args.seed}"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from src.compiler.skill_grounder import (build_probe_prompt,
                                         pick_hard_negatives)

random.seed(args.seed)

DATA = "data/phase6a_grounding"
train = [json.loads(l) for l in open(f"{DATA}/train.jsonl", encoding="utf-8") if l.strip()]
if args.config == "g1":
    train = [s for s in train if s["tier"] == "G1"]
print(f"config={args.config} seed={args.seed} train samples: {len(train)}", flush=True)

# build SFT records
records = []
for i, s in enumerate(train):
    r = random.random()
    if r < 0.40:
        probe, cands = "A", None
        target = s["target_skill"]
    elif r < 0.80:
        probe, cands = "B", None
        target = s["target_skill"]
    else:
        probe = "C"
        cands = pick_hard_negatives(s, train, k=4, rng=random.Random(i))
        # positive inserted at deterministic random position
        pos_idx = random.Random(i * 7).randrange(5)
        cands = cands[:pos_idx] + [s] + cands[pos_idx:]
        target = f"capability_id={pos_idx + 1}; skill={s['target_skill']}"
    p = build_probe_prompt(s, probe, candidates=cands)
    records.append({
        "messages": [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
            {"role": "assistant", "content": target},
        ]
    })
random.shuffle(records)
print(f"SFT records: {len(records)}", flush=True)

tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading model...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto")
base = prepare_model_for_kbit_training(base)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                  task_type="CAUSAL_LM")
model = get_peft_model(base, lora)
model.print_trainable_parameters()


def to_text(example):
    return {"text": tok.apply_chat_template(example["messages"], tokenize=False)}


ds = Dataset.from_list(records).map(to_text)

cfg = SFTConfig(
    output_dir=args.out,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=args.epochs,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="no",
    bf16=True,
    max_seq_length=1024,
    dataset_text_field="text",
    optim="paged_adamw_8bit",
    seed=args.seed,
    report_to=[],
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
trainer.train()

model.save_pretrained(f"{args.out}/final")
tok.save_pretrained(f"{args.out}/final")
print(f"saved {args.out}/final", flush=True)
print("DONE", flush=True)
