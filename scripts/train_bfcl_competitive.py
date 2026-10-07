"""Train a function-calling model for BFCL competition (Track A).

Approach: direct instruction-to-function-call training (no TaskIR).
Training data: BFCL V4 (3,621) + xLAM 60k = 63,621 samples.
Base: Qwen2.5-3B-Instruct + QLoRA.

This is the "leaderboard track" — separate from the cross-domain research model.
"""
import json, os, sys, random, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=float, default=2.0)
parser.add_argument("--model", default="weights/Qwen2.5-3B-Instruct")
parser.add_argument("--out", default="runs/bfcl_track/fc_3b_s42")
args = parser.parse_args()
random.seed(args.seed)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

SYSTEM_FC = (
    "You are a function-calling assistant. Given a task and available function "
    "schemas, output the correct function call(s). If no function is relevant, "
    "say 'No function should be called.' Otherwise output function call(s) in "
    "the format: function_name(param1=value1, param2=value2)"
)

# Load data
samples = []
for path in ["data/bfcl_training/bfcl_train.jsonl", "data/bfcl_training/xlam_train.jsonl"]:
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        samples.append(r)

# Cap xLAM at 30k to prevent overwhelming BFCL patterns
bfcl = [s for s in samples if s["category"] != "xlam"]
xlam = [s for s in samples if s["category"] == "xlam"]
random.shuffle(xlam)
xlam = xlam[:30000]
combined = bfcl + xlam
random.shuffle(combined)
print(f"BFCL: {len(bfcl)}, xLAM (capped): {len(xlam)}, total: {len(combined)}")

# Build SFT records
sft = []
for s in combined:
    user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
    sft.append({"messages": [
        {"role": "system", "content": SYSTEM_FC},
        {"role": "user", "content": user},
        {"role": "assistant", "content": s["target"]},
    ]})
print(f"SFT records: {len(sft)}")

# Load model
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model_short = "3b" if "3B" in args.model else "7b"
base = AutoModelForCausalLM.from_pretrained(
    args.model, trust_remote_code=True,
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

ds = Dataset.from_list(sft).map(
    lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})

# 3B can handle batch 4 with seq 1024 (short targets)
batch = 4 if model_short == "3b" else 2
accum = 4 if model_short == "3b" else 8
cfg = SFTConfig(
    output_dir=args.out,
    per_device_train_batch_size=batch,
    gradient_accumulation_steps=accum,
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
SFTTrainer(model=model, args=cfg, train_dataset=ds).train()
model.save_pretrained(f"{args.out}/final")
tok.save_pretrained(f"{args.out}/final")
print(f"saved {args.out}/final")
print("BFCL-TRAIN-DONE")
