"""BFCL 3B v2: multi_turn supplement + seq length fix.

Changes from v1:
- max_seq_length: 1024 -> 2048 (live_simple schemas were likely truncated)
- Adds multi_turn categories (parsed from BFCL multi_turn question format)
- Keeps same architecture and hyperparameters for fair comparison
"""
import json, os, sys, random, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="runs/bfcl_track/fc_3b_v2_s42")
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

# ── Load v1 data ──
samples = [json.loads(l) for l in open("data/bfcl_training/bfcl_train.jsonl",
                                        encoding="utf-8") if l.strip()]
print(f"v1 data: {len(samples)}")

# ── Parse multi_turn data ──
import glob, re
BFCL_DIR = "data/external_benchmarks/bfcl_v4"
gt_index = {}
for pa in glob.glob(os.path.join(BFCL_DIR, "possible_answer", "*.json")):
    for line in open(pa, encoding="utf-8"):
        if not line.strip(): continue
        g = json.loads(line)
        gt_index[g["id"]] = g["ground_truth"]

mt_samples = []
for cat in ["multi_turn_base", "multi_turn_long_context",
            "multi_turn_miss_func", "multi_turn_miss_param"]:
    path = os.path.join(BFCL_DIR, f"BFCL_v4_{cat}.json")
    if not os.path.exists(path): continue
    for line in open(path, encoding="utf-8"):
        if not line.strip(): continue
        rec = json.loads(line)
        sid = rec.get("id", "")
        q = rec.get("question", [])
        # multi_turn: question is list of list of {role, content}
        turns = []
        if isinstance(q, list):
            for turn in q:
                if isinstance(turn, list):
                    for msg in turn:
                        if isinstance(msg, dict):
                            turns.append(msg)
                elif isinstance(turn, dict):
                    turns.append(turn)

        # Extract first user instruction
        instruction = ""
        for msg in turns:
            if msg.get("role") == "user":
                instruction = msg.get("content", "")
                break

        # Extract function schemas from the turns
        schemas = []
        for msg in turns:
            if msg.get("role") == "system" and "function" in str(msg.get("content", "")):
                # Parse function definitions from system message
                content = msg.get("content", "")
                for fm in re.finditer(r'"name":\s*"(\w+)"', content):
                    schemas.append({"name": fm.group(1), "description": "",
                                    "parameters": {}})
                break

        if not instruction or not schemas:
            continue

        gt = gt_index.get(sid)
        if gt is None: continue

        calls = []
        for item in gt:
            if isinstance(item, dict):
                for fn_name, params in item.items():
                    if isinstance(params, dict):
                        flat = {k: ", ".join(str(x) for x in v) if isinstance(v, list) else v
                                for k, v in params.items()}
                        param_str = ", ".join(f"{k}={repr(v)}" for k, v in flat.items())
                        calls.append(f"{fn_name}({param_str})")
        if not calls: continue
        target = calls[0] if len(calls) == 1 else "\n".join(calls)

        schema_prompt = "\n".join(
            f"[{i}] {s['name']}() - {s.get('description','')[:100]}"
            for i, s in enumerate(schemas[:15], 1))

        mt_samples.append({
            "id": sid, "category": cat, "instruction": instruction,
            "schemas": schemas, "schema_prompt": schema_prompt, "target": target,
        })

print(f"multi_turn parsed: {len(mt_samples)}")
samples.extend(mt_samples)

# xLAM
xlam = [json.loads(l) for l in open("data/bfcl_training/xlam_train.jsonl",
                                     encoding="utf-8") if l.strip()]
random.shuffle(xlam)
xlam = xlam[:30000]
samples.extend(xlam)
random.shuffle(samples)
print(f"v2 total: {len(samples)}")

# Build SFT
sft = []
for s in samples:
    user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
    sft.append({"messages": [
        {"role": "system", "content": SYSTEM_FC},
        {"role": "user", "content": user},
        {"role": "assistant", "content": s["target"]},
    ]})
print(f"SFT records: {len(sft)}")

# Train
tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

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

ds = Dataset.from_list(sft).map(
    lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})

cfg = SFTConfig(
    output_dir=args.out,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2.0,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="no",
    bf16=True,
    max_seq_length=2048,  # ← FIX: was 1024 in v1
    dataset_text_field="text",
    optim="paged_adamw_8bit",
    seed=args.seed,
    report_to=[],
)
SFTTrainer(model=model, args=cfg, train_dataset=ds).train()
model.save_pretrained(f"{args.out}/final")
tok.save_pretrained(f"{args.out}/final")
print(f"saved {args.out}/final")
print("BFCL-V2-TRAIN-DONE")
