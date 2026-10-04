"""Phase 6B Stage 3-4: capability-conditioned TaskIR Composer training.

Arms (frozen before training):
  C1 skill-only   : task + selected canonical SKILL NAMES
  C2 selected-IR  : task + selected concrete capability + canonical skill  (PRIMARY)
  C3 available-IR : task + full available CapabilityIR table (selection implicit)
No-tool records (spider/code/rtl) are bare in every arm (shared control mass).

Base Qwen2.5-3B + QLoRA r16 a32 (same family as all prior arms), 1 epoch,
G1+audited-G2 canonical labels only (G3 metadata dropped from prompts).
"""
import json, os, sys, random, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["C1", "C2", "C3"], required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=float, default=1.0)
parser.add_argument("--out", default="")
args = parser.parse_args()
if not args.out:
    args.out = f"runs/phase6b/composer_{args.arm}_s{args.seed}"

random.seed(args.seed)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from src.compiler.prompt_format import SYSTEM_PROMPT

G3_ALLOWED = False  # G3 canonical labels are dropped from conditioning blocks


def cap_block(cap, canon):
    lines = [f"[{cap['capability_id']}] name: {cap['name']}"]
    if cap.get("description"):
        lines.append(f"description: {cap['description'][:200]}")
    if cap.get("parameters", {}).get("properties"):
        p = cap["parameters"]["properties"]
        lines.append("parameters: " + ", ".join(f"{k}:{v}" for k, v in list(p.items())[:6]))
    lines.append(f"canonical_skill: {canon['canonical_skill']}")
    return "\n".join(lines)


def build_user(rec):
    task = rec["instruction"]
    caps = {c["capability_id"]: c for c in rec["available_capabilities"]}
    canon = {c["capability_id"]: c for c in rec["canonical_capabilities"]}
    sel = [cid for cid in rec["selected_capabilities"]
           if G3_ALLOWED or canon.get(cid, {}).get("label_tier") != "G3"]
    if not sel and rec["available_capabilities"]:
        return None  # tool-use record with no usable conditioning -> skip in all arms
    if args.arm == "C1":
        skills = sorted({canon[cid]["canonical_skill"] for cid in sel if cid in canon})
        if not skills:
            return None
        return f"{task}\n\nSelected skills:\n" + "\n".join(f"- {s}" for s in skills)
    if args.arm == "C2":
        blocks = [cap_block(caps[cid], canon[cid]) for cid in sel if cid in caps and cid in canon]
        if not blocks:
            return None
        return f"{task}\n\nSelected capabilities:\n" + "\n\n".join(blocks)
    if args.arm == "C3":
        blocks = []
        for cid, c in caps.items():
            if not G3_ALLOWED and canon.get(cid, {}).get("label_tier") == "G3":
                continue
            if cid in canon:
                blocks.append(cap_block(c, canon[cid]))
        if not blocks:
            return None
        return f"{task}\n\nAvailable capabilities:\n" + "\n\n".join(blocks[:15])
    return task


records = [json.loads(l) for l in open("data/compiler_corpus_v4/train.jsonl",
                                       encoding="utf-8") if l.strip()]
sft = []
for r in records:
    user = build_user(r)
    if user is None:
        continue
    sft.append({"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": r["plan_target"]}]})
random.shuffle(sft)
print(f"arm={args.arm} seed={args.seed} SFT records: {len(sft)}", flush=True)

tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

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
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=args.epochs,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="no",
    bf16=True,
    max_seq_length=2048,
    dataset_text_field="text",
    optim="paged_adamw_8bit",
    seed=args.seed,
    report_to=[],
)
SFTTrainer(model=model, args=cfg, train_dataset=ds).train()
model.save_pretrained(f"{args.out}/final")
tok.save_pretrained(f"{args.out}/final")
print(f"saved {args.out}/final", flush=True)
print("DONE", flush=True)
