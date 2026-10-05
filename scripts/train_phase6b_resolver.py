"""Phase 6B-2: Resolver training — R1 generative / R2 pairwise scorer."""
import json, os, sys, random, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["R1", "R2"], required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="")
args = parser.parse_args()
if not args.out:
    args.out = f"runs/phase6b/resolver_{args.mode}_s{args.seed}"
random.seed(args.seed)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

SYSTEM_R = ("You are a capability resolver for a task compiler. Given a task and "
            "capability descriptions, decide which capabilities the task requires. "
            "Answer with the label ONLY.")

sft = []
if args.mode == "R2":
    for line in open("data/phase6b_resolver/r2_pairs_train.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        p = json.loads(line)
        if p["label"] == "NONE":
            # task + full foreign table -> NONE
            user = (f"Task:\n{p['task']}\n\nAvailable capabilities:\n" +
                    "\n\n".join(p["available_blocks"]) +
                    "\n\nWhich capabilities does the task require? "
                    "Answer 'NONE' if none apply.")
            target = "NONE"
        else:
            user = (f"Task:\n{p['task']}\n\nCapability:\n{p['block']}\n\n"
                    "Is this capability required by the task? "
                    "Answer 'relevant' or 'irrelevant'.")
            target = p["label"]
        sft.append({"messages": [
            {"role": "system", "content": SYSTEM_R},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target}]})
else:
    for line in open("data/phase6b_resolver/r1_train.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        user = (f"Task:\n{r['task']}\n\nAvailable capabilities:\n" +
                "\n\n".join(r["available_blocks"]) +
                "\n\nWhich capabilities does the task require? "
                "Answer as a comma-separated id list (e.g. c3,c7) or NONE.")
        target = ",".join(r["gold_ids"])
        sft.append({"messages": [
            {"role": "system", "content": SYSTEM_R},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target}]})
random.shuffle(sft)
print(f"mode={args.mode} seed={args.seed} SFT records: {len(sft)}", flush=True)

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
batch = 16 if args.mode == "R2" else 4
accum = 1 if args.mode == "R2" else 4
cfg = SFTConfig(
    output_dir=args.out,
    per_device_train_batch_size=batch,
    gradient_accumulation_steps=accum,
    num_train_epochs=1.0,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="no",
    bf16=True,
    max_seq_length=1024 if args.mode == "R2" else 2048,
    dataset_text_field="text",
    optim="paged_adamw_8bit",
    seed=args.seed,
    report_to=[],
)
SFTTrainer(model=model, args=cfg, train_dataset=ds).train()
model.save_pretrained(f"{args.out}/final")
tok.save_pretrained(f"{args.out}/final")
print(f"saved {args.out}/final", flush=True)
print("RESOLVER-TRAIN-DONE", flush=True)
