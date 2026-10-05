"""Phase 6B-1 Task 8: C0T — post-hoc matched-training control.

Bare-prompt retrain on EXACTLY the same filtered training records as the C2
arm (tool-use records with usable G1/G2 selected capabilities). Same QLoRA
config, 1 epoch, seed 42. The ONLY difference vs C2 is the absence of the
capability conditioning block -> isolates the conditioning effect from the
training-subset distribution effect.

POST-HOC CONTROL — explicitly not a preregistered arm.
"""
import json, os, sys, random

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
random.seed(42)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from src.compiler.prompt_format import SYSTEM_PROMPT

OUT = "runs/phase6b/composer_C0T_s42"

records = [json.loads(l) for l in open("data/compiler_corpus_v4/train.jsonl",
                                       encoding="utf-8") if l.strip()]
sft = []
for r in records:
    canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
    sel = [cid for cid in r["selected_capabilities"]
           if canon.get(cid, {}).get("label_tier") != "G3"]
    # EXACT C2 retention rule: tool-use record with >=1 usable selected capability
    if not sel and r["available_capabilities"]:
        continue
    if not sel:
        continue  # C2 also drops these (build_user returns None when blocks empty)
    sft.append({"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": r["instruction"]},   # BARE — no conditioning
        {"role": "assistant", "content": r["plan_target"]}]})
random.shuffle(sft)
print(f"C0T (post-hoc control) SFT records: {len(sft)}", flush=True)

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
    output_dir=OUT,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1.0,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_strategy="no",
    bf16=True,
    max_seq_length=2048,
    dataset_text_field="text",
    optim="paged_adamw_8bit",
    seed=42,
    report_to=[],
)
SFTTrainer(model=model, args=cfg, train_dataset=ds).train()
model.save_pretrained(f"{OUT}/final")
tok.save_pretrained(f"{OUT}/final")
print(f"saved {OUT}/final", flush=True)
print("C0T-DONE", flush=True)
