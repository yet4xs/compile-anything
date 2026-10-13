"""Phase 8B-1B Task 7: matched-budget M0/M1 QLoRA training.

M0: NL + capabilities -> TaskIR (target = DETERMINISTIC lowering of the
    record's skeleton — semantic supervision identical to M1)
M1: NL + capabilities -> Semantic Skeleton JSON -> deterministic lowering
Budget alignment: both run the SAME number of optimizer steps (set by
M0's 2-epoch plan); identical task ids, CapabilityIR, LoRA config.
Smoke mode first (--smoke): 120 samples, 20 steps, parse checks.
"""
import json, os, sys, time, argparse, math, random

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from torch.utils.data import Dataset

from src.compiler.plan_skeleton import skeleton_from_dict
from src.compiler.skeleton_lowering import lower_skeleton

DATA = "data/semantic_skeleton_v3/train_sft.jsonl"

SYS_M0 = ("You are a task compiler. Given a natural-language task and "
          "the available capabilities, emit a complete TaskIR program. "
          "Output ONLY the TaskIR text.")
SYS_M1 = ("You are a semantic task planner. Given a natural-language "
          "task and the available capabilities, emit ONLY a JSON object "
          '{"nodes": [{"id","capability_id","skill","depends_on",'
          '"order_on","arguments"}]} planning every required tool call '
          "in order.")


def caps_block(rec):
    lines = []
    for name in rec["available_capabilities"]:
        meta = rec.get("capability_registry", {}).get(name, {})
        req = ",".join(meta.get("required") or []) or "-"
        desc = (meta.get("description") or "")[:110]
        lines.append(f"- {name} | required: {req} | {desc}")
    return "\n".join(lines)


def build_examples(rows):
    m0, m1 = [], []
    skipped = 0
    for r in rows:
        sk = skeleton_from_dict(r["target_skeleton"])
        reg = {t: ({"params": [{"name": x, "optional": False}
                               for x in m["required"]]}
                   if m.get("schema") == "props" else {"params": None})
               for t, m in r["capability_registry"].items()}
        low = lower_skeleton(sk, None, reg, strict=True)
        if low.get("taskir") is None:
            skipped += 1
            continue
        user = (f"Task:\n{r['instruction']}\n\nAvailable capabilities:\n"
                f"{caps_block(r)}")
        m0.append({"task_id": r["task_id"], "user": user,
                   "target": low["taskir"]})
        m1.append({"task_id": r["task_id"], "user": user,
                   "target": json.dumps(r["target_skeleton"],
                                        ensure_ascii=False)})
    return m0, m1, skipped


class SFTData(Dataset):
    def __init__(self, rows, tok, sys_prompt, max_len=2048):
        self.items = []
        self.tok = tok
        self.sys = sys_prompt
        self.max_len = max_len
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": self.sys},
             {"role": "user", "content": r["user"]}],
            tokenize=False, add_generation_prompt=True)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(r["target"], add_special_tokens=False)["input_ids"] \
            + [self.tok.eos_token_id]
        ids = (p_ids + t_ids)[-self.max_len:]
        labels = ([-100] * len(p_ids) + t_ids)[-self.max_len:]
        return {"input_ids": ids, "labels": labels,
                "attn": [1] * len(ids)}


def collate(batch, pad_id):
    mx = max(len(b["input_ids"]) for b in batch)
    ii, ll, aa = [], [], []
    for b in batch:
        n = mx - len(b["input_ids"])
        ii.append(b["input_ids"] + [pad_id] * n)
        ll.append(b["labels"] + [-100] * n)
        aa.append(b["attn"] + [0] * n)
    return {"input_ids": torch.tensor(ii),
            "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(aa)}


def count_tokens(rows, tok):
    return sum(len(tok(r["target"], add_special_tokens=False)["input_ids"])
               for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m0", choices=["m0", "m1"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")
            if l.strip()]
    m0_rows, m1_rows, skipped = build_examples(rows)
    print(f"examples: m0={len(m0_rows)} m1={len(m1_rows)} "
          f"(skipped-lowering={skipped})")
    assert [r["task_id"] for r in m0_rows] == \
        [r["task_id"] for r in m1_rows]

    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.smoke:
        m0_rows, m1_rows = m0_rows[:120], m1_rows[:120]

    rows_sel = m0_rows if args.model == "m0" else m1_rows
    sys_sel = SYS_M0 if args.model == "m0" else SYS_M1

    t_tokens = count_tokens(rows_sel, tok)
    print(f"{args.model}: {len(rows_sel)} samples, "
          f"{t_tokens} target tokens")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        quantization_config=bnb, device_map="auto")
    base.config.use_cache = False
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()

    eff_batch = 16
    per_dev = 2
    accum = eff_batch // per_dev
    steps_per_epoch = math.ceil(len(rows_sel) / eff_batch)
    # matched budget: BOTH models run the SAME max_steps, defined by the
    # M0 2-epoch plan (see manifest)
    if args.model == "m0":
        max_steps = int(steps_per_epoch * args.epochs)
    else:
        max_steps = int(json.load(open(
            "experiments/phase8b/m0_m1_training_manifest.json"))
            ["budget"]["max_steps"]) if not args.smoke else 20
    if args.smoke:
        max_steps = 20
    print(f"max_steps={max_steps} (epochs-worth "
          f"{max_steps / steps_per_epoch:.2f})")

    out_dir = args.out or f"runs/phase8b/{args.model}_s{args.seed}" \
        + ("_smoke" if args.smoke else "")
    targs = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=accum, num_train_epochs=None,
        max_steps=max_steps, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="no", bf16=True,
        optim="paged_adamw_32bit", report_to=[], seed=args.seed,
        dataloader_shuffle=True, max_grad_norm=1.0)
    ds = SFTData(rows_sel, tok, sys_sel)
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(f"{out_dir}/final")
    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump({"model": args.model, "seed": args.seed,
                   "n_samples": len(rows_sel),
                   "target_tokens": t_tokens,
                   "max_steps": max_steps,
                   "steps_per_epoch": steps_per_epoch,
                   "effective_epochs": round(max_steps / steps_per_epoch, 3),
                   "loss_history": trainer.state.log_history,
                   "wall_time_s": round(dt, 1)}, f, indent=2)
    print(f"saved {out_dir}/final | {dt/60:.1f} min | TRAIN-DONE")


if __name__ == "__main__":
    main()
