"""Task 3: E1-A tau3 protocol-normalized rerun (freeze-fix).

E1-A's previous tau3 preds (tau3_preds_2048.jsonl) were generated with a
longer budget than S/D/SD. Rerun E1-A with the EXACT protocol of
eval_5c_tau3_semantic.py: SYSTEM_PROMPT + bare instruction, input
max_length=2048, max_new_tokens=512, greedy, left padding, same POLICY_OPS
filtering, same semantic oracle. Rescore all metrics and rewrite
results/phase5c/tau3_semantic.json so all four cells share one protocol.
"""
import json, sys, os
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.eval.tau3_skill_oracle import lower_reference_actions, semantic_match
from src.ir.parser import parse_text
from src.validator.validator import validate
from collections import Counter

BATCH = 16
POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = []
with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
    for line in f:
        if line.strip():
            base.append(json.loads(line))
print(f"tau3 tasks: {len(base)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "runs/phase5b1/e1_3b_qlora/final")
model.eval()

prompts = [tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": r["instruction"]}],
    tokenize=False, add_generation_prompt=True) for r in base]

completions = []
for bs in range(0, len(prompts), BATCH):
    if bs % 512 == 0:
        print(f"    {bs}/{len(prompts)}", flush=True)
    batch = prompts[bs:bs+BATCH]
    inputs = tok(batch, return_tensors="pt", padding=True,
                 truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
    completions.extend(tok.decode(outputs[j][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                       for j in range(len(batch)))

out_path = "runs/phase5c/e1_a_tau3_preds_norm.jsonl"
with open(out_path, "w") as f:
    for r, comp in zip(base, completions):
        rec = {k: v for k, v in r.items() if k != "taskir_text"}
        rec["taskir_text"] = extract_taskir_text(comp)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"saved {out_path}", flush=True)

del model
torch.cuda.empty_cache()

# ── Rescore E1-A normalized + rewrite tau3_semantic.json with uniform protocol ──
def score(path):
    g = Counter()
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    g["n"] = len(recs)
    for rec in recs:
        g["ref_count_sum"] += rec["n_ref"]
        try:
            module = parse_text(rec["taskir_text"])
        except Exception:
            module = None
        if module is None:
            g["ref_sem_missing"] += rec["n_ref"]
            continue
        try:
            ok = validate(module).valid
        except Exception:
            ok = False
        if not ok:
            g["ref_sem_missing"] += rec["n_ref"]
            continue
        g["valid"] += 1
        pred_nodes = [n for n in module.program.nodes if n.op not in POLICY_OPS]
        g["pred_count_sum"] += len(pred_nodes)
        ref_lowered = lower_reference_actions(
            [{"name": n} for n in rec.get("ref_action_names", [])])
        used = set()
        for rlow in ref_lowered:
            g["ref_sem_total"] += 1
            for pi, pn in enumerate(pred_nodes):
                if pi in used:
                    continue
                try:
                    hit = semantic_match(pn.op, pn.params, rlow)
                except Exception:
                    hit = False
                if hit:
                    used.add(pi)
                    g["ref_sem_matched"] += 1
                    break
            else:
                g["ref_sem_missing"] += 1
        if ref_lowered:
            g["sem_seq_total"] += 1
            if [n.op for n in pred_nodes] == [rl["skill"] for rl in ref_lowered]:
                g["sem_seq_exact"] += 1
        if rec["n_ref"] > 1:
            g["multi_tasks"] += 1
            if len(used) == len(ref_lowered):
                g["multi_complete"] += 1
    n = g["n"] or 1
    return {
        "n": g["n"],
        "valid_pct": round(100 * g["valid"] / n, 2),
        "ref_actions": g["ref_count_sum"],
        "pred_actions": g["pred_count_sum"],
        "semantic_recall_pct": round(100 * g["ref_sem_matched"] / max(1, g["ref_sem_total"]), 2),
        "matched_actions": g["ref_sem_matched"],
        "pred_per_task": round(g["pred_count_sum"] / n, 2),
        "ref_per_task": round(g["ref_count_sum"] / n, 2),
        "sem_seq_exact_pct": round(100 * g["sem_seq_exact"] / max(1, g["sem_seq_total"]), 2),
        "multi_complete_pct": round(100 * g["multi_complete"] / max(1, g["multi_tasks"]), 2),
        "semantic_precision_pct": round(100 * g["ref_sem_matched"] / max(1, g["pred_count_sum"]), 2),
    }

e1_norm = score(out_path)
print(f"\nE1-A (normalized): valid={e1_norm['valid_pct']}% "
      f"recall={e1_norm['semantic_recall_pct']}% ({e1_norm['matched_actions']}/{e1_norm['ref_actions']}) "
      f"pred/T={e1_norm['pred_per_task']} prec={e1_norm['semantic_precision_pct']}% "
      f"seq={e1_norm['sem_seq_exact_pct']}% multi={e1_norm['multi_complete_pct']}%", flush=True)

# Rewrite tau3_semantic.json: E1-A replaced by normalized; others unchanged
with open("results/phase5c/tau3_semantic.json") as f:
    results = json.load(f)
results["E1-A"] = e1_norm
results["_protocol_note"] = ("All four models now share one inference protocol: "
                             "SYSTEM_PROMPT + bare instruction, input max 2048, "
                             "max_new_tokens 512, greedy. E1-A preds: "
                             "runs/phase5c/e1_a_tau3_preds_norm.jsonl")
with open("results/phase5c/tau3_semantic.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nRewrote results/phase5c/tau3_semantic.json", flush=True)
print("DONE", flush=True)
