"""Phase 5C follow-up: τ³ semantic skill evaluation for the three new models.

The main eval only measured parse/valid/action-count. This script re-runs τ³
inference SAVING predictions, then computes semantic-skill recall (the metric
that answers: do the longer plans actually cover reference actions?).

Runs after eval_5c_schema.py finishes (waits for GPU).
"""
import json, sys, os, time
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")

# Wait for the schema eval to finish
while True:
    rc = os.popen("pgrep -f eval_5c_schema.py").read().strip()
    if not rc:
        break
    print("waiting for eval_5c_schema.py ...", flush=True)
    time.sleep(60)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.eval.tau3_skill_oracle import lower_reference_actions, semantic_match
from src.ir.parser import parse_text, TaskIRSyntaxError
from src.validator.validator import validate
from collections import Counter, defaultdict

MODELS = {
    "E5C-S":  "runs/phase5c/e5c_s/final",
    "E5C-D":  "runs/phase5c/e5c_d/final",
    "E5C-SD": "runs/phase5c/e5c_sd/final",
}
BATCH = 16
POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = []  # instruction + refs
with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            base.append(r)
print(f"τ³ tasks: {len(base)}", flush=True)

# Pre-lower reference actions once
for r in base:
    r["_ref_lowered"] = lower_reference_actions(
        [{"name": n} for n in r.get("ref_action_names", [])])


def run_inference(model, prompts):
    inputs = tok(prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
    return [tok.decode(outputs[j][inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True) for j in range(len(prompts))]


model = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto")

all_results = {}
for name, adapter in MODELS.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    m = PeftModel.from_pretrained(model, adapter)
    m.eval()

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": r["instruction"]}],
        tokenize=False, add_generation_prompt=True) for r in base]

    completions = []
    for bs in range(0, len(prompts), BATCH):
        if bs % 512 == 0:
            print(f"    {bs}/{len(prompts)}", flush=True)
        completions.extend(run_inference(m, prompts[bs:bs+BATCH]))

    # Save preds
    os.makedirs("runs/phase5c", exist_ok=True)
    out_path = f"runs/phase5c/{name.lower().replace('-', '_')}_tau3_preds.jsonl"
    with open(out_path, "w") as f:
        for r, comp in zip(base, completions):
            rec = {k: v for k, v in r.items() if not k.startswith("_")}
            rec["taskir_text"] = extract_taskir_text(comp)
            rec["raw_output"] = comp[:2000]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"    saved {out_path}", flush=True)

    # Semantic metrics over saved records
    g = Counter()
    g["n"] = len(base)
    with open(out_path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    for rec in recs:
        g["ref_count_sum"] += rec["n_ref"]
        try:
            module = parse_text(rec["taskir_text"])
        except TaskIRSyntaxError:
            module = None
        if module is None or not validate(module).valid:
            g["ref_sem_missing"] += rec["n_ref"]
            continue
        pred_nodes = [n for n in module.program.nodes if n.op not in POLICY_OPS]
        g["pred_count_sum"] += len(pred_nodes)
        ref_lowered = lower_reference_actions(
            [{"name": n} for n in rec.get("ref_action_names", [])])
        used = set()
        for ri, rlow in enumerate(ref_lowered):
            g["ref_sem_total"] += 1
            for pi, pn in enumerate(pred_nodes):
                if pi in used:
                    continue
                if semantic_match(pn.op, pn.params, rlow):
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
    res = {
        "n": g["n"],
        "ref_actions": g["ref_count_sum"],
        "pred_actions": g["pred_count_sum"],
        "semantic_recall_pct": round(100 * g["ref_sem_matched"] / max(1, g["ref_sem_total"]), 2),
        "matched_actions": g["ref_sem_matched"],
        "pred_per_task": round(g["pred_count_sum"] / n, 2),
        "ref_per_task": round(g["ref_count_sum"] / n, 2),
        "sem_seq_exact_pct": round(100 * g["sem_seq_exact"] / max(1, g["sem_seq_total"]), 2),
        "multi_complete_pct": round(100 * g["multi_complete"] / max(1, g["multi_tasks"]), 2),
    }
    all_results[name] = res
    print(f"    semantic_recall={res['semantic_recall_pct']}% "
          f"matched={res['matched_actions']}/{res['ref_actions']} "
          f"pred/T={res['pred_per_task']} ref/T={res['ref_per_task']} "
          f"seq_exact={res['sem_seq_exact_pct']}% multi_complete={res['multi_complete_pct']}%", flush=True)

    del m
    torch.cuda.empty_cache()

# ── E1-A baseline: score existing preds (no inference needed) ──
def score_preds(path, label):
    g = Counter()
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    g["n"] = len(recs)
    for rec in recs:
        g["ref_count_sum"] += rec["n_ref"]
        try:
            module = parse_text(rec["taskir_text"])
        except TaskIRSyntaxError:
            module = None
        if module is None or not validate(module).valid:
            g["ref_sem_missing"] += rec["n_ref"]
            continue
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
                if semantic_match(pn.op, pn.params, rlow):
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
    res = {
        "n": g["n"],
        "ref_actions": g["ref_count_sum"],
        "pred_actions": g["pred_count_sum"],
        "semantic_recall_pct": round(100 * g["ref_sem_matched"] / max(1, g["ref_sem_total"]), 2),
        "matched_actions": g["ref_sem_matched"],
        "pred_per_task": round(g["pred_count_sum"] / n, 2),
        "ref_per_task": round(g["ref_count_sum"] / n, 2),
        "sem_seq_exact_pct": round(100 * g["sem_seq_exact"] / max(1, g["sem_seq_total"]), 2),
        "multi_complete_pct": round(100 * g["multi_complete"] / max(1, g["multi_tasks"]), 2),
    }
    print(f"\n[{label}] semantic_recall={res['semantic_recall_pct']}% "
          f"matched={res['matched_actions']}/{res['ref_actions']} "
          f"pred/T={res['pred_per_task']} seq_exact={res['sem_seq_exact_pct']}%", flush=True)
    return res


if os.path.exists("runs/phase5b1/tau3_preds_2048.jsonl"):
    all_results["E1-A"] = score_preds("runs/phase5b1/tau3_preds_2048.jsonl", "E1-A")

with open("results/phase5c/tau3_semantic.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to results/phase5c/tau3_semantic.json", flush=True)
print("DONE", flush=True)
