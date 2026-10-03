"""Rescore tau3 semantic metrics for all 4 models (CPU only, no inference).

Fixes the E5C-SD crash: parse failures raise raw TypeError (not
TaskIRSyntaxError) when ast.literal_eval hits malformed set literals —
catch Exception broadly. Reads the already-saved preds files.
"""
import json, sys, os
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")

from src.eval.tau3_skill_oracle import lower_reference_actions, semantic_match
from src.ir.parser import parse_text
from src.validator.validator import validate
from collections import Counter

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

FILES = {
    "E1-A":  "runs/phase5b1/tau3_preds_2048.jsonl",
    "E5C-S":  "runs/phase5c/e5c_s_tau3_preds.jsonl",
    "E5C-D":  "runs/phase5c/e5c_d_tau3_preds.jsonl",
    "E5C-SD": "runs/phase5c/e5c_sd_tau3_preds.jsonl",
}


def score_preds(path, label):
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
    res = {
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
    print(f"[{label}] valid={res['valid_pct']}% recall={res['semantic_recall_pct']}% "
          f"({res['matched_actions']}/{res['ref_actions']}) pred/T={res['pred_per_task']} "
          f"prec={res['semantic_precision_pct']}% seq={res['sem_seq_exact_pct']}%", flush=True)
    return res


all_results = {}
for label, path in FILES.items():
    if os.path.exists(path):
        all_results[label] = score_preds(path, label)
    else:
        print(f"[{label}] MISSING {path}", flush=True)

with open("results/phase5c/tau3_semantic.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to results/phase5c/tau3_semantic.json", flush=True)
print("DONE", flush=True)
