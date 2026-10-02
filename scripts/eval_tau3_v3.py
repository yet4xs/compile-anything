"""τ³ v3 evaluation: aligned with Skill ISA abstraction layers.

L2-A: Semantic Skill Planning (pred semantic skills vs oracle-lowered ref skills)
L2-B: Action Intent (effect class, action count)
L2-C: Concrete Binding (pred semantic → binder → concrete tool name)
L4: Effect classification (via semantic correspondence)
"""
import json, sys, os, time, hashlib
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")

from src.eval.tau3_skill_oracle import (
    tool_to_semantic_skill, lower_reference_actions, semantic_match,
    classify_effect, READ_PAT, IRREV_PAT, REV_PAT
)
from src.ir.parser import parse_text, TaskIRSyntaxError
from src.validator.validator import validate
from collections import Counter, defaultdict

PRED_FILE = "runs/phase5b1/tau3_preds.jsonl"
OUT_DIR = "results/tau3_offline"
POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

# SKILL_EFFECTS for predicted nodes
SKILL_EFFECTS = {
    "SEARCH": "read_only", "FETCH": "read_only", "QUERY_DB": "read_only",
    "LOAD": "read_only", "FILTER": "read_only", "TRANSFORM": "read_only",
    "EXTRACT": "read_only", "DEDUP": "read_only", "SORT": "read_only",
    "JOIN": "read_only", "MERGE": "read_only", "ARGMIN": "read_only",
    "ARGMAX": "read_only", "MIN": "read_only", "MAX": "read_only",
    "SUM": "read_only", "AVG": "read_only", "COUNT": "read_only",
    "CALCULATE": "read_only", "COMPARE": "read_only", "CONVERT": "read_only",
    "GENERATE": "read_only", "SUMMARIZE": "read_only", "TRANSLATE": "read_only",
    "CLASSIFY": "read_only", "EXTRACT_ENTITIES": "read_only",
    "CODEGEN": "read_only", "PLAN": "read_only", "VERIFY": "read_only",
    "SELECT": "read_only", "SEND": "irreversible_world",
    "SAVE": "reversible_state",
    "EXEC_ACTION": "unknown",  # depends on params
}

def pred_effect(skill, params):
    if skill in SKILL_EFFECTS:
        ec = SKILL_EFFECTS[skill]
        if ec == "unknown" and params:
            return classify_effect(str(params.get("action", "")))[0]
        return ec
    return "unknown"

# ── Load data ──
TAU3_DIR = "data/external_benchmarks/tau3_bench/domains"
DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")

samples = []
for dom in DOMAINS:
    dpath = os.path.join(TAU3_DIR, dom)
    if not os.path.isdir(dpath): continue
    with open(os.path.join(dpath, "tasks.json")) as f:
        tasks = json.load(f)
    for t in tasks:
        ec = t.get("evaluation_criteria") or {}
        ref_actions = ec.get("actions") or []
        ref_lowered = lower_reference_actions(ref_actions)
        samples.append({
            "case_id": f"{dom}-{t.get('id','')}",
            "domain": dom,
            "ref_lowered": ref_lowered,
            "n_ref": len(ref_lowered),
            "ref_effects": [r["effect_class"] for r in ref_lowered],
        })

print(f"τ³ samples: {len(samples)}", flush=True)
print(f"Reference actions: {sum(s['n_ref'] for s in samples)}", flush=True)

# ── Load predictions ──
preds = []
with open(PRED_FILE) as f:
    for line in f:
        if line.strip():
            preds.append(json.loads(line))
print(f"Predictions loaded: {len(preds)}", flush=True)

# ── Token count analysis (Task 8: banking root cause) ──
print("\n=== Token Count Analysis (Banking root cause) ===", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
dom_tokens = defaultdict(list)
for p in preds:
    tokens = len(tok.encode(p["instruction"]))
    dom_tokens[p["domain"]].append(tokens)

print(f"{'Domain':<20s} {'p50':>6s} {'p90':>6s} {'>1024':>7s} {'%>1024':>7s}", flush=True)
for dom in DOMAINS:
    toks = sorted(dom_tokens.get(dom, []))
    if not toks: continue
    p50 = toks[len(toks)//2]
    p90 = toks[int(len(toks)*0.9)]
    over = sum(1 for t in toks if t > 1024)
    pct = 100*over/len(toks)
    print(f"{dom:<20s} {p50:>6d} {p90:>6d} {over:>7d} {pct:>6.1f}%", flush=True)

# ── L2-A: Semantic Skill Evaluation ──
print("\n=== L2-A: Semantic Skill Planning ===", flush=True)

g = {
    "ref_sem_total": 0, "ref_sem_matched": 0, "ref_sem_missing": 0,
    "pred_total": 0, "pred_extra": 0,
    "effect_cond_correct": 0, "effect_cond_total": 0,
    "effect_e2e_correct": 0,
    "sem_seq_exact": 0, "sem_seq_total": 0,
    "ref_count_sum": 0, "pred_count_sum": 0,
    "multi_action_tasks": 0, "multi_complete": 0,
    "retry_nodes": 0, "unsafe_retry": 0,
    "parse_ok": 0, "valid": 0,
}

effect_conf = defaultdict(lambda: Counter())
evidence_dist = Counter()
dom_metrics = defaultdict(lambda: {
    "n": 0, "parse": 0, "valid": 0,
    "ref_sem": 0, "matched": 0,
    "pred_count": 0, "ref_count": 0,
})

for p in preds:
    dom = p["domain"]
    dm = dom_metrics[dom]
    dm["n"] += 1
    g["ref_count_sum"] += p["n_ref"]
    dm["ref_count"] += p["n_ref"]

    # Parse & validate
    parse_ok = True
    module = None
    try:
        module = parse_text(p["taskir_text"])
    except TaskIRSyntaxError:
        parse_ok = False
    if parse_ok:
        g["parse_ok"] += 1
        dm["parse"] += 1
    else:
        g["ref_sem_missing"] += p["n_ref"]
        dm["ref_sem"] += p["n_ref"]
        continue

    valid = validate(module).valid
    if valid:
        g["valid"] += 1
        dm["valid"] += 1
    else:
        g["ref_sem_missing"] += p["n_ref"]
        dm["ref_sem"] += p["n_ref"]
        continue

    # Get predicted semantic action nodes
    pred_nodes = [n for n in module.program.nodes if n.op not in POLICY_OPS]
    g["pred_count_sum"] += len(pred_nodes)
    dm["pred_count"] += len(pred_nodes)
    g["pred_total"] += len(pred_nodes)

    # Get reference lowered semantic skills
    ref_lowered = p.get("ref_lowered") or []
    if not ref_lowered:
        # Recompute if not in prediction
        ref_lowered = lower_reference_actions(
            [{"name": n} for n in p.get("ref_action_names", [])])

    # Semantic matching
    matched_pairs = []
    used_pred = set()

    for ri, rlow in enumerate(ref_lowered):
        g["ref_sem_total"] += 1
        dm["ref_sem"] += 1
        found = False
        for pi, pn in enumerate(pred_nodes):
            if pi in used_pred:
                continue
            if semantic_match(pn.op, pn.params, rlow):
                matched_pairs.append((ri, pi, rlow, pn))
                used_pred.add(pi)
                found = True
                break
        if found:
            g["ref_sem_matched"] += 1
            dm["matched"] += 1
        else:
            g["ref_sem_missing"] += 1
            effect_conf[rlow["effect_class"]]["MISSING"] += 1
            evidence_dist[rlow.get("evidence", "unknown")] += 1

    g["pred_extra"] += len(pred_nodes) - len(used_pred)

    # Effect classification for matched pairs
    for ri, pi, rlow, pn in matched_pairs:
        ref_eff = rlow["effect_class"]
        pred_eff = pred_effect(pn.op, pn.params)
        g["effect_cond_total"] += 1
        effect_conf[ref_eff][pred_eff] += 1
        if ref_eff == pred_eff:
            g["effect_cond_correct"] += 1
            g["effect_e2e_correct"] += 1

    # Sequence exact
    if ref_lowered:
        g["sem_seq_total"] += 1
        pred_sem = [n.op for n in pred_nodes]
        ref_sem = [r["skill"] for r in ref_lowered]
        if pred_sem == ref_sem:
            g["sem_seq_exact"] += 1

    # Multi-action completion
    if p["n_ref"] > 1:
        g["multi_action_tasks"] += 1
        if len(used_pred) >= p["n_ref"] * 0.5:  # at least 50% coverage
            g["multi_complete"] += 1

    # Retry
    for n in module.program.nodes:
        if n.retry:
            g["retry_nodes"] += 1
            eff = pred_effect(n.op, n.params)
            if n.retry.on != "error" and eff == "irreversible_world":
                g["unsafe_retry"] += 1

# ── Results ──
n_tasks = len(preds)
n_ref = g["ref_sem_total"]

print(f"\n{'='*60}", flush=True)
print(f"LAYER 1 — STRUCTURAL", flush=True)
print(f"{'='*60}", flush=True)
print(f"Parse: {g['parse_ok']}/{n_tasks} ({100*g['parse_ok']/n_tasks:.1f}%)", flush=True)
print(f"Valid: {g['valid']}/{n_tasks} ({100*g['valid']/n_tasks:.1f}%)", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"LAYER 2-A — SEMANTIC SKILL PLANNING", flush=True)
print(f"{'='*60}", flush=True)
print(f"Reference semantic actions: {n_ref}", flush=True)
print(f"Matched: {g['ref_sem_matched']} ({100*g['ref_sem_matched']/n_ref:.1f}%)", flush=True)
print(f"Missing: {g['ref_sem_missing']} ({100*g['ref_sem_missing']/n_ref:.1f}%)", flush=True)
print(f"Predicted nodes: {g['pred_total']}", flush=True)
print(f"Extra: {g['pred_extra']}", flush=True)

sem_p = g["ref_sem_matched"] / max(1, g["pred_total"])
sem_r = g["ref_sem_matched"] / max(1, n_ref)
sem_f1 = 2 * sem_p * sem_r / max(1e-9, sem_p + sem_r)
print(f"Semantic Skill Precision: {100*sem_p:.1f}%", flush=True)
print(f"Semantic Skill Recall: {100*sem_r:.1f}%", flush=True)
print(f"Semantic Skill F1: {100*sem_f1:.1f}%", flush=True)
print(f"Sequence Exact: {g['sem_seq_exact']}/{g['sem_seq_total']} ({100*g['sem_seq_exact']/max(1,g['sem_seq_total']):.1f}%)", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"LAYER 2-B — ACTION INTENT / UNDER-PLANNING", flush=True)
print(f"{'='*60}", flush=True)
print(f"Ref actions/task: {g['ref_count_sum']/n_tasks:.2f}", flush=True)
print(f"Pred actions/task: {g['pred_count_sum']/n_tasks:.2f}", flush=True)
print(f"Action count ratio: {g['pred_count_sum']/max(1,g['ref_count_sum']):.2f}", flush=True)
print(f"Multi-action tasks: {g['multi_action_tasks']}", flush=True)
print(f"Multi-action ≥50% coverage: {g['multi_complete']} ({100*g['multi_complete']/max(1,g['multi_action_tasks']):.1f}%)", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"LAYER 4 — EFFECT CLASSIFICATION", flush=True)
print(f"{'='*60}", flush=True)
print(f"Conditional (aligned={g['effect_cond_total']}):", flush=True)
print(f"  Correct: {g['effect_cond_correct']} ({100*g['effect_cond_correct']/max(1,g['effect_cond_total']):.1f}%)", flush=True)
print(f"End-to-end (all {n_ref}):", flush=True)
print(f"  Correct: {g['effect_e2e_correct']} ({100*g['effect_e2e_correct']/max(1,n_ref):.1f}%)", flush=True)

# Per-class effect
ref_eff_dist = Counter()
for s in samples:
    for e in s["ref_effects"]:
        ref_eff_dist[e] += 1

print(f"\nPer-class (end-to-end):", flush=True)
for ec in ("read_only", "reversible_state", "irreversible_world", "other"):
    ref = ref_eff_dist.get(ec, 0)
    conf = effect_conf[ec]
    correct = conf.get(ec, 0)
    missing = conf.get("MISSING", 0)
    wrong = sum(c for k, c in conf.items() if k not in (ec, "MISSING"))
    print(f"  {ec:22s}: ref={ref:5d} correct={correct:4d} "
          f"({100*correct/max(1,ref):5.1f}%) missing={missing:5d} "
          f"({100*missing/max(1,ref):5.1f}%) wrong={wrong:4d}", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"LAYER 2c — RETRY", flush=True)
print(f"{'='*60}", flush=True)
print(f"Retry nodes: {g['retry_nodes']}", flush=True)
if g["retry_nodes"] == 0:
    print(f"Unsafe retry: N/A (vacuous — no retry nodes)", flush=True)
else:
    print(f"Unsafe retry: {g['unsafe_retry']}/{g['retry_nodes']}", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"PER-DOMAIN", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Domain':<20s} {'n':>5s} {'Valid%':>7s} {'RefSem':>7s} {'Match':>7s} {'Match%':>7s} {'Pred/T':>7s} {'Ref/T':>7s}", flush=True)
for dom in DOMAINS:
    dm = dom_metrics[dom]
    if dm["n"] == 0: continue
    print(f"{dom:<20s} {dm['n']:>5d} {100*dm['valid']/dm['n']:>6.1f}% "
          f"{dm['ref_sem']:>7d} {dm['matched']:>7d} "
          f"{100*dm['matched']/max(1,dm['ref_sem']):>6.1f}% "
          f"{dm['pred_count']/dm['n']:>7.2f} {dm['ref_count']/dm['n']:>7.2f}", flush=True)

# Evidence distribution
print(f"\nOracle evidence distribution: {dict(evidence_dist)}", flush=True)

# ── Save ──
summary = {
    "layer_1": {"parse_pct": round(100*g["parse_ok"]/n_tasks,2),
                 "valid_pct": round(100*g["valid"]/n_tasks,2)},
    "layer_2a_semantic": {
        "ref_total": n_ref, "matched": g["ref_sem_matched"],
        "missing": g["ref_sem_missing"],
        "pred_total": g["pred_total"], "extra": g["pred_extra"],
        "precision": round(100*sem_p,2), "recall": round(100*sem_r,2),
        "f1": round(100*sem_f1,2),
        "sequence_exact": g["sem_seq_exact"],
        "sequence_total": g["sem_seq_total"],
    },
    "layer_2b_planning": {
        "ref_per_task": round(g["ref_count_sum"]/n_tasks,2),
        "pred_per_task": round(g["pred_count_sum"]/n_tasks,2),
        "count_ratio": round(g["pred_count_sum"]/max(1,g["ref_count_sum"]),2),
        "multi_action_tasks": g["multi_action_tasks"],
        "multi_50pct_coverage": g["multi_complete"],
    },
    "layer_4_effect": {
        "conditional_total": g["effect_cond_total"],
        "conditional_correct": g["effect_cond_correct"],
        "conditional_pct": round(100*g["effect_cond_correct"]/max(1,g["effect_cond_total"]),2),
        "e2e_correct": g["effect_e2e_correct"],
        "e2e_total": n_ref,
        "e2e_pct": round(100*g["effect_e2e_correct"]/max(1,n_ref),2),
    },
    "layer_2c_retry": {
        "retry_nodes": g["retry_nodes"],
        "unsafe_retry": g["unsafe_retry"],
        "note": "N/A if 0 retry nodes" if g["retry_nodes"]==0 else "",
    },
    "per_domain": {dom: dict(dm) for dom, dm in dom_metrics.items() if dm["n"]>0},
    "effect_confusion": {rc: dict(c) for rc, c in effect_conf.items()},
    "evidence_dist": dict(evidence_dist),
    "accounting": {
        "tasks": f"{n_tasks}/{len(samples)}",
        "ref_actions": f"{g['ref_sem_matched']+g['ref_sem_missing']}/{n_ref}",
    },
}

with open(os.path.join(OUT_DIR, "metrics_v3.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Manifest
pred_sha = hashlib.sha256(open(PRED_FILE,"rb").read()).hexdigest()
manifest = {
    "model": "runs/phase5b1/e1_3b_qlora/final",
    "prediction_file": PRED_FILE,
    "prediction_sha256": pred_sha,
    "prediction_count": len(preds),
    "reference_total": n_ref,
    "oracle": "src/eval/tau3_skill_oracle.py",
    "protocol": "semantic-skill aligned (not concrete-name match)",
}
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nSaved to {OUT_DIR}/metrics_v3.json", flush=True)
print("DONE", flush=True)
