"""τ³ Exhaustive Offline Evaluation — Phase 5B-2B.1.

Accounts for ALL 2,546 tasks and ALL 14,834 reference actions.
Separates model evaluation from IR representability analysis.
Produces reproducible artifacts with full audit trail.

Usage: python scripts/eval_tau3_offline.py [--skip-inference]
  --skip-inference: reuse existing predictions from runs/phase5b1/tau3_preds.jsonl
"""
import json, sys, os, time, random, hashlib, argparse
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")

# ── Config ──
parser = argparse.ArgumentParser()
parser.add_argument("--skip-inference", action="store_true")
args = parser.parse_args()

PRED_FILE = "runs/phase5b1/tau3_preds.jsonl"
OUT_DIR = "results/tau3_offline"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load τ³ data ──
TAU3_DIR = "data/external_benchmarks/tau3_bench/domains"
DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")

import re
READ_ONLY = re.compile(r"^(get|find|search|query|list|retrieve|lookup|check|show|view|read|calculate|compare)", re.I)
IRREVERSIBLE = re.compile(r"(cancel|refund|purchase|pay|book|reserv|send|delete|remove|grant|revoke|reboot|reset|refuel|unlock)", re.I)
REVERSIBLE = re.compile(r"(toggle|enable|disable|set|update|change|modify|edit|switch|turn_on|turn_off|add|create|insert)", re.I)

def classify_action_effect(name):
    if READ_ONLY.match(name or ""): return "read_only"
    if IRREVERSIBLE.search(name or ""): return "irreversible_world"
    if REVERSIBLE.search(name or ""): return "reversible_state"
    return "other"

SKILL_EFFECTS = {
    "SEARCH": "read_only", "FETCH": "read_only", "QUERY_DB": "read_only",
    "LOAD": "read_only", "FILTER": "read_only", "TRANSFORM": "read_only",
    "EXTRACT": "read_only", "DEDUP": "read_only", "SORT": "read_only",
    "JOIN": "read_only", "MERGE": "read_only",
    "ARGMIN": "read_only", "ARGMAX": "read_only", "MIN": "read_only",
    "MAX": "read_only", "SUM": "read_only", "AVG": "read_only",
    "COUNT": "read_only", "CALCULATE": "read_only", "COMPARE": "read_only",
    "CONVERT": "read_only", "GENERATE": "read_only", "SUMMARIZE": "read_only",
    "TRANSLATE": "read_only", "CLASSIFY": "read_only",
    "EXTRACT_ENTITIES": "read_only", "CODEGEN": "read_only", "PLAN": "read_only",
    "VERIFY": "read_only", "SELECT": "read_only",
    "SEND": "irreversible_world",
    "EXEC_ACTION": "unknown",
    "SAVE": "reversible_state",
}
POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

def skill_effect(skill, params=None):
    if skill in SKILL_EFFECTS:
        ec = SKILL_EFFECTS[skill]
        if ec == "unknown" and params:
            return classify_action_effect(str(params.get("action", "")))
        return ec
    return "unknown"

# Build samples
samples = []
for dom in DOMAINS:
    dpath = os.path.join(TAU3_DIR, dom)
    if not os.path.isdir(dpath): continue
    with open(os.path.join(dpath, "tasks.json")) as f:
        tasks = json.load(f)
    for t in tasks:
        us = t.get("user_scenario") or {}
        instrs = us.get("instructions") or {}
        if isinstance(instrs, dict):
            instrs = instrs.get("task_instructions", "")
        if not isinstance(instrs, str):
            instrs = json.dumps(instrs, ensure_ascii=False)[:300]
        desc = t.get("description") or ""
        if not isinstance(desc, str):
            desc = json.dumps(desc, ensure_ascii=False)[:300]
        instruction = desc + ("\n\n" + instrs if instrs else "")

        ec = t.get("evaluation_criteria") or {}
        ref_actions = ec.get("actions") or []
        ref_names = [a.get("name", a.get("action_id", "")) for a in ref_actions]
        ref_effects = [classify_action_effect(n) for n in ref_names]

        samples.append({
            "case_id": f"{dom}-{t.get('id','')}",
            "domain": dom,
            "instruction": instruction,
            "instruction_len": len(instruction),
            "ref_action_names": ref_names,
            "ref_effects": ref_effects,
            "n_ref": len(ref_names),
        })

print(f"τ³ samples: {len(samples)}", flush=True)
print(f"Total reference actions: {sum(s['n_ref'] for s in samples)}", flush=True)
ref_effect_counts = {}
for s in samples:
    for e in s["ref_effects"]:
        ref_effect_counts[e] = ref_effect_counts.get(e, 0) + 1
print(f"Reference effect distribution: {ref_effect_counts}", flush=True)

# ── Inference (or load existing) ──
if args.skip_inference and os.path.exists(PRED_FILE):
    print("Loading existing predictions...", flush=True)
    preds = []
    with open(PRED_FILE) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    print(f"Loaded {len(preds)} predictions", flush=True)
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text

    BATCH = 16
    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, "runs/phase5b1/e1_3b_qlora/final")
    model.eval()

    preds = []
    t0 = time.time()
    for bs in range(0, len(samples), BATCH):
        batch = samples[bs:bs+BATCH]
        if bs % 320 == 0:
            el = time.time() - t0
            rate = bs / max(1, el)
            eta = int((len(samples) - bs) / max(0.1, rate))
            print(f"Batch {bs}/{len(samples)} rate={rate:.1f}/s eta={eta}s", flush=True)

        prompts = []
        for s in batch:
            p = tok.apply_chat_template(
                [{"role":"system","content":SYSTEM_PROMPT},
                 {"role":"user","content":s["instruction"]}],
                tokenize=False, add_generation_prompt=True)
            prompts.append(p)

        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512,
                                     do_sample=False, temperature=None,
                                     pad_token_id=tok.pad_token_id)
        for j, s in enumerate(batch):
            completion = tok.decode(outputs[j][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            preds.append({**s, "taskir_text": extract_taskir_text(completion),
                         "output_len": len(completion)})

    with open(PRED_FILE, "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(preds)} predictions to {PRED_FILE}", flush=True)

# ── Evaluation ──
print("\n" + "="*70, flush=True)
print("EXHAUSTIVE τ³ EVALUATION", flush=True)
print("="*70, flush=True)

from src.ir.parser import parse_text, TaskIRSyntaxError
from src.validator.validator import validate
from collections import Counter, defaultdict

# Global counters
g = {
    "total_tasks": len(preds),
    "parse_ok": 0, "parse_fail": 0,
    "valid": 0, "invalid": 0,
    # Action accounting (must sum to 14,834 for reference side)
    "ref_total": 0, "ref_matched": 0, "ref_missing": 0,
    "pred_action_total": 0, "pred_extra": 0,  # extra = not matched to any ref
    # Effect classification
    "effect_conditional_correct": 0, "effect_conditional_total": 0,
    "effect_e2e_correct": 0,  # over ALL reference actions
    # Retry
    "retry_nodes": 0, "unsafe_retry": 0,
    # Ordering
    "order_correct": 0, "order_total": 0,
}

# Effect confusion WITH missing column
effect_conf = defaultdict(lambda: Counter())

# Per-domain
dom_metrics = defaultdict(lambda: {
    "n": 0, "parse": 0, "valid": 0,
    "ref_total": 0, "matched": 0, "missing": 0,
    "extra": 0,
    "effect_cond_correct": 0, "effect_cond_total": 0,
    "instruction_len_sum": 0, "output_len_sum": 0,
    "parse_errors": Counter(),
})

# Error cases for debugging
error_cases = []

for p in preds:
    dom = p["domain"]
    dm = dom_metrics[dom]
    dm["n"] += 1
    dm["instruction_len_sum"] += p.get("instruction_len", 0)
    dm["output_len_sum"] += p.get("output_len", 0)

    # Reference actions
    ref_names = p["ref_action_names"]
    ref_effects = p["ref_effects"]
    n_ref = len(ref_names)
    g["ref_total"] += n_ref
    dm["ref_total"] += n_ref

    # Parse
    parse_ok = True
    module = None
    try:
        module = parse_text(p["taskir_text"])
    except TaskIRSyntaxError as e:
        parse_ok = False
        dm["parse_errors"]["syntax"] += 1
        error_cases.append({"case_id": p["case_id"], "type": "parse",
                           "error": str(e)[:100]})
    if parse_ok:
        g["parse_ok"] += 1
        dm["parse"] += 1
    else:
        g["parse_fail"] += 1
        # All reference actions in unparseable outputs are MISSING
        g["ref_missing"] += n_ref
        dm["missing"] += n_ref
        for re in ref_effects:
            effect_conf[re]["MISSING"] += 1
        continue

    # Validate
    valid = validate(module).valid
    if valid:
        g["valid"] += 1
        dm["valid"] += 1
    else:
        g["invalid"] += 1
        dm["parse_errors"]["validator"] += 1
        error_cases.append({"case_id": p["case_id"], "type": "validate"})
        g["ref_missing"] += n_ref
        dm["missing"] += n_ref
        for re in ref_effects:
            effect_conf[re]["MISSING"] += 1
        continue

    # Extract predicted action nodes
    pred_nodes = [n for n in module.program.nodes if n.op not in POLICY_OPS]
    g["pred_action_total"] += len(pred_nodes)

    # Action matching (name-based, case-insensitive substring)
    matched_ref_indices = set()
    matched_pred_indices = set()

    for ri, rn in enumerate(ref_names):
        rn_l = rn.lower()
        for pi, pn in enumerate(pred_nodes):
            if pi in matched_pred_indices:
                continue
            paction = str(pn.params.get("action", "")).lower()
            pop = pn.op.lower()
            # Match if action name appears in either op name or action param
            if rn_l in paction or rn_l in pop or \
               (paction and paction in rn_l):
                matched_ref_indices.add(ri)
                matched_pred_indices.add(pi)
                break

    g["ref_matched"] += len(matched_ref_indices)
    g["ref_missing"] += n_ref - len(matched_ref_indices)
    dm["matched"] += len(matched_ref_indices)
    dm["missing"] += n_ref - len(matched_ref_indices)

    n_extra = len(pred_nodes) - len(matched_pred_indices)
    g["pred_extra"] += n_extra
    dm["extra"] += n_extra

    # Effect classification for matched pairs
    for ri in matched_ref_indices:
        re = ref_effects[ri]
        # Find the matching predicted node
        for pi, pn in enumerate(pred_nodes):
            if pi in matched_pred_indices and \
               matched_ref_indices and \
               any(rn.lower() in str(pn.params.get("action","")).lower()
                   or rn.lower() in pn.op.lower()
                   for rn in [ref_names[ri]]):
                pe = skill_effect(pn.op, pn.params)
                g["effect_conditional_total"] += 1
                dm["effect_cond_total"] += 1
                effect_conf[re][pe] += 1
                if re == pe:
                    g["effect_conditional_correct"] += 1
                    dm["effect_cond_correct"] += 1
                    g["effect_e2e_correct"] += 1
                break

    # Missing actions get MISSING in confusion
    for ri in range(n_ref):
        if ri not in matched_ref_indices:
            effect_conf[ref_effects[ri]]["MISSING"] += 1

    # Retry analysis
    for n in module.program.nodes:
        if n.retry:
            g["retry_nodes"] += 1
            eff = skill_effect(n.op, n.params)
            if n.retry.on != "error" and eff == "irreversible_world":
                g["unsafe_retry"] += 1

    # Ordering (multi-action)
    if n_ref > 1:
        g["order_total"] += 1
        has_ordering = any(n.after for n in module.program.nodes)
        if has_ordering or len(pred_nodes) <= 1:
            g["order_correct"] += 1

# ── Compute final metrics ──
n_tasks = g["total_tasks"]
n_ref = g["ref_total"]

print(f"\n{'='*60}", flush=True)
print(f"LAYER 1 — STRUCTURAL", flush=True)
print(f"{'='*60}", flush=True)
print(f"Tasks: {n_tasks}", flush=True)
print(f"Parse OK: {g['parse_ok']} ({100*g['parse_ok']/n_tasks:.1f}%)", flush=True)
print(f"Validator: {g['valid']} ({100*g['valid']/n_tasks:.1f}%)", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"LAYER 2 — ACTION SEMANTIC", flush=True)
print(f"{'='*60}", flush=True)
print(f"Reference actions: {n_ref}", flush=True)
print(f"Matched: {g['ref_matched']} ({100*g['ref_matched']/n_ref:.1f}%)", flush=True)
print(f"Missing: {g['ref_missing']} ({100*g['ref_missing']/n_ref:.1f}%)", flush=True)
print(f"Predicted actions: {g['pred_action_total']}", flush=True)
print(f"Extra (unmatched): {g['pred_extra']}", flush=True)

action_precision = g["ref_matched"] / max(1, g["pred_action_total"])
action_recall = g["ref_matched"] / max(1, n_ref)
action_f1 = 2 * action_precision * action_recall / max(1e-9, action_precision + action_recall)
print(f"Action Precision: {100*action_precision:.1f}%", flush=True)
print(f"Action Recall: {100*action_recall:.1f}%", flush=True)
print(f"Action F1: {100*action_f1:.1f}%", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"LAYER 2b — EFFECT CLASSIFICATION", flush=True)
print(f"{'='*60}", flush=True)

# Conditional (aligned only)
cond_total = g["effect_conditional_total"]
cond_correct = g["effect_conditional_correct"]
cond_acc = 100 * cond_correct / max(1, cond_total)
print(f"Conditional (aligned={cond_total}):", flush=True)
print(f"  Effect accuracy: {cond_correct}/{cond_total} = {cond_acc:.1f}%", flush=True)

# End-to-end (all reference actions as denominator)
e2e_correct = g["effect_e2e_correct"]
e2e_acc = 100 * e2e_correct / max(1, n_ref)
print(f"End-to-end (all {n_ref} ref actions):", flush=True)
print(f"  Effect correct: {e2e_correct}/{n_ref} = {e2e_acc:.1f}%", flush=True)

# Per-class recall
print(f"\nPer-class effect recall (end-to-end):", flush=True)
for ec in ("read_only", "reversible_state", "irreversible_world", "other"):
    ref_count = ref_effect_counts.get(ec, 0)
    conf = effect_conf[ec]
    correct = conf.get(ec, 0)
    missing = conf.get("MISSING", 0)
    wrong = sum(c for k, c in conf.items() if k not in (ec, "MISSING"))
    print(f"  {ec:22s}: ref={ref_count:5d} correct={correct:4d} "
          f"({100*correct/max(1,ref_count):5.1f}%) missing={missing:5d} "
          f"({100*missing/max(1,ref_count):5.1f}%) wrong={wrong:4d}", flush=True)

# Confusion matrix with MISSING column
print(f"\nEffect Confusion Matrix (including MISSING):", flush=True)
all_classes = ["read_only", "reversible_state", "irreversible_world", "other", "unknown", "MISSING"]
hdr = "Ref\Pred"
print(f"{hdr:<22s}", end="", flush=True)
for pc in all_classes:
    print(f"{pc[:12]:>13s}", end="", flush=True)
print(flush=True)
for rc in ("read_only", "reversible_state", "irreversible_world", "other"):
    print(f"{rc[:20]:<22s}", end="", flush=True)
    for pc in all_classes:
        cnt = effect_conf[rc][pc]
        print(f"{cnt:>13d}", end="", flush=True)
    print(flush=True)

print(f"\n{'='*60}", flush=True)
print(f"LAYER 2c — RETRY SAFETY", flush=True)
print(f"{'='*60}", flush=True)
print(f"Retry nodes generated: {g['retry_nodes']}", flush=True)
if g["retry_nodes"] == 0:
    print(f"Unsafe retry rate: N/A (no retry nodes generated)", flush=True)
else:
    print(f"Unsafe retry: {g['unsafe_retry']}/{g['retry_nodes']}", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"LAYER 3 — IR/ARCHITECTURE (independent of model)", flush=True)
print(f"{'='*60}", flush=True)

# Static representability (from reference actions, not predictions)
v01_full = v01_partial = v01_none = 0
v02_full = v02_partial = v02_none = 0

for s in samples:
    if s["n_ref"] <= 1:
        continue
    effects = set(s["ref_effects"])

    # TaskIR v0.1 (no effect system, all actions independent)
    # Can express ordering via after edges but no atomicity
    v01_partial += 1  # v0.1 always partial for multi-action stateful

    # v0.2 proposal (per-class effect chains)
    if len(effects) == 1:
        v02_full += 1
    elif effects <= {"read_only", "reversible_state"}:
        v02_full += 1  # safe to chain without atomicity
    elif "irreversible_world" in effects:
        v02_partial += 1  # needs cross-class transaction
    else:
        v02_partial += 1

n_multi = sum(1 for s in samples if s["n_ref"] > 1)
print(f"Multi-action tasks: {n_multi}", flush=True)
print(f"\nTaskIR v0.1 (no effect system):", flush=True)
print(f"  Full: 0 (no effect semantics)", flush=True)
print(f"  Partial: {n_multi} (ordering only via after edges)", flush=True)
print(f"  None: 0", flush=True)
print(f"\nEffect-system v0.2 proposal (per-class chains):", flush=True)
print(f"  Full: {v02_full} ({100*v02_full/max(1,n_multi):.1f}%)", flush=True)
print(f"  Partial: {v02_partial} ({100*v02_partial/max(1,n_multi):.1f}%)", flush=True)
print(f"  None: {v02_none}", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"PER-DOMAIN BREAKDOWN", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Domain':<20s} {'n':>5s} {'Parse%':>7s} {'Valid%':>7s} {'Match%':>7s} {'Miss%':>7s} {'EffAcc':>7s} {'AvgLen':>7s}", flush=True)
print("-" * 70, flush=True)
for dom in DOMAINS:
    dm = dom_metrics[dom]
    if dm["n"] == 0: continue
    print(f"{dom:<20s} {dm['n']:>5d} {100*dm['parse']/dm['n']:>6.1f}% "
          f"{100*dm['valid']/dm['n']:>6.1f}% "
          f"{100*dm['matched']/max(1,dm['ref_total']):>6.1f}% "
          f"{100*dm['missing']/max(1,dm['ref_total']):>6.1f}% "
          f"{100*dm['effect_cond_correct']/max(1,dm['effect_cond_total']):>6.1f}% "
          f"{dm['instruction_len_sum']/dm['n']:>7.0f}", flush=True)

# Banking domain analysis
bank = dom_metrics.get("banking_knowledge", {})
if bank.get("n", 0) > 0:
    print(f"\nBanking domain analysis:", flush=True)
    print(f"  Avg instruction length: {bank['instruction_len_sum']/bank['n']:.0f} chars", flush=True)
    print(f"  Parse errors: {dict(bank['parse_errors'])}", flush=True)
    other_doms = [dom_metrics[d] for d in DOMAINS if d != "banking_knowledge" and dom_metrics[d]["n"] > 0]
    if other_doms:
        avg_other = sum(d["instruction_len_sum"]/d["n"] for d in other_doms) / len(other_doms)
        print(f"  Avg other domains: {avg_other:.0f} chars", flush=True)

# ── Save artifacts ──
summary = {
    "layer_1_structural": {
        "total_tasks": n_tasks,
        "parse_ok": g["parse_ok"], "parse_pct": round(100*g["parse_ok"]/n_tasks, 2),
        "valid": g["valid"], "valid_pct": round(100*g["valid"]/n_tasks, 2),
    },
    "layer_2_action_semantic": {
        "ref_total": n_ref,
        "ref_matched": g["ref_matched"],
        "ref_missing": g["ref_missing"],
        "pred_total": g["pred_action_total"],
        "pred_extra": g["pred_extra"],
        "action_precision": round(100*action_precision, 2),
        "action_recall": round(100*action_recall, 2),
        "action_f1": round(100*action_f1, 2),
    },
    "layer_2b_effect": {
        "conditional_total": cond_total,
        "conditional_correct": cond_correct,
        "conditional_acc": round(cond_acc, 2),
        "e2e_total": n_ref,
        "e2e_correct": e2e_correct,
        "e2e_acc": round(e2e_acc, 2),
        "per_class": {
            ec: {
                "ref": ref_effect_counts.get(ec, 0),
                "correct": effect_conf[ec].get(ec, 0),
                "missing": effect_conf[ec].get("MISSING", 0),
                "wrong": sum(c for k, c in effect_conf[ec].items()
                              if k not in (ec, "MISSING")),
            } for ec in ("read_only", "reversible_state", "irreversible_world", "other")
        },
    },
    "layer_2c_retry": {
        "retry_nodes": g["retry_nodes"],
        "unsafe_retry": g["unsafe_retry"],
        "note": "N/A if retry_nodes=0 (vacuous)" if g["retry_nodes"] == 0 else "",
    },
    "layer_3_representability": {
        "multi_action_tasks": n_multi,
        "v01": {"full": 0, "partial": n_multi, "none": 0},
        "v02_proposal": {"full": v02_full, "partial": v02_partial, "none": v02_none},
        "note": "Independent of model predictions; derived from reference actions",
    },
    "per_domain": {
        dom: {
            "n": dm["n"],
            "parse_pct": round(100*dm["parse"]/dm["n"], 2),
            "valid_pct": round(100*dm["valid"]/dm["n"], 2),
            "ref_total": dm["ref_total"],
            "matched": dm["matched"],
            "missing": dm["missing"],
            "extra": dm["extra"],
            "avg_instruction_len": round(dm["instruction_len_sum"]/dm["n"]),
        } for dom, dm in dom_metrics.items() if dm["n"] > 0
    },
    "effect_confusion": {
        rc: dict(effect_conf[rc]) for rc in effect_conf
    },
    "accounting": {
        "tasks": f"{n_tasks}/{len(samples)}",
        "ref_actions": f"{g['ref_matched'] + g['ref_missing']}/{n_ref}",
        "assertion": "matched + missing = total reference actions",
    },
}

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(OUT_DIR, "effect_confusion.json"), "w") as f:
    json.dump(summary["effect_confusion"], f, indent=2)

if error_cases:
    with open(os.path.join(OUT_DIR, "error_cases.jsonl"), "w") as f:
        for e in error_cases[:100]:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

# Manifest
manifest = {
    "model": "runs/phase5b1/e1_3b_qlora/final",
    "base_model": "weights/Qwen2.5-3B-Instruct",
    "prediction_file": PRED_FILE,
    "prediction_count": len(preds),
    "reference_total": n_ref,
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "protocol": "oracle_task_view (offline compiler analysis ONLY)",
}
with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nArtifacts saved to {OUT_DIR}/", flush=True)
print(f"  metrics.json", flush=True)
print(f"  effect_confusion.json", flush=True)
print(f"  manifest.json", flush=True)
if error_cases:
    print(f"  error_cases.jsonl ({len(error_cases)} cases)", flush=True)
print(f"\nDONE", flush=True)
