"""Phase 5C follow-up 3: BFCL semantic evaluation under MATCHED protocols.

Reviewer-mandated third supplement. The main eval only measured BFCL Valid —
structural legality. The schema hypothesis (schema training → semantic
grounding) must be tested with the frozen BFCL semantic evaluator
(audit_bfcl_ontology.py rules, unchanged).

Matched inference protocol:
  E1-A / E5C-D : instruction only          (they were trained without schema)
  E5C-S / E5C-SD: instruction + BFCL capability schemas (training format:
                 build_capability_prompt, same as train_phase5c.py)

E1-A preds are reused from the frozen phase 5B-1 run (bare protocol — matched).
Frozen adjudication logic (SKILL_FAMILIES / EQUIVALENT_SETS / classify_relation)
is copied verbatim from scripts/audit_bfcl_ontology.py.

Runs after eval_5c_tau3.py finishes (waits for GPU).
"""
import json, sys, os, time
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")

# Wait for the tau3 semantic eval to finish
while True:
    if not os.popen("pgrep -f eval_5c_tau3.py").read().strip():
        break
    print("waiting for eval_5c_tau3.py ...", flush=True)
    time.sleep(60)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.capability_format import build_capability_prompt
from src.eval.adapters import bfcl as bfcl_adapter
from src.eval.oracle_bfcl import oracle as bfcl_oracle
from src.ir.parser import parse_text, TaskIRSyntaxError
from src.validator.validator import validate
from collections import Counter, defaultdict

# ── Frozen adjudication rules (verbatim from audit_bfcl_ontology.py) ──
SKILL_FAMILIES = {
    "retrieval": {"SEARCH", "FETCH", "QUERY_DB"},
    "transform": {"FILTER", "TRANSFORM", "EXTRACT", "DEDUP", "SORT", "JOIN", "MERGE"},
    "compute": {"ARGMIN", "ARGMAX", "MIN", "MAX", "SUM", "AVG", "COUNT",
                "CALCULATE", "COMPARE", "CONVERT"},
    "lm": {"GENERATE", "SUMMARIZE", "TRANSLATE", "CLASSIFY", "EXTRACT_ENTITIES",
           "CODEGEN", "PLAN"},
    "action": {"SEND", "EXEC_ACTION"},
    "io": {"LOAD", "SAVE"},
    "control": {"VERIFY", "SELECT"},
}
EQUIVALENT_SETS = [
    {"SEARCH", "FETCH"},
    {"ARGMIN", "MIN"},
    {"ARGMAX", "MAX"},
    {"SUM", "CALCULATE"},
]
EXEC_ACTION_CANONICAL = {
    "search": "SEARCH", "find": "SEARCH", "get": "SEARCH",
    "retrieve": "SEARCH", "lookup": "SEARCH", "query": "QUERY_DB",
    "send": "SEND", "email": "SEND", "notify": "SEND",
    "create": "EXEC_ACTION", "add": "EXEC_ACTION", "update": "EXEC_ACTION",
    "delete": "EXEC_ACTION", "book": "EXEC_ACTION", "reserve": "EXEC_ACTION",
    "calculate": "CALCULATE", "compute": "CALCULATE",
    "convert": "CONVERT", "translate_currency": "CONVERT",
}


def get_skill_family(skill):
    for fam, skills in SKILL_FAMILIES.items():
        if skill in skills:
            return fam
    return "unknown"


def are_equivalent(skill_a, skill_b):
    if skill_a == skill_b:
        return "EXACT"
    for eset in EQUIVALENT_SETS:
        if skill_a in eset and skill_b in eset:
            return "EQUIVALENT"
    return None


def classify_relation(oracle_skill, pred_skill, pred_params=None):
    if oracle_skill == pred_skill:
        return "EXACT"
    eq = are_equivalent(oracle_skill, pred_skill)
    if eq:
        return eq
    oa = get_skill_family(oracle_skill)
    pa = get_skill_family(pred_skill)
    if oa == pa and oa != "unknown":
        if pred_skill == "EXEC_ACTION":
            return "MODEL_MORE_GENERAL"
        elif oracle_skill == "EXEC_ACTION":
            return "MODEL_MORE_SPECIFIC"
        else:
            return "SAME_FAMILY_DIFFERENT_OP"
    if pred_skill == "EXEC_ACTION" and pred_params:
        action = str(pred_params.get("action", "")).lower()
        for prefix, canonical in EXEC_ACTION_CANONICAL.items():
            if action.startswith(prefix):
                if canonical == oracle_skill:
                    return "ONTOLOGY_ALIAS"
                elif get_skill_family(canonical) == get_skill_family(oracle_skill):
                    return "MODEL_MORE_GENERAL"
                break
    if pred_skill == "GENERATE":
        return "GENERIC_FALLBACK"
    return "INCOMPATIBLE"
# ── end frozen rules ──


# ── Load samples + oracle (once) ──
samples = bfcl_adapter.load()
print(f"BFCL samples: {len(samples)}", flush=True)
by_case = {s.case_id: s for s in samples}
n_with_caps = sum(1 for s in samples if s.capabilities)
print(f"with function schemas: {n_with_caps}", flush=True)

oracle_cache = {}
for s in samples:
    oracle_cache[s.case_id] = bfcl_oracle(s)
print("oracle computed", flush=True)


def semantic_metrics(pred_records):
    """Frozen three-metric evaluation over preds records."""
    n = len(pred_records)
    parse_ok = valid = 0
    strict = eq = func = comparable_n = 0
    by_rep = defaultdict(lambda: {"strict": 0, "eq": 0, "func": 0, "n": 0})
    for p in pred_records:
        rep = p["representability"]
        by_rep[rep]["n"] += 1
        o = oracle_cache.get(p["case_id"], {})
        try:
            module = parse_text(p["taskir_text"])
            parse_ok += 1
        except (TaskIRSyntaxError, Exception):
            continue
        if validate(module).valid:
            valid += 1
        else:
            continue
        if not o.get("module") or rep not in ("full", "partial"):
            continue
        comparable_n += 1
        ref_ops = [x.op for x in o["module"].program.nodes]
        pred_nodes = module.program.nodes
        pred_ops = [x.op for x in pred_nodes]
        if ref_ops == pred_ops:
            strict += 1; eq += 1; func += 1
            by_rep[rep]["strict"] += 1; by_rep[rep]["eq"] += 1; by_rep[rep]["func"] += 1
            continue
        all_exact = all_equiv = all_func = True
        for i in range(max(len(ref_ops), len(pred_ops))):
            ref_op = ref_ops[i] if i < len(ref_ops) else "MISSING"
            pred_op = pred_ops[i] if i < len(pred_ops) else "MISSING"
            pred_p = pred_nodes[i].params if i < len(pred_nodes) else {}
            rel = classify_relation(ref_op, pred_op, pred_p)
            if rel != "EXACT":
                all_exact = False
            if rel not in ("EXACT", "EQUIVALENT", "ONTOLOGY_ALIAS"):
                all_equiv = False
            if rel not in ("EXACT", "EQUIVALENT", "ONTOLOGY_ALIAS",
                           "SAME_FAMILY_DIFFERENT_OP", "MODEL_MORE_SPECIFIC"):
                all_func = False
        if all_equiv:
            eq += 1; by_rep[rep]["eq"] += 1
        if all_func:
            func += 1; by_rep[rep]["func"] += 1
    d = max(1, comparable_n)
    return {
        "n": n, "parse_pct": round(100*parse_ok/n, 2),
        "valid_pct": round(100*valid/n, 2),
        "comparable_n": comparable_n,
        "strict_pct": round(100*strict/d, 2),
        "ontology_equivalent_pct": round(100*eq/d, 2),
        "functional_pct": round(100*func/d, 2),
        "by_representability": {k: v for k, v in by_rep.items()},
    }


# ── E1-A: reuse frozen preds (matched bare protocol) ──
e1_preds = []
with open("runs/phase5b1/bfcl_v4_full_preds.jsonl") as f:
    for line in f:
        if line.strip():
            e1_preds.append(json.loads(line))
print(f"E1-A frozen preds: {len(e1_preds)}", flush=True)

os.makedirs("runs/phase5c", exist_ok=True)
results = {"E1-A": semantic_metrics(e1_preds)}
print(f"E1-A: strict={results['E1-A']['strict_pct']}% eq={results['E1-A']['ontology_equivalent_pct']}% "
      f"func={results['E1-A']['functional_pct']}%", flush=True)

# ── New models: matched-protocol inference ──
tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto")

NEW = {
    "E5C-S":  ("runs/phase5c/e5c_s/final", True),
    "E5C-D":  ("runs/phase5c/e5c_d/final", False),
    "E5C-SD": ("runs/phase5c/e5c_sd/final", True),
}

for name, (adapter, use_schema) in NEW.items():
    print(f"\n{'='*60}\n{name} (schema at inference: {use_schema})\n{'='*60}", flush=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()

    prompts = []
    for s in samples:
        user = s.instruction
        if use_schema and s.capabilities:
            user = build_capability_prompt(s.instruction, s.capabilities)
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

    print(f"  inferring {len(prompts)} ...", flush=True)
    completions = []
    for bs in range(0, len(prompts), 16):
        if bs % 960 == 0:
            print(f"    {bs}/{len(prompts)}", flush=True)
        batch = prompts[bs:bs+16]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512,
                                    do_sample=False, temperature=None,
                                    pad_token_id=tok.pad_token_id)
        completions.extend(tok.decode(output[j][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
                           for j in range(len(batch)))

    recs = []
    for s, comp in zip(samples, completions):
        recs.append({
            "case_id": s.case_id,
            "category": s.metadata.get("category", ""),
            "kind": s.metadata.get("kind", ""),
            "representability": oracle_cache[s.case_id].get("status", "none"),
            "instruction": s.instruction,
            "taskir_text": extract_taskir_text(comp),
        })
    out_path = f"runs/phase5c/{name.lower().replace('-', '_')}_bfcl_preds.jsonl"
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  saved {out_path}", flush=True)

    results[name] = semantic_metrics(recs)
    m = results[name]
    print(f"  valid={m['valid_pct']}% strict={m['strict_pct']}% "
          f"eq={m['ontology_equivalent_pct']}% func={m['functional_pct']}%", flush=True)

    del model
    torch.cuda.empty_cache()

with open("results/phase5c/bfcl_semantic.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/phase5c/bfcl_semantic.json", flush=True)
print("DONE", flush=True)
