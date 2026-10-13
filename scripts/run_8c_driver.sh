#!/bin/bash
set -e
cd /ccfa2026/compile-anything
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/bin/python

echo "=== M0 deep training (v4) ==="
$PY scripts/train_m0_m1.py --model m0 --seed 43

echo "=== patch budget ==="
$PY - << 'PYEOF'
import json
log = json.load(open('runs/phase8b/m0_s43/train_log.json'))
mp = 'experiments/phase8b/m0_m1_training_manifest.json'
man = json.load(open(mp))
man['budget']['max_steps'] = log['max_steps']
json.dump(man, open(mp, 'w'), indent=2)
print('budget =', log['max_steps'])
PYEOF

echo "=== M1 deep training (v4) ==="
$PY scripts/train_m0_m1.py --model m1 --seed 43

echo "=== evaluation (incl deep test) ==="
$PY - << 'PYEOF'
import sys, os
sys.path.insert(0, '.')
os.chdir('/ccfa2026/compile-anything')
import json, time, re, gc, random
import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.compiler.plan_skeleton import skeleton_from_dict, validate_skeleton, hard_errors
from src.compiler.skeleton_lowering import lower_skeleton
from src.eval.semantic_eval_v3 import evaluate_v3
from scripts.train_m0_m1 import SYS_M0, SYS_M1, caps_block

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
SPLITS = {"test_sft": "data/semantic_skeleton_v4/test_sft.jsonl",
          "test_samefamily": "data/semantic_skeleton_v4/test_samefamily.jsonl",
          "test_deep": "data/semantic_skeleton_v4/test_deep.jsonl"}
SEED_ADP = "43"

def gen(model, tok, sys_p, user):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": sys_p},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    t0 = time.time()
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=700, do_sample=False,
                             temperature=None, pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text, round(time.time() - t0, 2), int(out.shape[1] - ids["input_ids"].shape[1])

tok = AutoTokenizer.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto")

# subsample samefamily
import random as _rnd
all_recs = {}
for k, path in SPLITS.items():
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    if k == "test_samefamily" and len(rows) > 200:
        _rnd.Random(7).shuffle(rows)
        rows = rows[:200]
    all_recs[k] = rows

results = {"details": []}
for arm, adapter, sys_p in (
        ("M0", f"runs/phase8b/m0_s{SEED_ADP}/final", SYS_M0),
        ("M1", f"runs/phase8b/m1_s{SEED_ADP}/final", SYS_M1)):
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    per_split = {}
    for sp, recs in all_recs.items():
        for rec in recs:
            user = (f"Task:\n{rec['instruction']}\n\n"
                    f"Available capabilities:\n{caps_block(rec)}")
            text, lat, ntok = gen(model, tok, sys_p, user)
            taskir = text
            sk_text = None
            if arm == "M1":
                sk_text = text
                try:
                    sk = skeleton_from_dict(json.loads(
                        re.search(r"\{.*\}", text, re.S).group(0)))
                    reg = {t: ({"params": [{"name": x, "optional": False}
                                           for x in m.get("required", [])]}
                           if m.get("properties") else {"params": None})
                           for t, m in rec["capability_registry"].items()}
                    errs = hard_errors(validate_skeleton(sk, None, reg))
                    low = lower_skeleton(sk, None, reg, strict=True) if not errs else {"taskir": None}
                    taskir = low.get("taskir") or ""
                except Exception:
                    taskir = ""
            v3 = evaluate_v3(taskir, rec)
            v3.update({"arm": arm, "split": sp, "task_id": rec["task_id"],
                       "depth": ("1" if len(rec["target_skeleton"]["nodes"]) == 1
                                 else "2-3" if len(rec["target_skeleton"]["nodes"]) <= 3
                                 else "4-6" if len(rec["target_skeleton"]["nodes"]) <= 6
                                 else "7+"),
                       "latency_s": lat, "gen_tokens": ntok})
            v3["raw_output"] = text[:4000]
            v3["taskir_text"] = taskir[:4000]
            results["details"].append(v3)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{arm} done")

# summary + paired stats
def agg(rows):
    n = len(rows)
    if not n: return None
    def m(k):
        v = [x.get(k) for x in rows if x.get(k) is not None]
        return round(sum(v)/len(v), 4) if v else None
    return {"n": n,
            "valid%": round(100*sum(x["parse"] for x in rows)/n, 1),
            "exact_joint_v3%": round(100*sum(x["exact_joint_v3"] for x in rows)/n, 1),
            "plan_f1": m("plan_f1") if "plan_f1" in rows[0] else None,
            "identity_ok%": round(100*sum(x["identity_ok"] for x in rows)/n, 1),
            "order_ok%": round(100*sum(x["order_ok"] for x in rows)/n, 1),
            "args_ok%": round(100*sum(x["args_ok"] for x in rows)/n, 1),
            "dag_data_f1": m("dag_data_f1"),
            "dag_order_f1": m("dag_order_f1"),
            "missing_params": m("missing_params_total"),
            "wrong_value": m("wrong_value_total"),
            "extra_calls": m("extra_calls")}

by = {}
for r in results["details"]:
    by.setdefault((r["arm"], r["split"]), []).append(r)
    by.setdefault((r["arm"], r["split"]), [])  # ensure key exists

summary = {}
for sp in SPLITS:
    for g in ("ALL", "depth:4-6", "depth:7+"):
        gf = (lambda x: True) if g == "ALL" else (lambda x: x["depth"] in ("4-6",) if g=="depth:4-6" else lambda x: x["depth"]=="7+")
        for arm in ("M0", "M1"):
            rows = [x for x in results["details"] if x["arm"]==arm and x["split"]==sp and gf(x)]
            if rows:
                summary[f"{sp}|{g}|{arm}"] = agg(rows)
    # paired
    m0 = {x["task_id"]: x for x in results["details"] if x["arm"]=="M0" and x["split"]==sp}
    m1 = {x["task_id"]: x for x in results["details"] if x["arm"]=="M1" and x["split"]==sp}
    for key in ("exact_joint_v3", "args_ok"):
        deltas = []
        fixed = broken = 0
        for tid in m0:
            if tid not in m1: continue
            a = 1 if m1[tid][key] else 0
            b = 1 if m0[tid][key] else 0
            deltas.append(a - b)
            if a and not b: fixed += 1
            elif b and not a: broken += 1
        if deltas:
            import random as _r
            rng = _r.Random(13)
            boots = []
            for _ in range(10000):
                s = [deltas[rng.randrange(len(deltas))] for _ in deltas]
                boots.append(sum(s)/len(s))
            boots.sort()
            summary[f"{sp}|paired|{key}"] = {
                "n": len(deltas),
                "M0%": round(100*sum(1 for d in deltas if d<0)/len(deltas), 1),
                "M1%": round(100*sum(1 for d in 
