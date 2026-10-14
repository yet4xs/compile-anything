"""Minimal eval-only script — avoids KeyError by wrapping in try/except."""
import json, os, sys, time, re, gc, random, hashlib
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.ir.parser import parse_text
from src.ir.taskir import to_text, module_from_dict
from src.validator.validator import validate

DATA = "data/runtime_dataflow_v2"
RUNS = "runs/phase8d"
OUTR = "results/phase8d"

# Import evaluation components
from scripts.train_three_arm import (
    SYS_M0T, SYS_M0J, SYS_M1, build_user, gen_and_eval)

def main():
    test = [json.loads(l) for l in open(f"{DATA}/test.jsonl", encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    SEEDS = [44, 45, 46]
    all_results = []
    for seed in SEEDS:
        for arm, label in (("m0t", "M0-T"), ("m0j", "M0-J"), ("m1", "M1")):
            adapter = f"{RUNS}/{arm}_s{seed}/final"
            if not os.path.exists(adapter):
                print(f"  skip {label}/s{seed}")
                continue
            model = PeftModel.from_pretrained(base, adapter)
            model.eval()
            for protocol in ("oracle", "distractor"):
                for rec in test:
                    try:
                        ev = gen_and_eval(model, tok, rec, label, protocol)
                    except Exception as e:
                        ev = {"task_id": rec["task_id"], "arm": label,
                              "protocol": protocol, "seed": seed,
                              "parse": False, "validator": False,
                              "binding_exact": False, "binding_f1": 0,
                              "tool_match": False, "execution": False,
                              "strict_semantic": False,
                              "topology": rec.get("topology", "?"),
                              "latency_s": 0, "gen_tokens": 0,
                              "raw": str(e)[:200]}
                    ev["seed"] = seed
                    all_results.append(ev)
            print(f"  {label}/s{seed} done", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    by = defaultdict(list)
    for r in all_results:
        by[(r["arm"], r["protocol"], "ALL")].append(r)

    summary = {}
    for (arm, proto, _), rows in sorted(by.items()):
        n = len(rows)
        def pct(k):
            return round(100 * sum(1 for x in rows if x.get(k)) / n, 1)
        summary[f"{proto}|{arm}"] = {
            "n": n, "binding_exact%": pct("binding_exact"),
            "validator%": pct("validator"),
            "strict_semantic%": pct("strict_semantic")}

    # Paired
    for proto in ("oracle", "distractor"):
        for a_arm, b_arm in (("M1", "M0-J"), ("M0-J", "M0-T"), ("M1", "M0-T")):
            deltas = []
            fx = bk = 0
            for seed in SEEDS:
                a = {x["task_id"]: x for x in all_results
                     if x["arm"] == a_arm and x["protocol"] == proto and x["seed"] == seed}
                b = {x["task_id"]: x for x in all_results
                     if x["arm"] == b_arm and x["protocol"] == proto and x["seed"] == seed}
                for tid in a:
                    if tid not in b: continue
                    av = 1 if a[tid].get("binding_exact") else 0
                    bv = 1 if b[tid].get("binding_exact") else 0
                    deltas.append(av - bv)
                    if av and not bv: fx += 1
                    elif bv and not av: bk += 1
            if deltas:
                rng = random.Random(13)
                boots = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(5000))
                summary[f"{proto}|paired|{a_arm}-vs-{b_arm}"] = {
                    "n": len(deltas),
                    "diff_pp": round(100 * sum(deltas) / len(deltas), 1),
                    "ci95": [round(100 * boots[125], 1), round(100 * boots[4875], 1)],
                    "fixed": fx, "broken": bk}

    os.makedirs(OUTR, exist_ok=True)
    with open(f"{OUTR}/three_arm_ablation.json", "w") as f:
        json.dump({"summary": summary}, f, indent=2, ensure_ascii=False)
    with open(f"{OUTR}/three_arm_raw_predictions.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    for k in sorted(summary):
        v = summary[k]
        if "paired" in k:
            print(f"  {k}: diff={v['diff_pp']}pp CI={v['ci95']} f/b={v['fixed']}/{v['broken']}")
        else:
            print(f"  {k}: binding={v['binding_exact%']}% strict={v['strict_semantic%']}%")
    print("THREE-ARM-DONE")

if __name__ == "__main__":
    main()
