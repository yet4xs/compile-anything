"""Phase 9B-2c: Evaluate M0-T/M0-J/M1 on real τ³ L3 benchmark.

Run on server after booting. Uses existing Phase 8D checkpoints
(retrained on runtime_dataflow_v2, evaluated on tau3_runtime_benchmark).

Usage: nohup python scripts/eval_tau3_real.py > tau3_eval.log 2>&1 &
"""
import json, os, sys, time, re, gc, random, hashlib, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.ir.parser import parse_text
from src.ir.taskir import to_text, module_from_dict
from src.validator.validator import validate
from src.compiler.plan_skeleton import skeleton_from_dict, validate_skeleton, hard_errors
from src.compiler.skeleton_lowering import lower_skeleton
from src.runtime.mock_capability_registry import (
    MockCapabilityRegistry, MockCapability, DataflowRuntime)

DATA = "data/tau3_runtime_benchmark"
RUNS = "runs/phase8d"
OUTR = "results/phase9"

# Reuse prompts from three-arm experiment
from scripts.train_three_arm import SYS_M0T, SYS_M0J, SYS_M1, gen_and_eval


def main(seed=44):
    test = [json.loads(l) for l in open(f"{DATA}/test.jsonl", encoding="utf-8") if l.strip()]
    dev = [json.loads(l) for l in open(f"{DATA}/dev.jsonl", encoding="utf-8") if l.strip()]
    eval_tasks = test + dev  # use both (small dataset)

    print(f"Evaluating {len(eval_tasks)} real τ³ tasks")

    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    all_results = []

    # Oracle baseline (compiler upper bound)
    print("Running Oracle baseline...")
    oracle_pass = 0
    for rec in eval_tasks:
        sk = skeleton_from_dict(rec["target_skeleton"])
        schemas = {}
        for tid, meta in rec["capability_registry"].items():
            schemas[tid] = {
                "params": [{"name": k, "optional": False, "type": v}
                            for k, v in meta.get("input_schema", {}).items()],
                "output_fields": meta.get("output_schema", {}),
                "effect_class": meta.get("effect_class", "read_only")}
        low = lower_skeleton(sk, None, schemas, strict=True)
        taskir = low.get("taskir") or ""
        v2 = {t: {"input_schema": rec["capability_registry"][t].get("input_schema", {}),
                   "output_schema": rec["capability_registry"][t].get("output_schema", {})}
              for t in rec["capability_registry"]}
        try:
            mod = parse_text(taskir)
            valid = validate(mod, capability_registry=v2).valid
        except Exception:
            valid = False
        if valid:
            oracle_pass += 1
    print(f"  Oracle: {oracle_pass}/{len(eval_tasks)}")

    # Evaluate three arms
    for arm, adapter in (("M0-T", f"{RUNS}/m0t_s{seed}/final"),
                          ("M0-J", f"{RUNS}/m0j_s{seed}/final"),
                          ("M1", f"{RUNS}/m1_s{seed}/final")):
        if not os.path.exists(adapter):
            print(f"  skip {arm} (adapter not found)")
            continue
        print(f"Running {arm}...", flush=True)
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()

        for rec in eval_tasks:
            try:
                ev = gen_and_eval(model, tok, rec, arm, "oracle")
            except Exception as e:
                ev = {"task_id": rec["task_id"], "arm": arm,
                      "protocol": "oracle", "seed": seed,
                      "parse": False, "validator": False,
                      "binding_exact": False, "binding_f1": 0,
                      "tool_match": False, "execution": False,
                      "strict_semantic": False,
                      "topology": rec.get("topology", "?"),
                      "latency_s": 0, "gen_tokens": 0,
                      "raw": str(e)[:200]}
            ev["seed"] = seed
            all_results.append(ev)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    by = defaultdict(list)
    for r in all_results:
        by[r["arm"]].append(r)

    summary = {"oracle_baseline": f"{oracle_pass}/{len(eval_tasks)}"}
    for arm, rows in sorted(by.items()):
        n = len(rows)
        def pct(k):
            return round(100 * sum(1 for x in rows if x.get(k)) / n, 1)
        summary[arm] = {
            "n": n, "binding_exact%": pct("binding_exact"),
            "validator%": pct("validator"),
            "strict_semantic%": pct("strict_semantic")}

    # Paired M1 vs M0-T and M1 vs M0-J
    m0t = {x["task_id"]: x for x in all_results if x["arm"] == "M0-T"}
    m0j = {x["task_id"]: x for x in all_results if x["arm"] == "M0-J"}
    m1 = {x["task_id"]: x for x in all_results if x["arm"] == "M1"}

    for name, baseline in (("M1_vs_M0-T", m0t), ("M1_vs_M0-J", m0j)):
        fx = bk = 0
        deltas = []
        for tid in m1:
            if tid not in baseline:
                continue
            a = 1 if m1[tid].get("binding_exact") else 0
            b = 1 if baseline[tid].get("binding_exact") else 0
            deltas.append(a - b)
            if a and not b:
                fx += 1
            elif b and not a:
                bk += 1
        if deltas:
            rng = random.Random(13)
            boots = sorted(
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
                for _ in range(5000))
            summary[f"paired|{name}"] = {
                "n": len(deltas),
                "diff_pp": round(100 * sum(deltas) / len(deltas), 1),
                "ci95": [round(100 * boots[125], 1),
                          round(100 * boots[4875], 1)],
                "fixed": fx, "broken": bk}

    os.makedirs(OUTR, exist_ok=True)
    with open(f"{OUTR}/tau3_real_evaluation.json", "w") as f:
        json.dump({"summary": summary, "details": all_results}, f, indent=2,
                  ensure_ascii=False)

    print(f"\n{'='*50}")
    print("REAL τ³ BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Oracle baseline: {summary['oracle_baseline']}")
    for k in sorted(summary):
        if "paired" in k:
            v = summary[k]
            print(f"  {k}: diff={v['diff_pp']}pp CI={v['ci95']} f/b={v['fixed']}/{v['broken']}")
        elif k != "oracle_baseline":
            v = summary[k]
            print(f"  {k}: binding={v.get('binding_exact%', '?')}% "
                  f"strict={v.get('strict_semantic%', '?')}% "
                  f"val={v.get('validator%', '?')}%")
    print("TAU3-REAL-EVAL-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()
    main(args.seed)
