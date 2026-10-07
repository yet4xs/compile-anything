"""BFCL held-out evaluation: train/test split, honest generalization scores.

Splits BFCL V4 data 80/20 by category-stratified random. Evaluates all
available adapters (3B v1, 3B v2, 7B, zero-shot baseline) on the held-out 20%.

Usage:
  python scripts/eval_bfcl_heldout.py                    # all available
  python scripts/eval_bfcl_heldout.py --adapter <path>   # specific adapter
"""
import json, os, sys, re, random, gc, argparse
from collections import Counter

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_FC = (
    "You are a function-calling assistant. Given a task and available function "
    "schemas, output the correct function call(s). If no function is relevant, "
    "say 'No function should be called.' Otherwise output function call(s) in "
    "the format: function_name(param1=value1, param2=value2)"
)


def parse_calls(text):
    if "no function" in text.lower():
        return []
    calls = []
    for m in re.finditer(r"(\w+)\(([^)]*)\)", text):
        name = m.group(1)
        param_str = m.group(2).strip()
        params = {}
        if param_str:
            depth = 0; parts = []; cur = ""
            for ch in param_str:
                if ch in "([{": depth += 1
                elif ch in ")]}": depth -= 1
                elif ch == "," and depth == 0:
                    parts.append(cur); cur = ""; continue
                cur += ch
            if cur.strip(): parts.append(cur)
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    params[k.strip()] = v.strip().strip("'\"")
        calls.append({"name": name, "params": params})
    return calls


def match_call(pred, gold):
    if pred["name"] != gold.get("name", ""):
        return False
    for k, v in gold.get("params", {}).items():
        if k not in pred["params"]: return False
        pv = str(pred["params"][k]).strip("'\" ")
        gv = str(v).strip("'\" ")
        if gv and gv not in pv and pv not in gv: return False
    return True


def evaluate_model(tok, base, adapter, data, label, model_size="3b"):
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    if adapter: model.eval()

    prompts = []
    for s in data:
        user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_FC},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

    batch = 16 if model_size == "3b" else 8
    comps = []
    for bs in range(0, len(prompts), batch):
        if bs % 160 == 0:
            print(f"  [{label}] {bs}/{len(prompts)}", flush=True)
        chunk = prompts[bs:bs+batch]
        inputs = tok(chunk, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        comps.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True) for o in out)

    stats = {}
    for s, comp in zip(data, comps):
        cat = s["category"]
        if cat not in stats: stats[cat] = Counter()
        stats[cat]["n"] += 1
        pred_calls = parse_calls(comp)
        gt = s.get("target", "")
        if "irrelevance" in cat or "No function" in gt:
            if not pred_calls: stats[cat]["correct"] += 1
        else:
            gt_calls = parse_calls(gt)
            if len(pred_calls) == len(gt_calls):
                ok = all(any(match_call(p, g) for g in gt_calls) for p in pred_calls) if gt_calls else not pred_calls
                if ok: stats[cat]["correct"] += 1

    if adapter: del model; gc.collect(); torch.cuda.empty_cache()

    results = {}
    tn = tc = 0
    for cat, c in sorted(stats.items()):
        acc = c["correct"] / max(1, c["n"])
        results[cat] = {"n": c["n"], "correct": c["correct"], "accuracy": round(acc, 4)}
        tn += c["n"]; tc += c["correct"]
    results["OVERALL"] = {"n": tn, "correct": tc, "accuracy": round(tc / max(1, tn), 4)}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    random.seed(999)

    # Load all BFCL data
    data = [json.loads(l) for l in open("data/bfcl_training/bfcl_train.jsonl",
                                         encoding="utf-8") if l.strip()]
    # Stratified 80/20 split
    by_cat = {}
    for s in data:
        by_cat.setdefault(s["category"], []).append(s)
    test = []
    for cat, items in by_cat.items():
        random.shuffle(items)
        n_test = max(1, int(len(items) * 0.2))
        test.extend(items[:n_test])
    random.shuffle(test)
    print(f"Held-out test: {len(test)} / {len(data)}", flush=True)

    tok3 = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok3.padding_side = "left"
    if tok3.pad_token is None: tok3.pad_token = tok3.eos_token

    base3 = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    all_results = {"_split": {"test_n": len(test), "total": len(data)}}

    # Zero-shot baseline
    print("\n[zero-shot 3B]", flush=True)
    all_results["zeroshot_3b"] = evaluate_model(tok3, base3, None, test, "zs")
    print(f"  OVERALL: {all_results['zeroshot_3b']['OVERALL']['accuracy']:.2%}")

    del base3; gc.collect(); torch.cuda.empty_cache()

    # Available adapters
    adapters = [
        ("fc_3b_v1", "runs/bfcl_track/fc_3b_s42/final", "3b"),
        ("fc_3b_v2", "runs/bfcl_track/fc_3b_v2_s42/final", "3b"),
        ("fc_7b", "runs/bfcl_track/fc_7b_s42/final", "7b"),
    ]
    if args.adapter:
        adapters = [("custom", args.adapter, "3b")]

    for name, path, size in adapters:
        if not os.path.exists(path):
            print(f"\n[skip {name}: not found]", flush=True)
            continue
        print(f"\n[{name}]", flush=True)
        if size == "7b":
            tok7 = AutoTokenizer.from_pretrained("weights/Qwen2.5-7B-Instruct", trust_remote_code=True)
            tok7.padding_side = "left"
            if tok7.pad_token is None: tok7.pad_token = tok7.eos_token
            base7 = AutoModelForCausalLM.from_pretrained(
                "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto")
            all_results[name] = evaluate_model(tok7, base7, path, name, "7b")
            del base7; gc.collect(); torch.cuda.empty_cache()
        else:
            base3 = AutoModelForCausalLM.from_pretrained(
                "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto")
            all_results[name] = evaluate_model(tok3, base3, path, name, "3b")
            del base3; gc.collect(); torch.cuda.empty_cache()
        print(f"  OVERALL: {all_results[name]['OVERALL']['accuracy']:.2%}")

    os.makedirs("results/bfcl_track", exist_ok=True)
    with open("results/bfcl_track/heldout_eval.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== HELD-OUT SUMMARY ===")
    for k, v in all_results.items():
        if k.startswith("_"): continue
        print(f"  {k:20s} {v['OVERALL']['accuracy']:6.2%}")
    print("HELDOUT-DONE")


if __name__ == "__main__":
    main()
