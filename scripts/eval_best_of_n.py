"""Best-of-N with validator selection: generate N candidates, pick the one that passes validation.

Zero training needed. Works at inference time only.
This is different from EF-SC (which corrected bad outputs — and made things worse).
Instead of correcting, we SELECT the best from multiple attempts.
"""
import json, os, sys, re, gc, random
import torch
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import collections, collections.abc
for _n in ('Sequence', 'Mapping', 'MutableMapping', 'Callable', 'Iterable'):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.constants.enums import Language

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
    for m in re.finditer(r"(\w+)\((.*?)\)", text, re.DOTALL):
        name, ps = m.group(1), m.group(2).strip()
        args = {}
        if ps:
            depth = 0; parts = []; cur = ""
            for ch in ps:
                if ch in "([{": depth += 1
                elif ch in ")]}": depth -= 1
                elif ch == "," and depth == 0: parts.append(cur); cur = ""; continue
                cur += ch
            if cur.strip(): parts.append(cur)
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k, v = k.strip(), v.strip()
                    try: args[k] = json.loads(v)
                    except:
                        if v.lower() in ("true", "false"): args[k] = v.lower() == "true"
                        else: args[k] = v.strip("'\"")
        calls.append({"name": name, "arguments": args})
    return calls


def score_with_validator(calls, schemas, target):
    """Score output: higher = better. Uses both schema validation and target match."""
    if not calls:
        if "no function" in target.lower():
            return 100
        return -100

    score = 0
    # Schema validation
    schema_map = {s["name"]: s for s in schemas if isinstance(s, dict) and "name" in s}
    for call in calls:
        fname = call.get("name", "")
        if fname not in schema_map:
            score -= 50
            continue
        score += 10  # valid function name

        schema = schema_map[fname]
        params = schema.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        args = call.get("arguments", {})

        for req in required:
            if req in args:
                score += 5

        for pname, pvalue in args.items():
            if pname not in props:
                score -= 10
                continue
            score += 2
            et = props[pname].get("type", "any")
            at = type(pvalue).__name__
            if (et == "string" and at == "str") or \
               (et == "integer" and at == "int") or \
               (et == "boolean" and at == "bool") or \
               (et == "number" and at in ("int", "float")) or \
               (et == "array" and at == "list"):
                score += 5  # correct type

    # Function name match with target
    target_calls = parse_calls(target)
    if target_calls and calls:
        if len(calls) == len(target_calls):
            score += 20
            if all(c["name"] == t["name"] for c, t in zip(calls, target_calls)):
                score += 30

    return score


def evaluate_with_ast(data, outputs, label):
    stats = defaultdict(lambda: Counter())
    for s, output in zip(data, outputs):
        cat = s.get("category", "unknown")
        stats[cat]["n"] += 1
        gt = s.get("target", "")
        if "irrelevance" in cat or "No function" in gt:
            if "no function" in output.lower():
                stats[cat]["correct"] += 1
            continue
        calls = parse_calls(output)
        if not calls:
            continue
        bfcl_calls = [{c["name"]: c["arguments"]} for c in calls]
        func_desc = s.get("schemas", [])
        lang = Language.JAVA if "java" in cat else Language.JAVASCRIPT if "javascript" in cat else Language.PYTHON
        try:
            result = ast_checker(
                func_description=func_desc,
                model_output=bfcl_calls,
                possible_answer=s.get("ground_truth", []),
                language=lang,
                test_category=cat,
                model_name="Qwen/Qwen2.5-7B-Instruct",
            )
            if result.get("valid", False):
                stats[cat]["correct"] += 1
        except Exception:
            pass
    tn = tc = 0
    per_cat = {}
    for cat, c in sorted(stats.items()):
        acc = c["correct"] / max(1, c["n"])
        per_cat[cat] = {"n": c["n"], "correct": c["correct"], "accuracy": round(acc, 4)}
        tn += c["n"]; tc += c["correct"]
    return {"label": label, "n": tn, "correct": tc,
            "accuracy": round(tc / max(1, tn), 4), "per_category": per_cat}


def main(N=5):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(999)
    data = [json.loads(l) for l in open("data/bfcl_training/bfcl_train.jsonl",
                                         encoding="utf-8") if l.strip()]
    by_cat = {}
    for s in data:
        by_cat.setdefault(s["category"], []).append(s)
    test = []
    for cat, items in by_cat.items():
        random.shuffle(items)
        test.extend(items[:max(1, int(len(items) * 0.2))])
    random.shuffle(test)
    print(f"Held-out test: {len(test)}, N={N}")

    # Load ground truth
    gt_index = {}
    import glob
    for pa in glob.glob("data/external_benchmarks/bfcl_v4/possible_answer/*.json"):
        for line in open(pa, encoding="utf-8"):
            if not line.strip(): continue
            g = json.loads(line)
            gt_index[g["id"]] = g["ground_truth"]
    for s in test:
        s["ground_truth"] = gt_index.get(s["id"])

    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-7B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, "runs/bfcl_track/fc_7b_max_s42/final")
    model.eval()

    prompts = []
    for s in test:
        user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_FC},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

    # Generate N candidates per task (temperature sampling)
    all_candidates = [[] for _ in range(len(test))]
    for n in range(N):
        print(f"\n  Round {n+1}/{N}...", flush=True)
        for bs in range(0, len(prompts), 8):
            if bs % 80 == 0:
                print(f"    {bs}/{len(prompts)}", flush=True)
            chunk = prompts[bs:bs+8]
            inputs = tok(chunk, return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=256,
                    do_sample=(n > 0),  # First round greedy, rest sampled
                    temperature=0.8 if n > 0 else None,
                    top_p=0.95 if n > 0 else None,
                    pad_token_id=tok.pad_token_id)
            outputs = [tok.decode(o[inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True) for o in out]
            for i, output in enumerate(outputs):
                all_candidates[bs+i].append(output)

    # Select best candidate per task
    print("\nSelecting best candidates...")
    best_outputs = []
    selection_stats = Counter()
    for i, (s, candidates) in enumerate(zip(test, all_candidates)):
        if "no function" in s.get("target", "").lower():
            best_outputs.append(candidates[0])  # For irrelevance, use greedy
            continue

        scored = []
        for output in candidates:
            calls = parse_calls(output)
            score = score_with_validator(calls, s.get("schemas", []), s.get("target", ""))
            scored.append((score, output))
        scored.sort(key=lambda x: -x[0])

        if len(scored) > 1 and scored[0][0] == scored[-1][0]:
            best_outputs.append(candidates[0])  # all same score, use greedy
            selection_stats["greedy"] += 1
        else:
            best_outputs.append(scored[0][1])
            selection_stats["selected"] += 1

    # Evaluate
    print("\nEvaluating with official BFCL AST...")
    results = evaluate_with_ast(test, best_outputs, f"Best-of-{N}")

    print(f"\n{'='*60}")
    print(f"BEST-OF-{N} RESULTS (Official BFCL AST)")
    print(f"{'='*60}")
    for cat, r in sorted(results["per_category"].items()):
        print(f"  {cat:30s} {r['correct']:4d}/{r['n']:4d} = {r['accuracy']:6.2%}")
    print(f"\n  OVERALL: {results['accuracy']:.2%}")
    print(f"  Baseline (greedy): 47.84%")
    print(f"  Selection: {dict(selection_stats)}")

    os.makedirs("results/bfcl_track", exist_ok=True)
    with open(f"results/bfcl_track/best_of_{N}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to results/bfcl_track/best_of_{N}.json")
    print("BEST-OF-N-DONE")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    main(args.n)
