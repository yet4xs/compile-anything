"""Phase 3: Final evaluation comparing SFT vs EF-SC vs V-DPO with official BFCL AST."""
import json, os, sys, re, random, gc
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
    for cat, c in stats.items():
        tn += c["n"]; tc += c["correct"]
    return {"label": label, "n": tn, "correct": tc,
            "accuracy": round(tc / max(1, tn), 4)}


def run_inference(model, tok, data, batch=8):
    prompts = []
    for s in data:
        user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_FC},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))
    outputs = []
    for bs in range(0, len(prompts), batch):
        if bs % 80 == 0:
            print(f"    {bs}/{len(prompts)}", flush=True)
        chunk = prompts[bs:bs+batch]
        inputs = tok(chunk, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        outputs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True) for o in out)
    return outputs


def main():
    import torch
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
    print(f"Held-out test: {len(test)}")

    # Add ground truth
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

    all_results = []

    # Model 1: Original SFT (7B MAX)
    print("\n[Evaluating: SFT 7B MAX]")
    m1 = PeftModel.from_pretrained(base, "runs/bfcl_track/fc_7b_max_s42/final")
    m1.eval()
    outputs1 = run_inference(m1, tok, test)
    r1 = evaluate_with_ast(test, outputs1, "SFT_7B_MAX")
    print(f"  AST accuracy: {r1['accuracy']:.2%}")
    all_results.append(r1)
    del m1; gc.collect(); torch.cuda.empty_cache()

    # Model 2: V-DPO (if available)
    vdpo_path = "runs/bfcl_track/vdpo_7b/final"
    if os.path.exists(vdpo_path):
        print("\n[Evaluating: V-DPO 7B]")
        m2 = PeftModel.from_pretrained(base, vdpo_path)
        m2.eval()
        outputs2 = run_inference(m2, tok, test)
        r2 = evaluate_with_ast(test, outputs2, "V-DPO_7B")
        print(f"  AST accuracy: {r2['accuracy']:.2%}")
        all_results.append(r2)
        del m2; gc.collect(); torch.cuda.empty_cache()
    else:
        print("\n[V-DPO adapter not found, skipping]")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL COMPARISON (Official BFCL AST)")
    print("=" * 60)
    print(f"\n{'Model':<20s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>8s}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['label']:<20s} {r['accuracy']:>9.2%} {r['correct']:>10d} {r['n']:>8d}")
    print(f"\n{'Baseline (no fix)':<20s} {'47.84%':>10s}")
    print(f"{'xLAM-7B (paper)':<20s} {'~85%':>10s}")

    os.makedirs("results/bfcl_track", exist_ok=True)
    with open("results/bfcl_track/final_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("FINAL-EVAL-DONE")


if __name__ == "__main__":
    main()
