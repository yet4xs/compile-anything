"""Evaluate BFCL Track A model: AST-style function-call accuracy.

Simplified BFCL evaluator: for each test sample, generate function call(s),
parse them, compare against ground truth by function name + parameter overlap.
This approximates the official AST evaluation (not exact — for leaderboard
submission, use the official gorilla eval harness).
"""
import json, os, sys, re, gc
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
    """Parse generated function calls."""
    if "no function" in text.lower():
        return []
    # Match function_name(param=value, ...)
    calls = []
    for m in re.finditer(r"(\w+)\(([^)]*)\)", text):
        name = m.group(1)
        param_str = m.group(2).strip()
        params = {}
        if param_str:
            # Split by comma but respect nested quotes
            depth = 0
            parts = []
            cur = ""
            for ch in param_str:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch == "," and depth == 0:
                    parts.append(cur)
                    cur = ""
                    continue
                cur += ch
            if cur.strip():
                parts.append(cur)
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    params[k] = v
        calls.append({"name": name, "params": params})
    return calls


def match_call(pred, gold):
    """Check if a predicted call matches a gold call."""
    if pred["name"] != gold.get("name", gold.get("__name__", "")):
        return False
    gold_params = gold.get("params", {})
    if isinstance(gold_params, dict):
        for k, v in gold_params.items():
            if k not in pred["params"]:
                return False
            # Loose value match (string contains)
            pv = str(pred["params"][k]).strip("'\" ")
            gv = str(v).strip("'\" ")
            if gv and gv not in pv and pv not in gv:
                return False
    return True


def eval_category(model_name, adapter, data, tok):
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    if adapter:
        model.eval()

    by_cat = {}
    prompts = []
    metas = []
    for s in data:
        user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_FC},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))
        metas.append(s)

    comps = []
    for bs in range(0, len(prompts), 16):
        if bs % 320 == 0:
            print(f"  {bs}/{len(prompts)}", flush=True)
        batch = prompts[bs:bs+16]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        comps.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True) for o in out)

    stats = defaultdict_stats = {}
    for s, comp in zip(metas, comps):
        cat = s["category"]
        if cat not in stats:
            stats[cat] = Counter()
        stats[cat]["n"] += 1

        pred_calls = parse_calls(comp)
        gt = s.get("target", "")

        if "irrelevance" in cat or "No function" in gt:
            # Should refuse
            if not pred_calls:
                stats[cat]["correct"] += 1
        else:
            # Parse GT calls
            gt_calls = parse_calls(gt)
            if len(pred_calls) == len(gt_calls):
                all_match = all(
                    any(match_call(p, g) for g in gt_calls) for p in pred_calls
                ) if gt_calls else not pred_calls
                if all_match:
                    stats[cat]["correct"] += 1

    if adapter:
        del model
    del base
    gc.collect()
    torch.cuda.empty_cache()

    results = {}
    total_n = total_c = 0
    for cat, c in sorted(stats.items()):
        acc = c["correct"] / max(1, c["n"])
        results[cat] = {"n": c["n"], "correct": c["correct"],
                         "accuracy": round(acc, 4)}
        total_n += c["n"]
        total_c += c["correct"]
    results["OVERALL"] = {"n": total_n, "correct": total_c,
                          "accuracy": round(total_c / max(1, total_n), 4)}
    return results


def main():
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Eval on BFCL test data (the same data we trained on — this is a sanity
    # check for learnability; leaderboard submission needs the official harness)
    data = [json.loads(l) for l in open("data/bfcl_training/bfcl_train.jsonl",
                                        encoding="utf-8") if l.strip()]
    print(f"Eval samples: {len(data)}")

    adapter = "runs/bfcl_track/fc_3b_s42/final"
    if not os.path.exists(adapter):
        print(f"adapter not found: {adapter}")
        return
    results = eval_category("fc_3b", adapter, data, tok)
    print("\n=== BFCL Track A Results (train-set sanity) ===")
    for cat, r in sorted(results.items()):
        print(f"  {cat:30s} {r['correct']:5d}/{r['n']:5d} = {r['accuracy']:6.2%}")
    print(f"\n{'OVERALL':30s} {results['OVERALL']['correct']:5d}/{results['OVERALL']['n']:5d} "
          f"= {results['OVERALL']['accuracy']:6.2%}")

    os.makedirs("results/bfcl_track", exist_ok=True)
    with open("results/bfcl_track/train_sanity.json", "w") as f:
        json.dump(results, f, indent=2)
    print("BFCL-TRACK-EVAL-DONE")


if __name__ == "__main__":
    main()
