"""Dual evaluation: held-out 20% + full training set, for the max-trained 7B model."""
import json, os, sys, re, random, gc
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
    if "no function" in text.lower(): return []
    calls = []
    for m in re.finditer(r"(\w+)\(([^)]*)\)", text):
        params = {}
        ps = m.group(2).strip()
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
                    params[k.strip()] = v.strip().strip("'\"")
        calls.append({"name": m.group(1), "params": params})
    return calls

def match_call(pred, gold):
    if pred["name"] != gold.get("name", ""): return False
    for k, v in gold.get("params", {}).items():
        if k not in pred["params"]: return False
        pv = str(pred["params"][k]).strip("'\" ")
        gv = str(v).strip("'\" ")
        if gv and gv not in pv and pv not in gv: return False
    return True

def eval_set(model, tok, data, label):
    prompts = []
    for s in data:
        user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_FC},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))
    comps = []
    batch = 8
    for bs in range(0, len(prompts), batch):
        if bs % 80 == 0: print(f"  [{label}] {bs}/{len(prompts)}", flush=True)
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
    tn = tc = 0
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
    results = {}
    for cat, c in sorted(stats.items()):
        acc = c["correct"] / max(1, c["n"])
        results[cat] = {"n": c["n"], "correct": c["correct"], "accuracy": round(acc, 4)}
        tn += c["n"]; tc += c["correct"]
    results["OVERALL"] = {"n": tn, "correct": tc, "accuracy": round(tc / max(1, tn), 4)}
    return results

def main():
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

    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-7B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, "runs/bfcl_track/fc_7b_max_s42/final")
    model.eval()

    print(f"\n=== HELD-OUT (n={len(test)}) ===")
    heldout = eval_set(model, tok, test, "heldout")
    for cat, r in sorted(heldout.items()):
        print(f"  {cat:30s} {r['correct']:4d}/{r['n']:4d} = {r['accuracy']:6.2%}")
    print(f"  {'OVERALL':30s} {heldout['OVERALL']['correct']:4d}/{heldout['OVERALL']['n']:4d} "
          f"= {heldout['OVERALL']['accuracy']:6.2%}")

    print(f"\n=== FULL TRAINING SET (n={len(data)}) ===")
    trainset = eval_set(model, tok, data, "train")
    print(f"  OVERALL: {trainset['OVERALL']['correct']}/{trainset['OVERALL']['n']} "
          f"= {trainset['OVERALL']['accuracy']:.2%}")

    os.makedirs("results/bfcl_track", exist_ok=True)
    with open("results/bfcl_track/max_eval.json", "w") as f:
        json.dump({"heldout": heldout, "train_full": trainset}, f, indent=2)
    print("BFCL-MAX-EVAL-DONE")


if __name__ == "__main__":
    main()
