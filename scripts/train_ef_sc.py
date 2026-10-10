"""Phase 1: EF-SC (Error-Feedback Self-Correction) for BFCL AST improvement.

Protocol:
1. Generate function call with our 7B MAX model
2. Parse + validate against BFCL function schema
3. If validation fails → feed error info back → regenerate
4. Up to 3 correction rounds
5. Evaluate with official BFCL AST checker

The "validator" here is a lightweight schema-aware type checker that:
- Checks function name against available schemas
- Checks parameter names against schema properties
- Checks parameter types against schema types (string/integer/boolean/etc)
- Checks required parameters are present
- Reports specific errors for feedback
"""
import json, os, sys, re, gc, time, random
import torch
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# BFCL AST checker
import collections
import collections.abc
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

MAX_RETRIES = 3


def validate_against_schema(calls, schemas):
    """Lightweight schema-aware validator. Returns list of error strings."""
    errors = []
    schema_map = {s["name"]: s for s in schemas if isinstance(s, dict) and "name" in s}

    for call in calls:
        fname = call.get("name", "")
        if fname not in schema_map:
            errors.append(f"Unknown function '{fname}'. Available: {list(schema_map.keys())[:5]}")
            continue

        schema = schema_map[fname]
        params = schema.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        args = call.get("arguments", {})

        # Check required parameters
        for req in required:
            if req not in args:
                errors.append(f"Missing required parameter '{req}' for {fname}")

        # Check parameter types
        for pname, pvalue in args.items():
            if pname not in props:
                errors.append(f"Unexpected parameter '{pname}' for {fname}")
                continue

            expected_type = props[pname].get("type", "")
            actual_type = type(pvalue).__name__

            type_ok = {
                "string": actual_type == "str",
                "integer": actual_type == "int",
                "number": actual_type in ("int", "float"),
                "boolean": actual_type == "bool",
                "array": actual_type == "list",
                "object": actual_type == "dict",
                "any": True,
            }.get(expected_type, True)

            if not type_ok:
                errors.append(
                    f"Parameter '{pname}' should be {expected_type}, "
                    f"got {actual_type}. Value: {str(pvalue)[:50]}")

    return errors


def parse_calls(text):
    if "no function" in text.lower():
        return []
    calls = []
    for m in re.finditer(r"(\w+)\((.*?)\)", text, re.DOTALL):
        name = m.group(1)
        param_str = m.group(2).strip()
        args = {}
        if param_str:
            depth = 0; parts = []; cur = ""
            for ch in param_str:
                if ch in "([{": depth += 1
                elif ch in ")]}": depth -= 1
                elif ch == "," and depth == 0: parts.append(cur); cur = ""; continue
                cur += ch
            if cur.strip(): parts.append(cur)
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k, v = k.strip(), v.strip()
                    try:
                        args[k] = json.loads(v)
                    except:
                        if v.lower() in ("true", "false"):
                            args[k] = v.lower() == "true"
                        else:
                            args[k] = v.strip("'\"")
        calls.append({"name": name, "arguments": args})
    return calls


def format_feedback_prompt(instruction, schemas, previous_output, errors):
    """Build correction prompt with validator feedback."""
    schema_lines = []
    for i, s in enumerate(schemas[:15], 1):
        if not isinstance(s, dict) or "name" not in s:
            continue
        params = s.get("parameters", {})
        props = params.get("properties", {})
        req = params.get("required", [])
        pstr = ", ".join(f"{k}:{v.get('type','?')}" for k, v in props.items() if isinstance(v, dict))
        schema_lines.append(f"[{i}] {s['name']}({pstr}) - {s.get('description','')[:120]}")

    error_str = "\n".join(f"  ERROR: {e}" for e in errors)

    return (
        f"{instruction}\n\nAvailable functions:\n" + "\n".join(schema_lines) +
        f"\n\nYour previous answer was:\n{previous_output}\n\n"
        f"But it has these validation errors:\n{error_str}\n\n"
        f"Please fix these errors and output the correct function call(s). "
        f"Make sure parameter types match the schema exactly "
        f"(string='value', integer=123, boolean=true/false)."
    )


def run_ef_sc(model, tok, data, batch=8):
    """Run with Error-Feedback Self-Correction."""
    results = []
    prompts_initial = []
    for s in data:
        user = f"{s['instruction']}\n\nAvailable functions:\n{s['schema_prompt']}"
        prompts_initial.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_FC},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

    # Round 1: Initial generation
    outputs = []
    for bs in range(0, len(prompts_initial), batch):
        if bs % 80 == 0:
            print(f"    initial {bs}/{len(prompts_initial)}", flush=True)
        chunk = prompts_initial[bs:bs+batch]
        inputs = tok(chunk, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        outputs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True) for o in out)

    # Rounds 2-3: Self-correction for failures
    for retry_round in range(1, MAX_RETRIES):
        retry_idx = []
        retry_prompts = []
        for i, (s, output) in enumerate(zip(data, outputs)):
            if "no function" in output.lower():
                continue
            calls = parse_calls(output)
            errors = validate_against_schema(calls, s.get("schemas", []))
            if errors:
                retry_idx.append(i)
                fb_prompt = format_feedback_prompt(
                    s["instruction"], s.get("schemas", []), output, errors)
                retry_prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_FC},
                     {"role": "user", "content": fb_prompt}],
                    tokenize=False, add_generation_prompt=True))

        if not retry_prompts:
            break

        print(f"    retry round {retry_round}: {len(retry_prompts)} items to fix", flush=True)
        retry_outputs = []
        for bs in range(0, len(retry_prompts), batch):
            chunk = retry_prompts[bs:bs+batch]
            inputs = tok(chunk, return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256,
                                     do_sample=False, temperature=None,
                                     pad_token_id=tok.pad_token_id)
            retry_outputs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                            skip_special_tokens=True) for o in out)

        for idx, new_output in zip(retry_idx, retry_outputs):
            outputs[idx] = new_output

    return outputs


def evaluate_with_ast(data, outputs):
    """Evaluate using official BFCL AST checker."""
    stats = defaultdict(lambda: Counter())
    corrections_needed = 0

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

        # Count how many needed correction
        errors = validate_against_schema(calls, s.get("schemas", []))
        if errors:
            corrections_needed += 1

        # Convert to BFCL format
        bfcl_calls = [{c["name"]: c["arguments"]} for c in calls]

        # Get function description
        func_desc = s.get("schemas", [])

        lang = Language.PYTHON
        if "java" in cat:
            lang = Language.JAVA
        elif "javascript" in cat:
            lang = Language.JAVASCRIPT

        try:
            result = ast_checker(
                func_description=func_desc,
                model_output=bfcl_calls,
                possible_answer=s.get("ground_truth", []),
                language=lang,
                test_category=cat,
                model_name="Qwen/Qwen2.5-7B-Instruct",  # registered name
            )
            if result.get("valid", False):
                stats[cat]["correct"] += 1
        except Exception:
            pass

    results = {}
    tn = tc = 0
    for cat, c in sorted(stats.items()):
        acc = c["correct"] / max(1, c["n"])
        results[cat] = {"n": c["n"], "correct": c["correct"], "accuracy": round(acc, 4)}
        tn += c["n"]; tc += c["correct"]
    results["OVERALL"] = {"n": tn, "correct": tc, "accuracy": round(tc / max(1, tn), 4)}
    results["_corrections_needed"] = corrections_needed
    return results


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    random.seed(999)

    # Load BFCL data
    data = [json.loads(l) for l in open("data/bfcl_training/bfcl_train.jsonl",
                                         encoding="utf-8") if l.strip()]
    # Stratified 20% held-out (same as before)
    by_cat = {}
    for s in data:
        by_cat.setdefault(s["category"], []).append(s)
    test = []
    for cat, items in by_cat.items():
        random.shuffle(items)
        test.extend(items[:max(1, int(len(items) * 0.2))])
    random.shuffle(test)
    print(f"Held-out test: {len(test)}")

    # Load model
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-7B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, "runs/bfcl_track/fc_7b_max_s42/final")
    model.eval()

    # Load ground truth for AST evaluation
    gt_index = {}
    import glob
    for pa in glob.glob("data/external_benchmarks/bfcl_v4/possible_answer/*.json"):
        for line in open(pa, encoding="utf-8"):
            if not line.strip():
                continue
            g = json.loads(line)
            gt_index[g["id"]] = g["ground_truth"]

    # Add ground truth to test data
    for s in test:
        s["ground_truth"] = gt_index.get(s["id"])

    # Run EF-SC
    print("\nRunning EF-SC (Error-Feedback Self-Correction)...")
    outputs = run_ef_sc(model, tok, test)

    # Evaluate
    print("\nEvaluating with official BFCL AST checker...")
    results = evaluate_with_ast(test, outputs)

    print("\n" + "=" * 60)
    print("EF-SC RESULTS (official BFCL AST)")
    print("=" * 60)
    for cat, r in sorted(results.items()):
        if cat.startswith("_"): continue
        print(f"  {cat:30s} {r['correct']:4d}/{r['n']:4d} = {r['accuracy']:6.2%}")
    print(f"\n  Corrections needed: {results.get('_corrections_needed', 0)}")
    print(f"\n  Previous AST score: 47.84%")
    print(f"  EF-SC AST score:    {results['OVERALL']['accuracy']:.2%}")

    os.makedirs("results/bfcl_track", exist_ok=True)
    with open("results/bfcl_track/ef_sc_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/bfcl_track/ef_sc_results.json")
    print("EF-SC-DONE")


if __name__ == "__main__":
    main()
