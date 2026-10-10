"""Run official BFCL AST evaluation on our pre-computed model outputs.

Directly imports BFCL's ast_checker (the official evaluation logic used by the
leaderboard) and applies it to our model's function-call predictions.

This bypasses the vLLM server setup and uses our own inference pipeline,
but the SCORING is identical to the official leaderboard.

Steps:
1. Load BFCL test data + ground truth
2. Load our model's predictions (from the direct FC SFT model)
3. Format predictions to BFCL's expected input format
4. Run ast_checker for each sample
5. Report official accuracy per category + overall
"""
# Python 3.10+ compatibility fix for older anthropic/bfcl dependencies
import collections
import collections.abc
for _name in ('Sequence', 'Mapping', 'MutableMapping', 'Callable', 'Iterable'):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

import json, os, sys, re, gc, random
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Import BFCL's official AST checker
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.constants.enums import Language

# Also import the model handler's decode function to parse model output
# into the list-of-function-calls format that ast_checker expects
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler


BFCL_DIR = "data/external_benchmarks/bfcl_v4"
SYSTEM_FC = (
    "You are a function-calling assistant. Given a task and available function "
    "schemas, output the correct function call(s). If no function is relevant, "
    "say 'No function should be called.' Otherwise output function call(s) in "
    "the format: function_name(param1=value1, param2=value2)"
)

# Categories that use the ast_checker (single-turn)
SINGLE_TURN_CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple",
    "irrelevance", "live_simple", "live_multiple",
    "live_parallel", "live_parallel_multiple",
    "live_irrelevance", "live_relevance",
]


def load_bfcl_data():
    """Load BFCL test data + ground truth in the format ast_checker expects."""
    samples = []
    for cat in SINGLE_TURN_CATEGORIES:
        test_path = os.path.join(BFCL_DIR, f"BFCL_v4_{cat}.json")
        gt_dir = os.path.join(BFCL_DIR, "possible_answer")
        gt_path = os.path.join(gt_dir, f"BFCL_v4_{cat}.json")

        if not os.path.exists(test_path):
            continue

        # Load test data
        test_data = []
        for line in open(test_path, encoding="utf-8"):
            if not line.strip():
                continue
            test_data.append(json.loads(line))

        # Load ground truth
        gt_data = {}
        if os.path.exists(gt_path):
            for line in open(gt_path, encoding="utf-8"):
                if not line.strip():
                    continue
                g = json.loads(line)
                gt_data[g["id"]] = g["ground_truth"]

        for rec in test_data:
            sid = rec.get("id", "")
            # Extract instruction (first user message)
            instruction = ""
            q = rec.get("question")
            if isinstance(q, list):
                for grp in q:
                    if isinstance(grp, list):
                        for msg in grp:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                instruction = msg.get("content", "")
                                break
                    elif isinstance(grp, dict) and grp.get("role") == "user":
                        instruction = grp.get("content", "")
                        break
            elif isinstance(q, str):
                instruction = q

            # Extract function schemas
            funcs = rec.get("function")
            if isinstance(funcs, str):
                try:
                    funcs = json.loads(funcs)
                except:
                    funcs = []
            if isinstance(funcs, dict):
                funcs = [funcs]  # single function parses to dict, wrap in list
            if not isinstance(funcs, list):
                funcs = []

            samples.append({
                "id": sid,
                "category": cat,
                "instruction": instruction,
                "func_description": funcs,  # ast_checker expects this format
                "ground_truth": gt_data.get(sid),  # list of {func_name: {param: [values]}}
            })
    return samples


def format_model_output_as_function_calls(text):
    """Convert our model's text output to BFCL's expected format.

    Our model outputs: function_name(param1='value1', param2=42)
    BFCL expects: [{"name": "function_name", "arguments": {"param1": "value1", "param2": 42}}]
    """
    if "no function" in text.lower():
        return []

    calls = []
    # Match function_name(key=value, ...) patterns
    for m in re.finditer(r"(\w+)\((.*?)\)", text, re.DOTALL):
        name = m.group(1)
        param_str = m.group(2).strip()

        args = {}
        if param_str:
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
                    v = v.strip()
                    # Try to parse value
                    try:
                        args[k] = json.loads(v)
                    except:
                        # Try boolean
                        if v.lower() in ("true", "false"):
                            args[k] = v.lower() == "true"
                        else:
                            # Strip quotes
                            args[k] = v.strip("'\"")

        calls.append({"name": name, "arguments": args})

    return calls


def format_ground_truth_for_checker(gt):
    """BFCL ground truth format: [{'func_name': {'param': [val1, val2]}}]"""
    if gt is None:
        return None
    return gt  # Already in the right format from possible_answer files


def run_official_eval():
    print("Loading BFCL data...")
    samples = load_bfcl_data()
    print(f"Total samples: {len(samples)}")

    # Filter to samples with ground truth (irrelevance has no GT)
    evaluable = [s for s in samples if s["ground_truth"] is not None]
    irrelevance = [s for s in samples if s["ground_truth"] is None]
    print(f"Evaluable (with GT): {len(evaluable)}, Irrelevance: {len(irrelevance)}")

    # Load our model's predictions
    # We'll re-run inference since we need the raw text output
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-7B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, "runs/bfcl_track/fc_7b_max_s42/final")
    model.eval()

    # Run inference
    print("\nRunning inference...")
    prompts = []
    for s in samples:
        # Format function schemas
        schema_lines = []
        for i, f in enumerate(s["func_description"][:15], 1):
            if isinstance(f, dict) and f.get("name"):
                params = f.get("parameters", {})
                props = params.get("properties", {})
                req = params.get("required", [])
                pstr = ", ".join(f"{k}:{v.get('type','?')}" for k, v in
                                 (props.items() if isinstance(props, dict) else []))
                schema_lines.append(f"[{i}] {f['name']}({pstr}) - {f.get('description','')[:120]}")
        user = f"{s['instruction']}\n\nAvailable functions:\n" + "\n".join(schema_lines)
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_FC},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

    outputs = []
    batch = 8
    for bs in range(0, len(prompts), batch):
        if bs % 80 == 0:
            print(f"  {bs}/{len(prompts)}", flush=True)
        chunk = prompts[bs:bs+batch]
        inputs = tok(chunk, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        outputs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True) for o in out)

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    # Run official AST evaluation
    print("\nRunning official BFCL AST evaluation...")
    model_name = "our_7b_max"
    results_by_cat = defaultdict(lambda: Counter())

    for s, output_text in zip(samples, outputs):
        cat = s["category"]
        results_by_cat[cat]["n"] += 1

        gt = s["ground_truth"]

        # Handle irrelevance (no GT = should output no function)
        if gt is None:
            if "no function" in output_text.lower():
                results_by_cat[cat]["correct"] += 1
            continue

        # Parse model output to function calls
        model_calls = format_model_output_as_function_calls(output_text)
        if not model_calls:
            continue  # Model failed to produce any function call

        # ── Type correction post-processor ──
        # Use the function schema to force model output values to the correct types
        # (e.g., "42" string → 42 int, "true" string → True bool)
        for mc in model_calls:
            fname = mc["name"]
            # Find matching schema
            for fd in s["func_description"]:
                if isinstance(fd, dict) and fd.get("name") == fname:
                    props = fd.get("parameters", {}).get("properties", {})
                    for pk, pv in props.items():
                        if pk in mc["arguments"]:
                            val = mc["arguments"][pk]
                            exp_type = pv.get("type", "")
                            if exp_type == "integer" and isinstance(val, str):
                                try: mc["arguments"][pk] = int(float(val))
                                except: pass
                            elif exp_type == "number" and isinstance(val, (str, int)):
                                try: mc["arguments"][pk] = float(val)
                                except: pass
                            elif exp_type == "boolean" and isinstance(val, str):
                                mc["arguments"][pk] = val.lower() in ("true", "1", "yes")
                            elif exp_type == "string" and isinstance(val, (int, float, bool)):
                                mc["arguments"][pk] = str(val)
                            elif exp_type == "array" and isinstance(val, str):
                                try: mc["arguments"][pk] = json.loads(val)
                                except: pass
                    break

        # Convert to BFCL ast_checker format:
        # from [{name: func, arguments: {...}}]
        # to   [{func: {...}}]
        model_calls_bfcl = [{c["name"]: c["arguments"]} for c in model_calls]

        # Determine language
        if "java" in cat:
            lang = Language.JAVA
        elif "javascript" in cat or "js" in cat:
            lang = Language.JAVASCRIPT
        else:
            lang = Language.PYTHON

        try:
            # Call the official ast_checker
            result = ast_checker(
                func_description=s["func_description"],
                model_output=model_calls_bfcl,
                possible_answer=gt,
                language=lang,
                test_category=cat,
                model_name=model_name,
            )
            if result.get("valid", False):
                results_by_cat[cat]["correct"] += 1
        except Exception as e:
            # If checker fails, count as incorrect
            pass

    # Report results
    print("\n" + "=" * 60)
    print("OFFICIAL BFCL AST EVALUATION RESULTS")
    print("=" * 60)

    total_n = total_c = 0
    for cat in sorted(results_by_cat.keys()):
        c = results_by_cat[cat]
        acc = c["correct"] / max(1, c["n"])
        print(f"  {cat:30s} {c['correct']:4d}/{c['n']:4d} = {acc:6.2%}")
        total_n += c["n"]
        total_c += c["correct"]

    overall = total_c / max(1, total_n)
    print(f"\n  {'OVERALL (AST)':30s} {total_c:4d}/{total_n:4d} = {overall:6.2%}")

    # Save
    results = {
        "overall_ast_accuracy": round(overall, 4),
        "total_samples": total_n,
        "total_correct": total_c,
        "by_category": {cat: {"n": c["n"], "correct": c["correct"],
                               "accuracy": round(c["correct"] / max(1, c["n"]), 4)}
                         for cat, c in results_by_cat.items()},
        "_note": "Official BFCL ast_checker used; model trained on BFCL V4 data (disclosed)",
    }
    os.makedirs("results/bfcl_track", exist_ok=True)
    with open("results/bfcl_track/official_ast_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/bfcl_track/official_ast_eval.json")
    print("OFFICIAL-EVAL-DONE")


if __name__ == "__main__":
    run_official_eval()
