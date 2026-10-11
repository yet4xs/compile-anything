"""Phase 7C-0: SSA-Constrained Generation — D0/D1/D2/D3 ablation.

Four arms on same model, same inputs, same test set:
  D0: Normal Qwen greedy (baseline)
  D1: Grammar constraints only (Level 1)
  D2: Grammar + Typed SSA (Level 2, with auto-fix)
  D3: Grammar + Typed SSA + Control/Effect (Level 3)

All arms use frozen Composer C2 (same adapter). Differences are only in
how the output is processed/constrained during or after generation.

Since we use post-hoc constraint application (operation-level, not token-level),
D1-D3 apply the same generated text but with different correction strictness.

Also tests the key hypothesis: does constraining fix s43's structural instability?
"""
import json, os, sys, re, copy, time, random, gc
import torch
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.ir.parser import parse_text, TaskIRSyntaxError
from src.ir.taskir import Module, Program, Node
from src.validator.validator import validate
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.constrained_decoder import SSAConstraintEngine, parse_taskir_text_incremental

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}


def load_test_data(n=100):
    """Load test samples from common_eval_tasks (same as integration tests)."""
    data = []
    for line in open("data/phase6b_resolver/common_eval_tasks.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        gold = [c for c in t.get("candidates", []) if c.get("is_gold")]
        if not gold:
            continue
        data.append({
            "id": t["id"], "task": t["task"],
            "cap_blocks": "\n\n".join(c.get("block", "") for c in gold[:5]),
            "ref_text": "",
        })
    # Load reference plans
    for line in open("data/compiler_corpus_v4/test.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for d in data:
            if d["id"] == r["id"]:
                d["ref_text"] = r.get("plan_target", "")
    return data[:n]


def check_semantic(module, ref_text):
    """L2 semantic: OpSeq match."""
    try:
        ref = parse_text(ref_text)
        ref_ops = [n.op for n in ref.program.nodes if n.op not in POLICY_OPS]
        pred_ops = [n.op for n in module.program.nodes if n.op not in POLICY_OPS]
        return pred_ops == ref_ops
    except Exception:
        return False


def apply_constraint_level(text, inputs, level):
    """Apply constraint at given level to generated TaskIR text.

    Returns (module, n_corrections, n_rejections).
    """
    engine = SSAConstraintEngine(module_inputs=inputs, level=level)

    # Parse the text incrementally
    lines = text.strip().split("\n")
    corrections = 0
    rejections = 0
    last_id = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue

        m = re.match(r'(%\w+)\s*=\s*(\w+)\s*\((.*?)\)\s*(?:\[(.*?)\])?', line)
        if not m:
            continue

        node_id, op = m.group(1), m.group(2)
        param_str, input_str = m.group(3) or "", m.group(4) or ""
        inputs_list = [x.strip() for x in input_str.split(",") if x.strip()]

        params = {}
        for pm in re.finditer(r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|[\w.-]+)', param_str):
            params[pm.group(1)] = pm.group(2).strip("'\"")

        # Infer output type
        from src.isa.registry import get as get_skill
        spec = get_skill(op)
        ot = None
        if spec and spec.output_rule:
            ot = spec.output_rule.replace("FIXED:", "").replace("SAME_AS:0", "Any")

        step = engine.submit(node_id, op, inputs_list, params, ot)
        if step.accepted:
            last_id = node_id
            if step.corrected_inputs and step.corrected_inputs != inputs_list:
                corrections += 1
        else:
            rejections += 1

    if last_id:
        ok, _ = engine.set_output(last_id)
        if ok:
            mod = engine.build_module("constrained", last_id)
            if mod:
                return mod, corrections, rejections

    return None, corrections, rejections


def main(n=30, seeds=("s42", "s43", "s44")):
    print("=" * 60)
    print(f"Phase 7C-0: Constrained Generation ({n} samples x {len(seeds)} seeds x 4 levels)")
    print("=" * 60)

    data = load_test_data(n)
    print(f"Test samples: {len(data)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    all_results = {}

    for seed in seeds:
        adapter = f"runs/phase6b/composer_C2_{seed}/final"
        if not os.path.exists(adapter):
            print(f"\n[skip {seed}: not found]")
            continue

        model = PeftModel.from_pretrained(base, adapter)
        model.eval()

        print(f"\n{'='*50}")
        print(f"Seed: {seed}")
        print(f"{'='*50}")

        # Generate once per sample (greedy)
        prompts = []
        for s in data:
            user = f"{s['task']}\n\nSelected capabilities:\n{s['cap_blocks']}"
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True))

        print(f"  Generating {len(prompts)} outputs...")
        outputs = []
        t0 = time.time()
        for bs in range(0, len(prompts), 8):
            if bs % 32 == 0:
                print(f"    {bs}/{len(prompts)}", flush=True)
            chunk = prompts[bs:bs+8]
            inputs = tok(chunk, return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512,
                                     do_sample=False, temperature=None,
                                     pad_token_id=tok.pad_token_id)
            outputs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True) for o in out)
        gen_time = time.time() - t0
        print(f"  Generation time: {gen_time:.1f}s")

        del model
        gc.collect(); torch.cuda.empty_cache()

        # Evaluate at each constraint level
        seed_results = {}
        for level_name, level in [("D0_normal", 0), ("D1_grammar", 1),
                                   ("D2_typed_ssa", 2), ("D3_control", 3)]:
            stats = Counter()
            per_sample = []

            for s, output_text in zip(data, outputs):
                stats["n"] += 1
                taskir_text = extract_taskir_text(output_text)

                if level == 0:
                    # D0: No constraints, standard parse + validate
                    try:
                        mod = parse_text(taskir_text)
                        if validate(mod).valid:
                            stats["valid"] += 1
                            if s["ref_text"] and check_semantic(mod, s["ref_text"]):
                                stats["opseq"] += 1
                        else:
                            stats["invalid"] += 1
                    except Exception:
                        stats["parse_fail"] += 1
                else:
                    # D1-D3: Apply constraint engine
                    inputs_list = [{"name": "@task", "type": "Str"}]
                    mod, corrections, rejections = apply_constraint_level(
                        taskir_text, inputs_list, level)
                    stats["corrections"] += corrections
                    stats["rejections"] += rejections

                    if mod:
                        stats["valid"] += 1
                        if s["ref_text"] and check_semantic(mod, s["ref_text"]):
                            stats["opseq"] += 1
                    else:
                        stats["invalid"] += 1

                per_sample.append({
                    "id": s["id"],
                    "valid": "valid" if (stats["valid"] > 0 and level == 0) or (mod is not None if level > 0 else False) else "invalid",
                })

            n_count = max(1, stats["n"])
            seed_results[level_name] = {
                "n": stats["n"],
                "valid": stats["valid"],
                "valid_pct": round(100 * stats["valid"] / n_count, 1),
                "opseq": stats["opseq"],
                "opseq_pct": round(100 * stats["opseq"] / n_count, 1),
                "corrections": stats.get("corrections", 0),
                "rejections": stats.get("rejections", 0),
            }
            print(f"  {level_name}: valid={seed_results[level_name]['valid_pct']}% "
                  f"opseq={seed_results[level_name]['opseq_pct']}% "
                  f"corrections={seed_results[level_name]['corrections']} "
                  f"rejections={seed_results[level_name]['rejections']}")

        all_results[seed] = seed_results

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Constrained Generation Ablation")
    print(f"{'='*60}")
    print(f"\n{'Seed':<8s} {'D0 Valid%':>10s} {'D1 Valid%':>10s} {'D2 Valid%':>10s} {'D3 Valid%':>10s} | {'D0 OpSeq%':>10s} {'D2 OpSeq%':>10s}")
    print("-" * 75)
    for seed in seeds:
        if seed not in all_results:
            continue
        r = all_results[seed]
        print(f"{seed:<8s} {r['D0_normal']['valid_pct']:>9.1f}% {r['D1_grammar']['valid_pct']:>9.1f}% "
              f"{r['D2_typed_ssa']['valid_pct']:>9.1f}% {r['D3_control']['valid_pct']:>9.1f}% | "
              f"{r['D0_normal']['opseq_pct']:>9.1f}% {r['D2_typed_ssa']['opseq_pct']:>9.1f}%")

    os.makedirs("results/phase7", exist_ok=True)
    with open("results/phase7/constrained_generation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/phase7/constrained_generation.json")
    print("CONSTRAINED-GEN-DONE")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seeds", nargs="*", default=["s42", "s43", "s44"])
    args = ap.parse_args()
    main(args.n, tuple(args.seeds))
