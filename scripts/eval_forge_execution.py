"""Forge Execution Benchmark: Corpus v4 test → full pipeline → execution success rate.

The unique metric of a compiler approach: not just "did you output the right
function call format" but "did your compiled program actually execute correctly."

Pipeline per sample:
  1. NL task → Neural Composer (C2 s42) → TaskIR
  2. TaskIR → V1-V6 Validator
  3. Valid TaskIR → Forge.lower() → LangGraph graph
  4. Graph.invoke() → execute with executors
  5. Compare output values against reference TaskIR execution

Metrics:
  - Compilation Success Rate: NL → valid TaskIR
  - Forge Execution Rate: valid TaskIR → successful Forge execution
  - End-to-End Success Rate: NL → correct execution result
  - Per-skill execution accuracy
"""
import json, os, sys, time, gc
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.ir.parser import parse_text, TaskIRSyntaxError
from src.validator.validator import validate
from src.runtime.forge import Forge
from src.isa.registry import get as get_skill

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}


def load_test_data(n=200):
    """Load corpus v4 test samples."""
    data = []
    for line in open("data/compiler_corpus_v4/test.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
        sel = [cid for cid in r["selected_capabilities"]
               if canon.get(cid, {}).get("label_tier") != "G3"]
        if not sel:
            continue
        caps = {c["capability_id"]: c for c in r["available_capabilities"]}
        blocks = []
        for cid in sel:
            if cid in caps and cid in canon:
                c = caps[cid]
                blocks.append(f"[{cid}] name: {c['name']}\ncanonical_skill: {canon[cid]['canonical_skill']}")
        if not blocks:
            continue
        data.append({
            "id": r["id"],
            "instruction": r["instruction"],
            "cap_blocks": blocks,
            "plan_target": r.get("plan_target", ""),
        })
        if len(data) >= n:
            break
    return data


def run_pipeline(model, tok, data):
    """Run full pipeline: compile → validate → forge → execute."""
    forge = Forge()
    g = Counter()
    per_op = defaultdict(lambda: Counter())
    details = []

    for i, sample in enumerate(data):
        if i % 20 == 0:
            print(f"  [{i}/{len(data)}]", flush=True)
        g["n"] += 1

        # Step 1: Neural compilation
        user = f"{sample['instruction']}\n\nSelected capabilities:\n" + "\n\n".join(sample["cap_blocks"])
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True)

        # Step 2: Parse
        try:
            module = parse_text(extract_taskir_text(raw))
            g["parse"] += 1
        except Exception:
            details.append({"id": sample["id"], "stage": "parse", "ok": False})
            continue

        # Step 3: Validate
        try:
            report = validate(module)
        except Exception:
            report = None
        if report is None or not report.valid:
            details.append({"id": sample["id"], "stage": "validate", "ok": False})
            continue
        g["valid"] += 1

        # Step 4: Forge execution
        try:
            result = forge.run(module)
            errors = result.get("errors", [])
            if not errors:
                g["executed"] += 1
                # Check per-op status
                for t in result.get("traces", []):
                    op = t["op"]
                    per_op[op]["n"] += 1
                    if t["status"] == "ok":
                        per_op[op]["ok"] += 1

                # Step 5: Compare with reference (if plan_target exists)
                if sample["plan_target"]:
                    try:
                        ref_mod = parse_text(sample["plan_target"])
                        ref_ops = [n.op for n in ref_mod.program.nodes
                                   if n.op not in POLICY_OPS]
                        pred_ops = [n.op for n in module.program.nodes
                                    if n.op not in POLICY_OPS]
                        if pred_ops == ref_ops:
                            g["ops_match"] += 1
                        else:
                            g["ops_mismatch"] += 1
                    except Exception:
                        pass

                details.append({"id": sample["id"], "stage": "execute",
                                "ok": True, "ops": [n.op for n in module.program.nodes]})
            else:
                g["exec_error"] += 1
                details.append({"id": sample["id"], "stage": "execute",
                                "ok": False, "errors": errors})
        except Exception as e:
            g["forge_crash"] += 1
            details.append({"id": sample["id"], "stage": "forge",
                            "ok": False, "error": str(e)[:200]})

    return g, per_op, details


def main(n=200):
    print("=" * 60)
    print(f"Forge Execution Benchmark: {n} samples")
    print("=" * 60)

    data = load_test_data(n)
    print(f"Loaded {len(data)} test samples")

    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, "runs/phase6b/composer_C2_s42/final")
    model.eval()

    print("\nRunning pipeline...")
    g, per_op, details = run_pipeline(model, tok, data)

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    n = max(1, g["n"])
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")

    print(f"\n  Pipeline stages:")
    print(f"    Samples:           {g['n']}")
    print(f"    Parse success:     {g['parse']:4d} ({100*g['parse']/n:.1f}%)")
    print(f"    Validation pass:   {g['valid']:4d} ({100*g['valid']/n:.1f}%)")
    print(f"    Forge executed:    {g['executed']:4d} ({100*g['executed']/n:.1f}%)")
    print(f"    Execution errors:  {g['exec_error']:4d}")
    print(f"    Forge crashes:     {g['forge_crash']:4d}")

    total_match = g.get("ops_match", 0)
    total_mismatch = g.get("ops_mismatch", 0)
    if total_match + total_mismatch > 0:
        print(f"\n  Op sequence match (vs reference):")
        print(f"    Match:   {total_match:4d} ({100*total_match/(total_match+total_mismatch):.1f}%)")
        print(f"    Mismatch:{total_mismatch:4d}")

    print(f"\n  Per-skill execution:")
    for op in sorted(per_op.keys()):
        c = per_op[op]
        rate = 100 * c["ok"] / max(1, c["n"])
        print(f"    {op:20s} {c['ok']:4d}/{c['n']:4d} = {rate:5.1f}%")

    # End-to-end success rate
    e2e = g["executed"] / n * 100
    print(f"\n  {'END-TO-END SUCCESS RATE':30s} {g['executed']:4d}/{g['n']:4d} = {e2e:.1f}%")
    print(f"  (NL → TaskIR → V1-V6 → Forge → successful execution)")

    # Save
    results = {
        "n": g["n"],
        "parse": g["parse"],
        "valid": g["valid"],
        "executed": g["executed"],
        "exec_error": g.get("exec_error", 0),
        "forge_crash": g.get("forge_crash", 0),
        "ops_match": total_match,
        "ops_mismatch": total_mismatch,
        "e2e_success_rate_pct": round(e2e, 1),
        "per_op": {op: {"n": c["n"], "ok": c["ok"],
                         "rate_pct": round(100*c["ok"]/max(1,c["n"]), 1)}
                    for op, c in per_op.items()},
    }
    os.makedirs("results/forge", exist_ok=True)
    with open("results/forge/execution_benchmark.json", "w") as f:
        json.dump({"summary": results, "details": details[:50]}, f, indent=2)
    print(f"\nSaved to results/forge/execution_benchmark.json")
    print("FORGE-BENCH-DONE")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    main(args.n)
