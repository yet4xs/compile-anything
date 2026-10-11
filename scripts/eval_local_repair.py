"""Phase 7B-1: Local Repair vs Full Regeneration comparison.

Three arms (R0/R1/R2) on s43's 310 failed TaskIR programs:
  R0: Improved rule-based repair (deterministic)
  R1: Full regeneration with frozen Qwen Composer
  R2: Validator-guided SSA local patch (neural_repair.py)

Reports L1 (structural), L2 (semantic), cost (tokens/latency), and edit scope.
Runs 30-sample smoke test first (--smoke), then full 310 (--full).
"""
import json, os, sys, re, copy, time, gc, random
import torch
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.ir.parser import parse_text, TaskIRSyntaxError
from src.ir.taskir import Module
from src.validator.validator import validate
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.repair.ssa_slicer import extract_slice
from src.repair.patch_ir import apply_patch
from src.repair.neural_repair import neural_repair, REPAIR_SYSTEM
from src.isa.registry import get as get_skill

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}


def load_failures():
    failures = []
    path = "results/phase6b/preds/c2_s43.jsonl"
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        taskir_text = extract_taskir_text(rec.get("output", ""))
        if not taskir_text:
            continue
        try:
            module = parse_text(taskir_text)
        except Exception:
            continue
        try:
            report = validate(module)
        except Exception:
            continue
        if not report.valid:
            failures.append({
                "id": rec["id"], "taskir_text": taskir_text, "module": module,
                "report": report, "error_codes": [e.code for e in report.errors],
            })
    return failures


# ── R0: Improved rule-based repair ──

def rule_repair_v2(module):
    """Improved deterministic repair: SSA-order-aware, type-checked."""
    mod = copy.deepcopy(module)
    total_fixes = 0

    for _ in range(5):
        try:
            if validate(mod).valid:
                break
        except Exception:
            break

        fixes = 0
        nodes = mod.program.nodes
        all_ids = {n.id for n in nodes}
        input_ids = {f"@{i['name'].lstrip('@')}" for i in mod.program.inputs}

        # UNDEFINED_REF: replace with type-compatible node defined BEFORE use
        for i, node in enumerate(nodes):
            defined_before = {n.id for n in nodes[:i]} | input_ids
            for j, inp in enumerate(node.inputs):
                if inp not in all_ids and inp not in input_ids:
                    spec = get_skill(node.op)
                    candidates = [n.id for n in nodes[:i]
                                  if not spec or not spec.input_types
                                  or spec.input_types[0] in (n.output_type or "Any", "Any")]
                    if not candidates:
                        candidates = list(defined_before)
                    if candidates:
                        node.inputs[j] = candidates[-1]
                        fixes += 1

        # ARITY: truncate extra inputs
        for node in nodes:
            spec = get_skill(node.op)
            if spec and spec.max_inputs and len(node.inputs) > spec.max_inputs:
                node.inputs = node.inputs[:spec.max_inputs]
                fixes += 1

        # Missing output
        if not mod.program.output and nodes:
            mod.program.output = nodes[-1].id
            fixes += 1

        total_fixes += fixes
        if fixes == 0:
            break

    try:
        valid = validate(mod).valid
    except Exception:
        valid = False
    return mod, valid, total_fixes


def check_semantic(module, ref_text):
    """L2: OpSeq match with reference."""
    try:
        ref = parse_text(ref_text)
        ref_ops = [n.op for n in ref.program.nodes if n.op not in POLICY_OPS]
        pred_ops = [n.op for n in module.program.nodes if n.op not in POLICY_OPS]
        return pred_ops == ref_ops
    except Exception:
        return False


def main(n_samples=30, model_path="runs/phase6b/composer_C2_s42/final"):
    print("=" * 60)
    print(f"Phase 7B-1: Local Repair Comparison ({n_samples} samples)")
    print("=" * 60)

    failures = load_failures()
    print(f"Total failures available: {len(failures)}")

    # Stratified sample
    random.seed(42)
    by_error = defaultdict(list)
    for f in failures:
        primary = f["error_codes"][0] if f["error_codes"] else "UNKNOWN"
        by_error[primary].append(f)

    sample = []
    for etype, items in sorted(by_error.items()):
        random.shuffle(items)
        sample.extend(items[:max(1, n_samples // len(by_error))])
    sample = sample[:n_samples]
    sample_dist = dict(Counter(f["error_codes"][0] for f in sample))
    print(f"Stratified sample: {len(sample)} ({sample_dist})")

    # Load reference plans for semantic check
    ref_map = {}
    for line in open("data/compiler_corpus_v4/test.jsonl", encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        ref_map[r["id"]] = r.get("plan_target", "")

    # Load capability blocks
    cap_map = {}
    for line in open("data/phase6b_resolver/common_eval_tasks.jsonl", encoding="utf-8"):
        if not line.strip(): continue
        t = json.loads(line)
        caps = [c for c in t.get("candidates", []) if c.get("is_gold")]
        cap_map[t["id"]] = "\n".join(c.get("block", "") for c in caps[:3])

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()

    # ── R0: Rule-based ──
    print("\n--- R0: Rule-based repair v2 ---")
    r0_stats = Counter()
    r0_details = []
    for f in sample:
        r0_stats["n"] += 1
        repaired, valid, fixes = rule_repair_v2(f["module"])
        if valid:
            r0_stats["l1"] += 1
            ref = ref_map.get(f["id"], "")
            if ref and check_semantic(repaired, ref):
                r0_stats["l2"] += 1
        r0_details.append({"id": f["id"], "l1": valid, "fixes": fixes})
    print(f"  L1 (structural): {r0_stats['l1']}/{r0_stats['n']} ({100*r0_stats['l1']/max(1,r0_stats['n']):.1f}%)")
    print(f"  L2 (semantic):   {r0_stats['l2']}/{r0_stats['n']} ({100*r0_stats['l2']/max(1,r0_stats['n']):.1f}%)")
    print(f"  Cost: 0 tokens (deterministic)")

    # ── R1: Full regeneration ──
    print("\n--- R1: Full regeneration ---")
    r1_stats = Counter()
    r1_details = []

    # Get original task + capabilities for re-prompting
    task_info = {}
    for line in open("data/phase6b_resolver/common_eval_tasks.jsonl", encoding="utf-8"):
        if not line.strip(): continue
        t = json.loads(line)
        caps = [c for c in t.get("candidates", []) if c.get("is_gold")]
        task_info[t["id"]] = {
            "task": t.get("task", ""),
            "caps": "\n\n".join(c.get("block", "") for c in caps[:5]),
        }

    for f in sample:
        r1_stats["n"] += 1
        info = task_info.get(f["id"], {})
        if not info:
            r1_details.append({"id": f["id"], "l1": False, "l2": False, "tokens": 0})
            continue

        user = f"{info['task']}\n\nSelected capabilities:\n{info['caps']}"
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        output_text = tok.decode(out[0][input_len:], skip_special_tokens=True)
        tokens = out.shape[1] - input_len
        latency = (time.time() - t0) * 1000

        try:
            mod = parse_text(extract_taskir_text(output_text))
            if validate(mod).valid:
                r1_stats["l1"] += 1
                ref = ref_map.get(f["id"], "")
                if ref and check_semantic(mod, ref):
                    r1_stats["l2"] += 1
        except Exception:
            pass
        r1_details.append({"id": f["id"], "l1": False, "l2": False, "tokens": tokens, "latency_ms": latency})

    print(f"  L1: {r1_stats['l1']}/{r1_stats['n']} ({100*r1_stats['l1']/max(1,r1_stats['n']):.1f}%)")
    print(f"  L2: {r1_stats['l2']}/{r1_stats['n']} ({100*r1_stats['l2']/max(1,r1_stats['n']):.1f}%)")
    avg_tokens = sum(d.get("tokens", 0) for d in r1_details) / max(1, len(r1_details))
    print(f"  Avg tokens: {avg_tokens:.0f}")

    # ── R2: Neural local repair ──
    print("\n--- R2: Validator-guided local patch ---")
    r2_stats = Counter()
    r2_details = []

    for f in sample:
        r2_stats["n"] += 1
        info = task_info.get(f["id"], {})
        caps = info.get("caps", "")

        result = neural_repair(
            task=info.get("task", ""),
            module=f["module"],
            report=f["report"],
            capabilities=caps,
            model=model, tok=tok,
            max_attempts=2, max_repair_nodes=3,
        )

        if result["success"]:
            r2_stats["l1"] += 1
            ref = ref_map.get(f["id"], "")
            if ref and check_semantic(result["module"], ref):
                r2_stats["l2"] += 1

        r2_details.append({
            "id": f["id"], "l1": result["success"],
            "l2": r2_stats["l2"] if result["success"] else False,
            "tokens": result["total_tokens"],
            "changed": result["changed_nodes"],
            "attempts": len(result["attempts"]),
        })
        if result["success"]:
            r2_details[-1]["l2"] = bool(ref and check_semantic(result["module"], ref))

    print(f"  L1: {r2_stats['l1']}/{r2_stats['n']} ({100*r2_stats['l1']/max(1,r2_stats['n']):.1f}%)")
    print(f"  L2: {r2_stats['l2']}/{r2_stats['n']} ({100*r2_stats['l2']/max(1,r2_stats['n']):.1f}%)")
    avg_tokens2 = sum(d.get("tokens", 0) for d in r2_details) / max(1, len(r2_details))
    print(f"  Avg tokens: {avg_tokens2:.0f}")
    avg_changed = sum(len(d.get("changed", [])) for d in r2_details) / max(1, len(r2_details))
    print(f"  Avg changed nodes: {avg_changed:.1f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Arm':<35s} {'L1%':>6s} {'L2%':>6s} {'Avg Tokens':>10s}")
    print("-" * 60)
    print(f"{'R0 Rule-based':<35s} {100*r0_stats['l1']/max(1,r0_stats['n']):>5.1f}% {100*r0_stats['l2']/max(1,r0_stats['n']):>5.1f}% {0:>10d}")
    print(f"{'R1 Full regeneration':<35s} {100*r1_stats['l1']/max(1,r1_stats['n']):>5.1f}% {100*r1_stats['l2']/max(1,r1_stats['n']):>5.1f}% {avg_tokens:>10.0f}")
    print(f"{'R2 Local patch':<35s} {100*r2_stats['l1']/max(1,r2_stats['n']):>5.1f}% {100*r2_stats['l2']/max(1,r2_stats['n']):>5.1f}% {avg_tokens2:>10.0f}")

    # Save
    os.makedirs("results/repair", exist_ok=True)
    with open("results/repair/local_repair_comparison.json", "w") as fo:
        json.dump({
            "n_samples": len(sample),
            "R0": {"l1": r0_stats["l1"], "l2": r0_stats["l2"], "tokens": 0, "details": r0_details},
            "R1": {"l1": r1_stats["l1"], "l2": r1_stats["l2"], "avg_tokens": avg_tokens, "details": r1_details},
            "R2": {"l1": r2_stats["l1"], "l2": r2_stats["l2"], "avg_tokens": avg_tokens2, "details": r2_details},
        }, fo, indent=2)
    print("\nSaved to results/repair/local_repair_comparison.json")
    print("LOCAL-REPAIR-DONE")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of samples (smoke=30, full=310)")
    ap.add_argument("--model", default="runs/phase6b/composer_C2_s42/final")
    args = ap.parse_args()
    main(args.n, args.model)
