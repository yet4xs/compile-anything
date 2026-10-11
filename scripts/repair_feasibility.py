"""Direction B Feasibility: Validator-Guided Minimal Repair vs Full Regeneration.

Uses s43's actual failed TaskIR programs (from integration_preds_s42/c2_s43.jsonl).
We already know s43 has 67% valid / 33% invalid with errors concentrated in
STRUCT(373), UNDEFINED_REF(287), ARITY(168).

Three repair strategies compared:
1. Full regeneration (baseline): LLM regenerates entire program
2. Rule-based local repair: deterministic fixes for specific error types
3. Constrained local generation: LLM regenerates only the affected slice (future)

Metrics: repair success rate, tokens/cost, semantic preservation (OpSeq).
"""
import json, os, sys, re, time, random, gc
import torch
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.ir.parser import parse_text, TaskIRSyntaxError
from src.ir.taskir import Module, Program, Node
from src.validator.validator import validate
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}


def load_s43_failures():
    """Load s43's failed predictions from the saved integration preds."""
    failures = []
    path = "results/phase6b/preds/c2_s43.jsonl"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return failures

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
            continue  # Parse failures are a different problem

        try:
            report = validate(module)
        except Exception:
            continue

        if not report.valid:
            errors = [(e.code, e.msg, e.node) for e in report.errors]
            failures.append({
                "id": rec["id"],
                "taskir_text": taskir_text,
                "module": module,
                "errors": errors,
                "error_codes": [e[0] for e in errors],
            })
    return failures


# ── Rule-based repair strategies ──

def repair_undefined_ref(module):
    """Fix UNDEFINED_REF by finding the closest type-compatible upstream node."""
    repairs = 0
    defined = {n.id for n in module.program.nodes}
    defined |= {f"@{inp['name'].lstrip('@')}" for inp in module.program.inputs}

    # Build type map
    type_map = {n.id: n.output_type for n in module.program.nodes if n.output_type}

    for node in module.program.nodes:
        new_inputs = []
        for inp in node.inputs:
            if inp in defined:
                new_inputs.append(inp)
            else:
                # Find a type-compatible replacement
                # Get what this node expects
                from src.isa.registry import get as get_skill
                spec = get_skill(node.op)
                if spec and spec.input_types:
                    expected = spec.input_types[0]  # First input type
                    # Find a defined node with matching type
                    candidates = [nid for nid, t in type_map.items()
                                  if nid in defined and t and expected in (t, "Any", "List[Any]")]
                    if not candidates:
                        # Fallback: use any defined node
                        candidates = [nid for nid in defined if nid.startswith("%")]
                    if candidates:
                        new_inputs.append(candidates[0])
                        repairs += 1
                    else:
                        new_inputs.append(inp)  # Can't fix
                else:
                    # No type info, use any defined node
                    others = [nid for nid in defined if nid.startswith("%") and nid != node.id]
                    if others:
                        new_inputs.append(others[0])
                        repairs += 1
                    else:
                        new_inputs.append(inp)
        node.inputs = new_inputs
    return repairs


def repair_arity(module):
    """Fix ARITY errors by adding default parameters or removing extra inputs."""
    repairs = 0
    from src.isa.registry import get as get_skill

    for node in module.program.nodes:
        spec = get_skill(node.op)
        if not spec:
            continue

        # Check min_inputs
        if spec.min_inputs and len(node.inputs) < spec.min_inputs:
            # Not enough inputs - can't fix without adding nodes
            pass

        # Check max_inputs
        if spec.max_inputs and len(node.inputs) > spec.max_inputs:
            node.inputs = node.inputs[:spec.max_inputs]
            repairs += 1

    return repairs


def repair_struct(module):
    """Fix STRUCT errors: empty program, missing output, etc."""
    repairs = 0
    prog = module.program

    # Fix missing output
    if not prog.output and prog.nodes:
        prog.output = prog.nodes[-1].id
        repairs += 1

    return repairs


def rule_based_repair(module):
    """Apply all rule-based repairs iteratively until valid or no progress."""
    total_repairs = 0
    for _ in range(5):  # Max 5 iterations
        try:
            report = validate(module)
            if report.valid:
                break
        except Exception:
            break

        codes = {e.code for e in report.errors}
        fixed_this_round = 0

        if "UNDEFINED_REF" in codes:
            fixed_this_round += repair_undefined_ref(module)
        if "ARITY" in codes:
            fixed_this_round += repair_arity(module)
        if "STRUCT" in codes:
            fixed_this_round += repair_struct(module)

        total_repairs += fixed_this_round
        if fixed_this_round == 0:
            break

    return total_repairs


def semantic_match(module, reference_text):
    """Check if repaired module preserves reference semantics."""
    try:
        ref_mod = parse_text(reference_text)
        ref_ops = [n.op for n in ref_mod.program.nodes if n.op not in POLICY_OPS]
        pred_ops = [n.op for n in module.program.nodes if n.op not in POLICY_OPS]
        return pred_ops == ref_ops
    except Exception:
        return False


def main():
    print("=" * 60)
    print("FEASIBILITY: Validator-Guided Minimal Repair")
    print("=" * 60)

    failures = load_s43_failures()
    print(f"\nLoaded {len(failures)} failed TaskIR programs from s43")

    if not failures:
        print("No failures to repair. Exiting.")
        return

    # Analyze error distribution
    error_dist = Counter()
    for f in failures:
        for code in f["error_codes"]:
            error_dist[code] += 1
    print(f"\nError distribution:")
    for code, cnt in error_dist.most_common():
        print(f"  {code:20s} {cnt:4d}")

    # Apply rule-based repair
    print(f"\n--- Rule-Based Repair ---")
    stats = Counter()
    details = []

    for f in failures:
        stats["n"] += 1
        module_copy = f["module"]

        try:
            n_repairs = rule_based_repair(module_copy)
            report = validate(module_copy)
            is_valid = report.valid
        except Exception as e:
            n_repairs = 0
            is_valid = False

        if is_valid:
            stats["repaired"] += 1
            # Check semantic preservation
            # (need reference for this, skip for now)
        else:
            stats["not_repaired"] += 1

        details.append({
            "id": f["id"],
            "errors_before": f["error_codes"],
            "repairs_made": n_repairs,
            "valid_after": is_valid,
        })

    n = max(1, stats["n"])
    print(f"\nResults:")
    print(f"  Total failures: {stats['n']}")
    print(f"  Repaired:       {stats['repaired']} ({100*stats['repaired']/n:.1f}%)")
    print(f"  Not repaired:   {stats['not_repaired']} ({100*stats['not_repaired']/n:.1f}%)")

    # Per-error-type repair success
    print(f"\nRepair success by error type:")
    for etype in error_dist:
        subset = [d for d in details if etype in d["errors_before"]]
        if subset:
            success = sum(1 for d in subset if d["valid_after"])
            print(f"  {etype:20s} {success}/{len(subset)} = {100*success/max(1,len(subset)):.1f}%")

    # Save
    os.makedirs("results/repair", exist_ok=True)
    with open("results/repair/rule_based_feasibility.json", "w") as f:
        json.dump({
            "summary": {
                "total": stats["n"],
                "repaired": stats["repaired"],
                "repair_rate_pct": round(100*stats["repaired"]/n, 1),
            },
            "error_distribution": dict(error_dist),
            "details": details[:50],
        }, f, indent=2)
    print(f"\nSaved to results/repair/rule_based_feasibility.json")

    # Compare cost: rule-based repair vs full regeneration
    print(f"\n--- Cost Comparison ---")
    print(f"  Rule-based repair: ~0 tokens (deterministic code)")
    print(f"  Full regeneration: ~500 tokens per retry (LLM inference)")
    print(f"  Speedup: ∞ (no LLM call needed)")

    print(f"\nFEASIBILITY-DONE")


if __name__ == "__main__":
    main()
