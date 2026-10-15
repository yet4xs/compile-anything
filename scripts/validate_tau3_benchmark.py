"""Phase 9B-2b: Validate τ³ benchmark — TaskIR generation + strict validation + Mock Runtime."""
import json, os, sys, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir("/ccfa2026/compile-anything") else "/ccfa2026/compile-anything")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.compiler.plan_skeleton import skeleton_from_dict, validate_skeleton, hard_errors
from src.compiler.skeleton_lowering import lower_skeleton
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.runtime.mock_capability_registry import (
    MockCapabilityRegistry, MockCapability, DataflowRuntime)

DATA = "data/tau3_runtime_benchmark"


def make_mock_registry(cap_registry):
    """Build deterministic mock executors from tool schemas."""
    reg = MockCapabilityRegistry()
    for tid, meta in cap_registry.items():
        outs = meta.get("output_schema", {})
        ins = meta.get("input_schema", {})
        eff = meta.get("effect_class", "read_only")

        def mk(tid, outs):
            def fn(params):
                h = int(hashlib.md5(json.dumps(
                    params, sort_keys=True, default=str).encode()
                ).hexdigest()[:6], 16)
                r = {}
                for k, t in outs.items():
                    if t.startswith("List"):
                        r[k] = [{"id": f"{tid}_{k}_{h%99}"}]
                    elif t == "Int": r[k] = h % 1000
                    elif t == "Float": r[k] = float(h % 10000) / 100
                    elif t == "Bool": r[k] = h % 2 == 0
                    else: r[k] = f"{tid}_{k}_{h%9999}"
                return r
            return fn

        reg.register(MockCapability(tid, ins, outs, eff, mk(tid, outs)))
    return reg


def lower_skeleton_to_taskir(rec):
    """Lower the task's skeleton to TaskIR v0.2."""
    sk = skeleton_from_dict(rec["target_skeleton"])

    schemas = {}
    for tid, meta in rec["capability_registry"].items():
        schemas[tid] = {
            "params": [{"name": k, "optional": False, "type": v}
                        for k, v in meta.get("input_schema", {}).items()],
            "output_fields": meta.get("output_schema", {}),
            "effect_class": meta.get("effect_class", "read_only"),
        }

    low = lower_skeleton(sk, None, schemas, strict=True)
    return low.get("taskir"), low.get("errors", [])


def main():
    results = {"total": 0, "lowering_pass": 0, "validator_pass": 0,
               "execution_pass": 0, "binding_extracted": 0, "failures": []}

    for split in ("train", "dev", "test"):
        path = os.path.join(DATA, f"{split}.jsonl")
        if not os.path.exists(path):
            continue

        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            results["total"] += 1

            # 1. Lower skeleton → TaskIR
            taskir, errs = lower_skeleton_to_taskir(rec)
            if taskir is None:
                results["failures"].append(
                    {"task": rec["task_id"], "stage": "lowering",
                     "err": str(errs)[:100]})
                continue
            results["lowering_pass"] += 1

            # Store TaskIR
            rec["m0_taskir"] = taskir

            # 2. Strict validation
            v2_reg = {t: {"input_schema": rec["capability_registry"][t].get("input_schema", {}),
                          "output_schema": rec["capability_registry"][t].get("output_schema", {})}
                      for t in rec["capability_registry"]}
            mod = parse_text(taskir)
            rep = validate(mod, capability_registry=v2_reg)
            if not rep.valid:
                results["failures"].append(
                    {"task": rec["task_id"], "stage": "validator",
                     "err": str(rep.errors[:2])[:100]})
                continue
            results["validator_pass"] += 1

            # 3. Mock Runtime execution
            mock_reg = make_mock_registry(rec["capability_registry"])
            rt = DataflowRuntime(mock_reg)
            tr = rt.execute(mod, capability_registry=v2_reg)
            if not tr.success:
                results["failures"].append(
                    {"task": rec["task_id"], "stage": "runtime",
                     "err": tr.error[:100]})
                continue
            results["execution_pass"] += 1

            # 4. Verify runtime deps have EXTRACT_FIELD in TaskIR
            n_extract = taskir.count("EXTRACT_FIELD")
            n_ref_deps = rec["n_runtime_deps"]
            if n_extract >= n_ref_deps:
                results["binding_extracted"] += 1

    # Report
    n = results["total"]
    print(f"Total tasks: {n}")
    print(f"Lowering pass: {results['lowering_pass']}/{n} "
          f"({100*results['lowering_pass']/n:.1f}%)")
    print(f"Validator pass: {results['validator_pass']}/{n} "
          f"({100*results['validator_pass']/n:.1f}%)")
    print(f"Execution pass: {results['execution_pass']}/{n} "
          f"({100*results['execution_pass']/n:.1f}%)")
    print(f"Binding extracted: {results['binding_extracted']}/{n}")
    if results["failures"]:
        print(f"\nFailures ({len(results['failures'])}:")
        for f in results["failures"][:5]:
            print(f"  {f['task']}: {f['stage']} — {f['err']}")

    # Save validation report
    with open("results/phase9/tau3_benchmark_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved results/phase9/tau3_benchmark_validation.json")


if __name__ == "__main__":
    main()
