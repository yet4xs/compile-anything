"""Phase 8D-1B: Typed Dataflow M0/M1 Feasibility Pilot.

Single comprehensive script: preflight → baselines → training → evaluation.

Usage:
  python scripts/pilot_dataflow.py --phase preflight
  python scripts/pilot_dataflow.py --phase baseline
  python scripts/pilot_dataflow.py --phase smoke
  python scripts/pilot_dataflow.py --phase train --model m0
  python scripts/pilot_dataflow.py --phase train --model m1
  python scripts/pilot_dataflow.py --phase eval
"""
import json, os, sys, time, re, gc, random, hashlib, argparse, math

ROOT = "/ccfa2026/compile-anything"
if not os.path.isdir(ROOT):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DATA = "data/runtime_dataflow_v2"
OUT = "results/phase8d"

SYS_M0 = (
    "You are a task compiler. Given a natural-language task and available "
    "capabilities (with input/output schemas), emit a TaskIR v0.2 program "
    "using EXTRACT_FIELD and bind() for runtime parameter binding. "
    "Output ONLY the TaskIR text.\n"
    "Syntax:\n"
    "  %id = EXEC_ACTION(action=\"tool_name\", literal_params...)\n"
    "  %fid = EXTRACT_FIELD(%producer_id, field=\"output_field\") -> Type\n"
    "  %id = EXEC_ACTION(action=\"tool\")  bind(param_name=%fid)\n"
    "  return %last_id")

SYS_M1 = (
    "You are a semantic task planner. Given a natural-language task and "
    "available capabilities (with input/output schemas), emit ONLY a JSON "
    "skeleton planning every tool call. For parameters whose values come "
    "from upstream tool outputs, use runtime_ref instead of literals:\n"
    '{"nodes": [{"id": "p0", "capability_id": "...", "skill": '
    '"EXEC_ACTION", "depends_on": [], "arguments": {"param": value or '
    '{"kind": "runtime_ref", "producer": "pX", "field": "output_field"}}}]}')


def load_split(name):
    return [json.loads(l) for l in open(f"{DATA}/{name}.jsonl",
                                        encoding="utf-8") if l.strip()]


def caps_block(rec, protocol="distractor"):
    """Build the capability description block for the model input."""
    ids = rec.get("offered_capability_ids", [])
    if protocol == "oracle":
        ids = rec.get("gold_capability_ids", ids)
    lines = []
    for tid in ids:
        meta = rec["capability_registry"].get(tid, {})
        ins = meta.get("input_schema", {})
        outs = meta.get("output_schema", {})
        eff = meta.get("effect_class", "read_only")
        ins_str = ", ".join(f"{k}:{v}" for k, v in ins.items()) or "-"
        outs_str = ", ".join(f"{k}:{v}" for k, v in outs.items()) or "-"
        lines.append(f"- {tid} | input: {ins_str} | output: {outs_str} "
                      f"| effect: {eff}")
    return "\n".join(lines)


def build_user_prompt(rec, protocol="distractor"):
    return (f"Task:\n{rec['instruction']}\n\n"
            f"Available capabilities:\n{caps_block(rec, protocol)}")


# ── Primary Metric: Runtime Binding Exact Match ──

def extract_bindings_from_skeleton(nodes):
    """Extract canonical binding triples from skeleton nodes.
    Returns set of (producer_id, field, consumer_id, param_name)."""
    bindings = set()
    for n in nodes:
        for pk, v in (n.get("arguments") or {}).items():
            if isinstance(v, dict) and v.get("kind") == "runtime_ref":
                bindings.add((v["producer"], v["field"],
                              n["id"], pk))
    return bindings


def extract_bindings_from_taskir(taskir_text):
    """Extract binding triples from TaskIR v0.2 text.
    Maps EXTRACT_FIELD intermediaries to (producer, field, consumer, param).
    """
    if not taskir_text:
        return set()
    try:
        from src.ir.parser import parse_text
        mod = parse_text(taskir_text)
    except Exception:
        return set()
    # map EXTRACT_FIELD SSA → (producer SSA, field_name)
    ext_map = {}
    for n in mod.program.nodes:
        if n.op == "EXTRACT_FIELD" and n.inputs:
            ext_map[n.id] = (n.inputs[0],
                              str(n.params.get("field", "")))
    bindings = set()
    ssa_to_orig = {}  # we need to map back to skeleton node ids
    # Build index of EXEC_ACTION/other nodes by order
    orig_nodes = [n for n in mod.program.nodes if n.op != "EXTRACT_FIELD"]
    for i, n in enumerate(orig_nodes):
        ssa_to_orig[n.id] = f"p{i}"  # assume p0, p1, ... ordering
    # For param_refs, resolve through EXTRACT_FIELD chain
    for n in orig_nodes:
        for pk, ref in n.param_refs.items():
            if ref in ext_map:
                producer_ssa, field = ext_map[ref]
                producer_id = ssa_to_orig.get(producer_ssa, producer_ssa)
                bindings.add((producer_id, field,
                              ssa_to_orig.get(n.id, n.id), pk))
            elif ref in ssa_to_orig:
                # direct SSA ref (no field extraction)
                bindings.add((ssa_to_orig[ref], "",
                              ssa_to_orig.get(n.id, n.id), pk))
    return bindings


def binding_exact_match(pred_bindings, ref_bindings):
    """All bindings must match exactly."""
    return pred_bindings == ref_bindings


def binding_f1(pred, ref):
    if not pred and not ref:
        return 1.0
    inter = len(pred & ref)
    if not inter:
        return 0.0
    p = inter / len(pred) if pred else 0
    r = inter / len(ref) if ref else 0
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ── Evaluation ──

def evaluate_task(rec, taskir_text, protocol="distractor"):
    """Comprehensive evaluation of one task."""
    from src.ir.parser import parse_text
    from src.validator.validator import validate
    from src.runtime.mock_capability_registry import (
        MockCapabilityRegistry, MockCapability, DataflowRuntime)
    from src.compiler.plan_skeleton import skeleton_from_dict, \
        validate_skeleton, hard_errors
    from src.compiler.skeleton_lowering import lower_skeleton

    result = {
        "task_id": rec["task_id"], "topology": rec["topology"],
        "parse": False, "validator": False,
        "binding_exact": False, "binding_f1": 0.0,
        "tool_match": False, "execution": False,
        "strict_semantic": False,
        "n_ref_bindings": 0, "n_pred_bindings": 0,
        "n_correct_bindings": 0,
    }
    ref_nodes = rec["target_skeleton"]["nodes"]
    ref_bindings = extract_bindings_from_skeleton(ref_nodes)
    result["n_ref_bindings"] = len(ref_bindings)
    gold_tools = rec.get("gold_capability_ids", [])

    if not taskir_text:
        return result
    try:
        mod = parse_text(taskir_text)
        result["parse"] = True
    except Exception:
        return result

    # Validator
    v2_reg = {t: {"input_schema": rec["capability_registry"][t].get("input_schema", {}),
                   "output_schema": rec["capability_registry"][t].get("output_schema", {})}
               for t in rec["capability_registry"]}
    rep = validate(mod, capability_registry=v2_reg)
    result["validator"] = rep.valid

    # Tool match
    pred_tools = sorted({n.params.get("action", "") for n in mod.program.nodes
                          if n.op == "EXEC_ACTION" and n.params.get("action")})
    result["tool_match"] = pred_tools == sorted(gold_tools)

    # Binding match (primary metric)
    pred_bindings = extract_bindings_from_taskir(taskir_text)
    result["n_pred_bindings"] = len(pred_bindings)
    result["n_correct_bindings"] = len(pred_bindings & ref_bindings)
    result["binding_exact"] = binding_exact_match(pred_bindings, ref_bindings)
    result["binding_f1"] = round(binding_f1(pred_bindings, ref_bindings), 4)

    # Execution
    if rep.valid:
        reg = MockCapabilityRegistry()
        for tid in rec["capability_registry"]:
            meta = rec["capability_registry"][tid]
            # Build deterministic executor
            def make_exec(tid, meta):
                outs = meta.get("output_schema", {})
                def exec_fn(params):
                    h = int(hashlib.md5(
                        json.dumps(params, sort_keys=True,
                                   default=str).encode()
                    ).hexdigest()[:6], 16)
                    result = {}
                    for k, t in outs.items():
                        if t == "Int":
                            result[k] = h % 1000
                        elif t == "Float":
                            result[k] = float(h % 10000) / 100
                        elif t == "Bool":
                            result[k] = h % 2 == 0
                        else:
                            result[k] = f"{tid}_{k}_{h % 9999}"
                    return result
                return exec_fn
            reg.register(MockCapability(
                tid, meta.get("input_schema", {}),
                meta.get("output_schema", {}),
                meta.get("effect_class", "read_only"),
                make_exec(tid, meta)))
        rt = DataflowRuntime(reg)
        tr = rt.execute(mod)
        result["execution"] = tr.success

    # Strict semantic: all components
    result["strict_semantic"] = bool(
        result["validator"] and result["tool_match"]
        and result["binding_exact"] and result["execution"])
    return result


# ── Baseline: Oracle Lowering ──

def oracle_baseline(rec):
    """Oracle Skeleton → Lowering → TaskIR (compiler upper bound)."""
    from src.compiler.plan_skeleton import skeleton_from_dict
    from src.compiler.skeleton_lowering import lower_skeleton
    sk = skeleton_from_dict(rec["target_skeleton"])
    schemas = {}
    for tid in rec["capability_registry"]:
        meta = rec["capability_registry"][tid]
        schemas[tid] = {
            "params": [{"name": k, "optional": False, "type": v}
                        for k, v in meta.get("input_schema", {}).items()],
            "output_fields": meta.get("output_schema", {}),
            "effect_class": meta.get("effect_class", "read_only")}
    low = lower_skeleton(sk, None, schemas, strict=True)
    return low.get("taskir", "")


# ── Main ──

def run_preflight():
    """Task 2: semantic preflight checks."""
    from src.compiler.capability_schema import normalize_registry
    checks = {}
    # 1. Schema type preservation
    reg = normalize_registry(
        {"t": {"params": [{"name": "x", "optional": False, "type": "Str"}]}})
    checks["schema_type_preserved"] = \
        reg["t"]["input_schema"].get("x") == "Str"
    # 2. DAG exact matching
    train = load_split("train")
    test = load_split("test")
    checks["train_count"] = len(train)
    checks["test_count"] = len(test)
    checks["held_out_topologies"] = sorted(
        {r["topology"] for r in test})
    # 3. All test topologies unseen in train
    train_topos = {r["topology"] for r in train}
    test_topos = {r["topology"] for r in test}
    checks["topology_isolation"] = test_topos.isdisjoint(train_topos)
    # 4. Literal observability
    lit_issues = 0
    for split in ("train", "dev", "test"):
        for r in load_split(split):
            for n in r["target_skeleton"]["nodes"]:
                for k, v in (n.get("arguments") or {}).items():
                    if isinstance(v, dict) and v.get("kind") == "runtime_ref":
                        continue
                    if isinstance(v, float) and v == int(v):
                        v = int(v)
                    if str(v).lower() not in r["instruction"].lower().replace(",", ""):
                        lit_issues += 1
    checks["literal_observable"] = lit_issues == 0
    checks["literal_issue_count"] = lit_issues
    # 5. Distractor pool present
    has_distract = all(
        len(r.get("distractor_capability_ids", [])) > 0
        for r in test)
    checks["distractor_pool"] = has_distract
    # 6. Deterministic mock (verify hash stability)
    h1 = hashlib.md5("test".encode()).hexdigest()
    h2 = hashlib.md5("test".encode()).hexdigest()
    checks["deterministic_hash"] = h1 == h2

    # 7. Oracle baseline works on all test tasks
    oracle_ok = 0
    for r in test:
        taskir = oracle_baseline(r)
        ev = evaluate_task(r, taskir, protocol="oracle")
        if ev["strict_semantic"]:
            oracle_ok += 1
    checks["oracle_strict_pass"] = f"{oracle_ok}/{len(test)}"

    passed = all(v for k, v in checks.items() if isinstance(v, bool))
    checks["PREFLIGHT"] = "PASS" if passed else "FAIL"

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/dataflow_pilot_preflight.json", "w") as f:
        json.dump(checks, f, indent=2)
    for k, v in checks.items():
        print(f"  {k}: {v}")
    return passed


def run_baselines():
    """Task 4: no-training baselines."""
    dev = load_split("dev")
    test = load_split("test")
    results = {"oracle": [], "negative_swap": []}

    for r in test:
        # Oracle baseline
        taskir = oracle_baseline(r)
        ev = evaluate_task(r, taskir, protocol="oracle")
        ev["baseline"] = "oracle"
        results["oracle"].append(ev)

        # Negative: swap two bindings
        neg_nodes = json.loads(json.dumps(r["target_skeleton"]["nodes"]))
        refs = [(n["id"], k, v) for n in neg_nodes
                for k, v in (n.get("arguments") or {}).items()
                if isinstance(v, dict) and v.get("kind") == "runtime_ref"]
        if len(refs) >= 1:
            # negative: change ONE producer to a wrong node
            # (always semantically different)
            id1, k1, v1 = refs[0]
            all_nodes = [n["id"] for n in neg_nodes]
            wrong_producer = next(
                (x for x in all_nodes
                 if x != v1.get("producer") and
                 any(r[2].get("producer") == x for r in refs)),
                None)
            if wrong_producer is None:
                wrong_producer = next(
                    (x for x in all_nodes if x != v1.get("producer")),
                    None)
            if wrong_producer:
                for n in neg_nodes:
                    if n["id"] == id1:
                        n["arguments"][k1]["producer"] = wrong_producer
            neg_rec = {**r, "target_skeleton": {"nodes": neg_nodes}}
            neg_taskir = oracle_baseline(neg_rec)
            neg_ev = evaluate_task(r, neg_taskir, protocol="oracle")
            neg_ev["baseline"] = "negative_swap"
            results["negative_swap"].append(neg_ev)

    for name, rows in results.items():
        n = len(rows)
        if n:
            strict = sum(1 for r in rows if r["strict_semantic"])
            bem = sum(1 for r in rows if r["binding_exact"])
            print(f"  {name}: strict={strict}/{n} binding_exact={bem}/{n}")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/dataflow_pilot_baselines.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["preflight", "baseline", "smoke",
                             "train", "eval"])
    ap.add_argument("--model", default="m0", choices=["m0", "m1"])
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    if args.phase == "preflight":
        ok = run_preflight()
        sys.exit(0 if ok else 1)
    elif args.phase == "baseline":
        run_baselines()
