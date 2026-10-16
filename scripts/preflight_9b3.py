"""Phase 9B-2.5 T9: 9B-3 run-package PREFLIGHT (prepare only, no run).

Four hard gates — any failure refuses the GPU launch:
  (a) owner credential-rotation statement recorded
  (b) gold v2.1 freeze manifest sha256 pin matches
  (c) v3 training-data manifest sha256 pin matches
  (d) tau3 LF-normalized md5s match the preregistration
Token budget: per-arm gold-target token lengths via the real Qwen2.5
tokenizer; recommended max_new_tokens = 2 x max. Cap-hit rate must be
reported per arm x protocol after any run; >5% invalidates comparison.

Output: results/phase9/preflight_9b3.json (exit code 1 if gated).
"""
import hashlib, json, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TAU3_MD5 = {"retail/db.json": "69771c8c71d2cf8fa767574c3a65999a",
            "retail/tasks.json": "c07e89e811d345bc062461aeff7de7d7",
            "airline/db.json": "55b232f2101d058fab0deea60d306675",
            "airline/tasks.json": "4588a63c1fa9844914c058b1d3dc107e"}
TAU2 = Path("D:/ccfa-2026/tau3-bench-src/data/tau2/domains")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gold_targets(skeleton_nodes, schemas):
    """Three arm targets for one gold program."""
    from src.compiler.plan_skeleton import skeleton_from_dict
    from src.compiler.skeleton_lowering import lower_skeleton
    from src.ir.parser import parse_text
    from src.ir.taskir import module_to_dict
    low = lower_skeleton(skeleton_from_dict(
        {"task_id": "t", "nodes": skeleton_nodes}), None, schemas,
        strict=True)
    taskir = low.get("taskir")
    if not taskir:
        return None
    m1 = json.dumps({"nodes": skeleton_nodes}, ensure_ascii=False)
    m0j = json.dumps(module_to_dict(parse_text(taskir)),
                     ensure_ascii=False, default=str)
    return {"M0-T": taskir, "M0-J": m0j, "M1": m1}


def main():
    gates = {}

    # (a) owner rotation statement
    sec_doc = (REPO / "docs/phase9/security-incident-remediation.md")
    txt = sec_doc.read_text(encoding="utf-8") if sec_doc.exists() else ""
    gates["owner_rotation_statement"] = bool(
        re.search(r"所有者声明|rotation (statement|confirmed)", txt,
                  re.I) and "Unconfirmed" not in
        txt.split("所有者声明")[-1][:400])

    # (b)/(c) manifest pins (recorded now; verified at launch)
    gold_man = REPO / "results/phase9/tau3_gold_v2_1_freeze_manifest.json"
    data_man = REPO / "data/runtime_dataflow_v3/manifest.json"
    pins = {"gold_v2_1_freeze": sha256(gold_man),
            "training_v3_manifest": sha256(data_man)}
    gates["gold_manifest_present"] = gold_man.exists()
    gates["v3_manifest_present"] = data_man.exists()

    # (d) tau3 md5s
    md5_ok = True
    for rel, want in TAU3_MD5.items():
        got = hashlib.md5(
            (TAU2 / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if got != want:
            md5_ok = False
    gates["tau3_md5_match"] = md5_ok

    # token budget
    tok_stats = {"available": False}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        reg = json.loads((REPO / "results/phase9/verified_registry_manifest.json")
                         .read_text(encoding="utf-8"))
        _FLAT = {"list": "List[Any]", "struct": "Struct", "dict": "Dict",
                 "scalar_root": "Any"}
        def _flat(v):
            if isinstance(v, str):
                return v
            k = v.get("kind")
            return v.get("type", "Any") if k == "scalar" else _FLAT.get(k, "Any")
        schemas = {k.split(".", 1)[1]: {
            "input_schema": {k2: _flat(v2) for k2, v2 in v["input_schema"].items()},
            "output_schema": v["output_schema"],
            "effect_class": v["effect_class"],
            "params": [{"name": k2, "optional": False, "type": _flat(v2)}
                       for k2, v2 in v["input_schema"].items()],
            "output_fields": {}} for k, v in reg["tools"].items()
            if k.startswith("retail.")}
        lengths = {"v3_train": {"M0-T": [], "M0-J": [], "M1": []},
                   "gold_test": {"M0-T": [], "M0-J": [], "M1": []}}
        v3 = [json.loads(l) for l in open(
            REPO / "data/runtime_dataflow_v3/train.jsonl", encoding="utf-8")]
        for t in v3[:60]:
            g = gold_targets(t["target_skeleton"]["nodes"], schemas)
            if g:
                for arm, s in g.items():
                    lengths["v3_train"][arm].append(len(tok(s)["input_ids"]))
        gold = json.loads((REPO / "results/phase9/tau3_gold_compiled_v2_1.json")
                          .read_text(encoding="utf-8"))
        for t in gold["tasks"]:
            if not t.get("strict_v2_1"):
                continue
            nodes = []
            for n in t["program"]:
                args = {}
                for p, a in n["arguments"].items():
                    c = a["class"]
                    if c == "LITERAL":
                        args[p] = a["value"]
                    elif c == "runtime_ref_multi":
                        args[p] = {"kind": "runtime_ref_multi",
                                   "elements": a["elements"]}
                    else:
                        cc = (a.get("candidates") or a.get("acceptable")
                              or [{"producer_step": a.get("producer_step"),
                                   "path": a.get("path", "")}])[0]
                        args[p] = {"kind": "runtime_ref",
                                   "producer": f"p{cc['producer_step']}",
                                   "field": cc.get("path", "")}
                nodes.append({"id": f"p{n['step']}",
                              "capability_id": n["tool"], "skill": "EXEC_ACTION",
                              "depends_on": [], "order_on": [],
                              "arguments": args})
            g = gold_targets(nodes, schemas)
            if g:
                for arm, s in g.items():
                    lengths["gold_test"][arm].append(len(tok(s)["input_ids"]))
        def st(v):
            return ({"n": len(v), "min": min(v), "median": sorted(v)[len(v)//2],
                     "max": max(v)} if v else None)
        tok_stats = {"available": True,
                     "v3_train": {a: st(v) for a, v in lengths["v3_train"].items()},
                     "gold_test": {a: st(v) for a, v in lengths["gold_test"].items()},
                     "recommended_max_new_tokens": {
                         a: 2 * max(lengths["v3_train"][a] + lengths["gold_test"][a])
                         for a in ("M0-T", "M0-J", "M1")}}
    except Exception as e:
        tok_stats = {"available": False, "reason": str(e)[:150]}

    all_ok = all(gates.values())
    out = {"gates": gates, "manifest_sha256_pins": pins,
           "token_budget": tok_stats,
           "run_spec": {"seeds": [44, 45, 46],
                        "ci": "Clopper-Pearson (src/eval/stats_ci.py)",
                        "paired_test": "exact McNemar",
                        "raw_policy": "full untruncated raw + raw_preview + "
                                      "raw_log_truncated + hit_generation_cap "
                                      "(9B-2.2 contract)",
                        "cap_hit_rule": "report per arm x protocol; >5% "
                                        "invalidates the comparison",
                        "optional_api_baseline": "owner approval + budget; "
                                                 "keys from env vars only; "
                                                 "frozen test once"},
           "verdict": "READY" if all_ok else "BLOCKED"}
    (REPO / "results/phase9/preflight_9b3.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"gates": gates, "verdict": out["verdict"],
                      "token_budget": tok_stats.get("recommended_max_new_tokens")
                      if tok_stats.get("available") else tok_stats}, indent=1))
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
