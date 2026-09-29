"""Build frozen evaluation suites (Task 10/11) + external-eval manifest
(Task 15). IDs only — raw content never copied; suites are frozen
(post-training sample selection is impossible by construction).

    python scripts/build_eval_suites.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.adapters import (load_bfcl, load_tau3, load_agentboard,   # noqa
                               load_rtl_repo, load_bird, load_toolbench)
from src.eval.oracle_bfcl import oracle as bfcl_oracle

EXT = ROOT / "data" / "external_benchmarks"
SUITES = EXT / "eval_suites"

# three-layer suite structure (paper main tables)
LAYER = {
    "bfcl_v4": "A_function_calling",
    "toolbench_full": "A_function_calling",
    "agentboard_tools_tool_query": "A_function_calling",
    "tau3_bench": "B_stateful_agent",
    "agentboard_tools_tool_operation": "B_stateful_agent",
    "agentboard_webshop": "B_stateful_agent",
    "bird_mini_dev": "C_domain_transfer",
    "rtl_repo": "C_domain_transfer",
}


def case_hash(s) -> str:
    payload = json.dumps({"i": s.instruction[:500],
                          "a": str(s.reference_actions)[:300],
                          "m": s.metadata.get("category",
                                              s.metadata.get("domain",
                                              s.metadata.get("split", "")))},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_suite(name, samples, extra=None):
    cases = [{"case_id": s.case_id,
              "category": (s.metadata.get("category")
                           or s.metadata.get("domain")
                           or s.metadata.get("split")
                           or s.benchmark),
              "sha": case_hash(s)} for s in samples]
    suite = {"benchmark": name, "layer": LAYER.get(name, ""),
             "case_count": len(cases), "cases": cases, **(extra or {})}
    return suite


def main() -> int:
    SUITES.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                            capture_output=True, text=True).stdout.strip()
    manifest = {}

    def emit(name, samples, raw_rev, extra=None):
        suite = build_suite(name, samples, extra)
        p = SUITES / f"{name}.json"
        p.write_text(json.dumps(suite, indent=1, ensure_ascii=False),
                     encoding="utf-8")
        manifest[name] = {
            "layer": suite["layer"], "case_count": suite["case_count"],
            "raw_revision": raw_rev,
            "eval_suite_sha256": hashlib.sha256(
                p.read_bytes()).hexdigest(),
            "adapter_version": commit}
        print(f"{name:28s} {suite['case_count']:7d} -> {p.name}")

    # ---- Suite A / B / C sources ----
    bfcl = load_bfcl()
    emit("bfcl_v4", bfcl, "gorilla@6ea57973",
         extra={"note": "4697 raw lines = 4696 case records + 1 "
                        "format_sensitivity id-list document"})

    tau3 = load_tau3()
    emit("tau3_bench", tau3, "codesque16/tau3-bench@f0a0173e")

    ab_tq = load_agentboard(tasks=["tool-query"])
    ab_to = load_agentboard(tasks=["tool-operation"])
    ab_ws = load_agentboard(tasks=["webshop"])
    emit("agentboard_tools_tool_query", ab_tq, "AgentBoard@bb7255e2")
    emit("agentboard_tools_tool_operation", ab_to, "AgentBoard@bb7255e2")
    emit("agentboard_webshop", ab_ws, "AgentBoard@bb7255e2")

    rtl_test = load_rtl_repo("test")
    emit("rtl_repo", rtl_test, "ahmedallam/RTL-Repo (test split)")

    bird = load_bird()
    emit("bird_mini_dev", bird, "birdsql/bird_mini_dev")

    tb = load_toolbench()
    emit("toolbench_full", tb, "modelscope swift/ToolBench")

    # ---- Task 13: BFCL oracle representability (derived, eval-only) ----
    counts = {"full": 0, "partial": 0, "none": 0}
    reasons = {}
    by_cat = {}
    oracle_dir = EXT / "derived_oracle" / "bfcl_v4"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for s in bfcl:
        r = bfcl_oracle(s)
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        cat = s.metadata.get("category", "?")
        by_cat.setdefault(cat, {"full": 0, "partial": 0, "none": 0})
        by_cat[cat][r["status"]] += 1
        for reason in r["reasons"]:
            key = reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        if r.get("module") is not None and n_saved < 50:   # spot samples
            (oracle_dir / f"{s.case_id}.taskir.json").write_text(
                json.dumps({"case_id": s.case_id,
                            "status": r["status"],
                            "taskir": {"nodes": [
                                {"id": n.id, "op": n.op}
                                for n in r["module"].program.nodes]},
                            "reasons": r["reasons"]},
                           indent=1, ensure_ascii=False), encoding="utf-8")
            n_saved += 1
    (EXT / "derived_oracle" / "bfcl_v4_representability.json").write_text(
        json.dumps({"total": len(bfcl), "counts": counts,
                    "reasons": reasons, "by_category": by_cat},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"BFCL oracle: {counts} reasons={reasons}")

    # ---- Task 15 manifest ----
    ext_manifest = json.loads((EXT / "MANIFEST.json").read_text(
        encoding="utf-8")) if (EXT / "MANIFEST.json").exists() else {}
    out = {"generated": commit, "suites": manifest,
           "oracle_bfcl": {"total": len(bfcl), **counts},
           "raw_benchmark_manifest": {
               k: {"revision": v.get("commit", v.get("version", "")),
                   "status": v.get("status"),
                   "file_count": v.get("file_count")}
               for k, v in ext_manifest.items() if isinstance(v, dict)},
           "principle": "official test sets + official metrics + frozen "
                        "case ids; adapter outputs can never enter SFT "
                        "(external_guard)"}
    mdir = ROOT / "experiments" / "external_eval"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "manifest.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest -> experiments/external_eval/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
