"""Phase 9B-3: three-arm training on runtime_dataflow_v3 (server entry).

Adaptations vs train_three_arm (v2):
  - prompt schema rendered via schema_render_v2 (nested List[Struct]...)
  - lowering schemas built from the task registry in the PROVEN format
    (same adapter as execute_taskir_official, which validated 42/43)
  - M0-T/M0-J targets computed by strict lowering here (v3 records carry
    no precomputed m0_taskir)
  - M1 target = skeleton JSON, no task_id (template-memory fix)
  - offered ids: distractor protocol = full registry (v3 bakes in 2
    distractors); oracle protocol = tools used in the target skeleton

Usage (server, from /root/compile-anything):
  python3 scripts/train_v3.py --seeds 44
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.compiler.plan_skeleton import skeleton_from_dict
from src.compiler.skeleton_lowering import lower_skeleton
from src.ir.parser import parse_text
from src.ir.taskir import module_to_dict
from src.eval.schema_render_v2 import render_tool
from transformers import AutoTokenizer  # server-side

import os
DATA = os.environ.get("V3DATA", "data/runtime_dataflow_v3")
RUNS = os.environ.get("V3RUNS", "runs_v3")
_FLAT = {"list": "List[Any]", "struct": "Struct", "dict": "Dict",
         "scalar_root": "Any"}


def _flat(v):
    if isinstance(v, str):
        return v
    k = v.get("kind")
    if k == "scalar":
        return v.get("type", "Any")
    return _FLAT.get(k, "Any")


def schemas_for(rec):
    return {t: {"input_schema": {k: _flat(v)
                                  for k, v in m["input_schema"].items()},
                "output_schema": m["output_schema"],
                "effect_class": m.get("effect_class", "read_only"),
                "params": [{"name": k, "optional": False, "type": _flat(v)}
                           for k, v in m["input_schema"].items()],
                "output_fields": {}}
            for t, m in rec["capability_registry"].items()}


def caps_block_v2(rec, protocol):
    gold = {n["capability_id"] for n in rec["target_skeleton"]["nodes"]}
    ids = sorted(gold) if protocol == "oracle" else \
        sorted(rec["capability_registry"].keys())
    lines = []
    for tid in ids:
        m = rec["capability_registry"][tid]
        lines.append(render_tool(tid, {
            "input_schema": {k: _flat(v)
                             for k, v in m["input_schema"].items()},
            "output_schema": m["output_schema"],
            "effect_class": m.get("effect_class", "read_only")}))
    return "\n".join(lines)


def build_user(rec, protocol):
    return (f"Task:\n{rec['instruction']}\n\n"
            f"Available capabilities:\n{caps_block_v2(rec, protocol)}")


def build_targets(rec):
    sk = skeleton_from_dict({"task_id": "t",
                             "nodes": rec["target_skeleton"]["nodes"]})
    low = lower_skeleton(sk, None, schemas_for(rec), strict=True)
    m0t = low.get("taskir") or ""
    m0j = None
    if m0t:
        try:
            m0j = json.dumps(module_to_dict(parse_text(m0t)),
                             ensure_ascii=False)
        except Exception:
            m0j = None
    m1 = json.dumps(rec["target_skeleton"], ensure_ascii=False)
    return m0t, m0j, m1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="44")
    ap.add_argument("--protocol", default="distractor")
    ap.add_argument("--arms", default="m0t,m0j,m1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    from scripts.train_three_arm import train_arm  # needs torch/peft

    train = [json.loads(l) for l in open(f"{DATA}/train.jsonl",
                                         encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows_by_arm = {"m0t": [], "m0j": [], "m1": []}
    for rec in train:
        m0t, m0j, m1 = build_targets(rec)
        user = build_user(rec, args.protocol)
        if m0t:
            rows_by_arm["m0t"].append({"user": user, "target": m0t})
        if m0j:
            rows_by_arm["m0j"].append({"user": user, "target": m0j})
        rows_by_arm["m1"].append({"user": user, "target": m1})

    logs = {}
    arms = [a.strip() for a in args.arms.split(",")]
    for seed in seeds:
        for arm in arms:
            out = f"{RUNS}/{arm}_s{seed}"
            os.makedirs(out, exist_ok=True)
            logs[f"{arm}_s{seed}"] = train_arm(
                arm, rows_by_arm[arm], tok, seed, out)
            with open(f"{RUNS}/train_logs.json", "w") as f:
                json.dump(logs, f, indent=1)
    print(json.dumps(logs, indent=1))


if __name__ == "__main__":
    main()
