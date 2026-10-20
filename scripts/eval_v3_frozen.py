"""Phase 9B-3: evaluate trained arms on the FROZEN formal test (server).

For each arm adapter (runs_v3/<arm>_s<seed>/final) and protocol
(oracle / distractor), generates on the frozen A-test (15 retail) and
B-test (3 airline) tasks and saves FULL untruncated raw predictions
(9B-2.2 contract: raw, raw_preview, raw_log_truncated,
hit_generation_cap). Per-arm max_new_tokens from the preflight token
budget (2x gold max). Scoring with scorer v3 happens locally
(score_tau3_offline) — the server stays lean (no tau2 deps).

Usage (server, from /root/compile-anything):
  python3 scripts/eval_v3_frozen.py --seed 44
"""
import argparse
import gc
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, ".")
from src.eval.compile_input_v2 import render_compile_input
from src.eval.schema_render_v2 import render_tool
from scripts.train_v3 import _flat
from scripts.train_three_arm import SYS_M0T, SYS_M0J, SYS_M1
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

GOLD = "results/phase9/tau3_gold_compiled_v2_2.json"
FREEZE = "results/phase9/tau3_gold_v2_2_freeze_manifest.json"
REG = "results/phase9/verified_registry_manifest.json"
TAU2_TASKS = "/root/tau2_tasks"   # raw tasks dumped locally below
_F = float(__import__("os").environ.get("EVAL_FACTOR", "2"))
MAX_NEW = {k: int(v * _F) for k, v in
          {"M0-T": 864, "M0-J": 2396, "M1": 1070}.items()}


def build_eval_records():
    gold = json.loads(open(GOLD, encoding="utf-8").read())
    freeze = json.loads(open(FREEZE, encoding="utf-8").read())
    reg = json.loads(open(REG, encoding="utf-8").read())
    testA = freeze["summary"]["condition_A_test"]
    testB = freeze["summary"]["condition_B_test"]
    want = set(testA) | set(testB)
    recs = []
    for t in gold["tasks"]:
        if t["task_id"] not in want:
            continue
        dom = t["domain"]
        sid = t.get("source_task_id") or t["task_id"].rsplit("_", 1)[1]
        raw = json.load(open(f"{TAU2_TASKS}/{dom}_tasks.json", encoding="utf-8"))
        rt = next(x for x in raw if str(x["id"]) == str(sid))
        instruction = render_compile_input(rt.get("user_scenario"))
        # skeleton view of the gold program (same mapping as the executor)
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
        gold_tools = sorted({n["capability_id"] for n in nodes})
        dom_reg = {k.split(".", 1)[1]: v for k, v in reg["tools"].items()
                   if k.startswith(dom + ".")}
        recs.append({"task_id": t["task_id"], "condition":
                     "A" if t["task_id"] in testA else "B",
                     "domain": dom, "instruction": instruction,
                     "nodes": nodes, "gold_tools": gold_tools,
                     "dom_reg": dom_reg, "gold_program": t["program"]})
    return recs


def caps_block(rec, protocol):
    if protocol == "oracle":
        ids = rec["gold_tools"]
    else:
        others = [t for t in rec["dom_reg"] if t not in rec["gold_tools"]]
        random.seed(44)
        ids = rec["gold_tools"] + random.sample(
            others, min(2, len(others)))
    lines = []
    for tid in sorted(ids):
        m = rec["dom_reg"][tid]
        lines.append(render_tool(tid, {
            "input_schema": {k: _flat(v)
                             for k, v in m["input_schema"].items()},
            "output_schema": m["output_schema"],
            "effect_class": m.get("effect_class", "read_only")}))
    return "\n".join(lines)


def build_user(rec, protocol):
    return (f"Task:\n{rec['instruction']}\n\n"
            f"Available capabilities:\n{caps_block(rec, protocol)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    recs = build_eval_records()
    print(f"eval tasks: {len(recs)} "
          f"(A: {sum(1 for r in recs if r['condition'] == 'A')}, "
          f"B: {sum(1 for r in recs if r['condition'] == 'B')})")

    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    out_path = f"results/phase9/{__import__('os').environ.get('EVAL_TAG', 'v3')}_frozen_predictions_s{args.seed}.jsonl"
    os.makedirs("results/phase9", exist_ok=True)
    fout = open(out_path, "w", encoding="utf-8")

    _arms = __import__("os").environ.get("EVAL_ARMS", "M0-T,M0-J,M1").split(",")
    _sys = {"M0-T": SYS_M0T, "M0-J": SYS_M0J, "M1": SYS_M1}
    for arm in _arms:
        sys_p = _sys[arm]
        adapter = f"{__import__('os').environ.get('V3RUNS', 'runs_v3')}/{arm.lower().replace('-', '')}_s{args.seed}/final"
        if not os.path.exists(adapter):
            print(f"skip {arm} (no adapter)")
            continue
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        for protocol in ("oracle", "distractor"):
            for rec in recs:
                user = build_user(rec, protocol)
                prompt = tok.apply_chat_template(
                    [{"role": "system", "content": sys_p},
                     {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True)
                ids = tok(prompt, return_tensors="pt", truncation=True,
                          max_length=2048).to(model.device)
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(
                        **ids, max_new_tokens=MAX_NEW[arm], do_sample=False,
                        temperature=None, pad_token_id=tok.pad_token_id)
                text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                ntok = int(out.shape[1] - ids["input_ids"].shape[1])
                row = {"task_id": rec["task_id"], "condition": rec["condition"],
                       "arm": arm, "protocol": protocol, "seed": args.seed,
                       "gen_tokens": ntok,
                       "hit_generation_cap": ntok >= MAX_NEW[arm] - 2,
                       "latency_s": round(time.time() - t0, 2),
                       "raw": text,
                       "raw_preview": text[:2000],
                       "raw_log_truncated": len(text) > 2000}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
            print(f"  {arm}/{protocol} done", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    fout.close()
    print("saved", out_path)


if __name__ == "__main__":
    main()
