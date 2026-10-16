"""Phase 9B-2.5 T7: reproduce the 8D-2B generation-audit numbers.

Claims to verify (advisor clean-clone reproduction):
  1. M0-J cap hits: oracle 84 + distractor 91 = 175 of 218 parse failures
  2. Of the 175 cap-hit failures, 166 had already generated MORE nodes
     than the gold target before truncation
  3. M0-T targets 134/134 carry '; task: <id>'; M1 targets 134/134 carry
     task_id; the id never appears in the instruction
  4. Predictions leak train ids: M0-T 132/132 per protocol, M1 264/264
  5. Gold M0-J target length (Qwen2.5 tokenizer): attempted if the
     tokenizer is locally available, otherwise reported as N/A.

Output: results/phase9/three_arm_generation_audit.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results/phase8d/three_arm_raw_predictions.jsonl"
CORPUS = REPO / "data/runtime_dataflow_v2"
OUT = REPO / "results/phase9/three_arm_generation_audit.json"


def load_corpus():
    tasks = {}
    for split in ("train", "dev", "test"):
        p = CORPUS / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            t = json.loads(line)
            tasks[t["task_id"]] = (t, split)
    return tasks


def count_json_nodes(text):
    """Best-effort node count in generated M0-J JSON AST."""
    try:
        m = re.search(r"\{.*", text, re.S)
        if not m:
            return 0
        d = json.loads(m.group(0) + ("}" * max(0, text.count("{") - text.count("}"))))
        prog = d.get("program") or d.get("module") or d
        nodes = prog.get("nodes") if isinstance(prog, dict) else prog
        return len(nodes or [])
    except Exception:
        # fallback: count node-like SSA lines / dict entries
        return len(re.findall(r'"op"\s*:', text)) or \
            len(re.findall(r"%\w+\s*=", text))


def main():
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    corpus = load_corpus()

    # 1+2: M0-J cap hits and overgeneration
    m0j = [r for r in rows if r["arm"] == "M0-J"]
    parse_fail = [r for r in m0j if not r.get("parse")]
    cap_hits = [r for r in parse_fail if r.get("gen_tokens", 0) >= 698]
    by_proto = Counter(r["protocol"] for r in cap_hits)
    overgen = 0
    for r in cap_hits:
        gold_task = corpus.get(r["task_id"])
        gold_nodes = len(gold_task[0]["target_skeleton"]["nodes"]) \
            if gold_task else None
        if gold_nodes is not None and count_json_nodes(r["raw"]) > gold_nodes:
            overgen += 1

    # 3: target hygiene (from corpus train; v2 was train-only SFT)
    train_ids = [tid for tid, (_, sp) in corpus.items() if sp == "train"]
    m0t_task_header = m1_taskid = instr_contains = 0
    n_train = len(train_ids)
    for tid in train_ids:
        t, _ = corpus[tid]
        instr = json.dumps(t.get("instruction", ""))
        if tid in instr:
            instr_contains += 1
    # targets: derive from the skeleton (the training target format)
    for tid in train_ids:
        t, _ = corpus[tid]
        sk = t.get("target_skeleton", {})
        if tid.startswith("v2"):
            m1_taskid += 1 if sk.get("task_id") == tid else 0

    # 4: train-id leakage in predictions
    m0t_leak = Counter()
    m1_leak = 0
    m1_total = 0
    for r in rows:
        pred_id = None
        m = re.search(r";\s*task:\s*(\S+)", r.get("raw", ""))
        if m:
            pred_id = m.group(1)
        m2 = re.search(r'"task_id"\s*:\s*"([^"]+)"', r.get("raw", ""))
        if m2:
            pred_id = pred_id or m2.group(1)
        train_set = set(train_ids)
        leaked = pred_id is not None and pred_id in train_set \
            and pred_id != r["task_id"]
        if r["arm"] == "M0-T":
            m0t_leak[(r["protocol"], leaked)] += 1
        elif r["arm"] == "M1":
            m1_total += 1
            m1_leak += bool(leaked)

    # 5: gold M0-J target token length (needs Qwen tokenizer; optional)
    tok_info = {"available": False}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        lens = []
        test_ids = [tid for tid, (_, sp) in corpus.items() if sp == "test"]
        for tid in test_ids:
            t, _ = corpus[tid]
            nodes = t["target_skeleton"]["nodes"]
            # M0-J target is the JSON AST serialization
            payload = json.dumps({"nodes": nodes}, ensure_ascii=False)
            lens.append(len(tok(payload)["input_ids"]))
        if lens:
            tok_info = {"available": True, "n": len(lens),
                        "min": min(lens), "max": max(lens),
                        "median": sorted(lens)[len(lens)//2]}
    except Exception as e:
        tok_info = {"available": False, "reason": str(e)[:120]}

    out = {
        "m0j": {
            "n_details": len(m0j),
            "parse_failures": len(parse_fail),
            "cap_hits_total": len(cap_hits),
            "cap_hits_by_protocol": dict(by_proto),
            "cap_hit_overgenerated_vs_gold": overgen,
        },
        "target_hygiene": {
            "n_train": n_train,
            "instruction_contains_task_id": instr_contains,
            "m1_targets_with_task_id": m1_taskid,
        },
        "prediction_leakage": {
            "m0t_by_protocol_leak": {f"{p}|leak={k}": v
                                     for (p, k), v in m0t_leak.items()},
            "m1_leaked": m1_leak, "m1_total": m1_total,
        },
        "gold_m0j_token_length": tok_info,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
