"""Phase 7C-2.1 Task 5: verify the fast-path mask is byte-identical and
measure the new end-to-end latency/throughput.

Re-runs selected confirmation samples (s43, arms G1/G2) with the optimized
processor and diffs the generated text against the FROZEN results stored in
results/phase7/online_large_runs.jsonl (produced by the pre-optimization
code, deterministic greedy). Reports per-arm: identical count, tokens/s
(new), P50/P95 latency (new) alongside the stored latencies (old).
"""
import json, os, sys, time, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch

from src.compiler.prompt_format import SYSTEM_PROMPT
from src.compiler.online_constrained_decoder import generate_constrained


def load_tasks():
    refs = {}
    for line in open("data/compiler_corpus_v4/test.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        refs[r["id"]] = r.get("plan_text") or r.get("plan_target") or ""
    tasks = {}
    for line in open("data/phase6b_resolver/common_eval_tasks.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        gold = [c for c in t.get("candidates", []) if c.get("is_gold")]
        if not gold:
            continue
        tasks[t["id"]] = {
            "task": t["task"],
            "cap_blocks": "\n\n".join(c.get("block", "") for c in gold[:5]),
        }
    return tasks


def main(n=12, seed="s43", arms=("G1", "G2")):
    frozen = {}
    for line in open("results/phase7/online_large_runs.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["seed"] == seed and r["set"] == "confirmation":
            frozen[(r["id"], r["arm"])] = r

    ids = sorted({k[0] for k in frozen})[:n]
    tasks = load_tasks()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(
        base, f"runs/phase6b/composer_C2_{seed}/final")
    model.eval()

    levels = {"G0": 0, "G1": 1, "G2a": 2, "G2": 3, "G3": 4}
    report = {}
    for arm in arms:
        level = levels[arm]
        identical = 0
        new_lat, old_lat, tok_s = [], [], []
        for tid in ids:
            s = tasks[tid]
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",
                  "content": f"{s['task']}\n\nSelected capabilities:\n{s['cap_blocks']}"}],
                tokenize=False, add_generation_prompt=True)
            t0 = time.time()
            text, _stats, _pl = generate_constrained(
                model, tok, prompt, level, max_new_tokens=512)
            dt = time.time() - t0
            new_lat.append(round(dt, 2))
            ntok = len(tok(text, add_special_tokens=False)["input_ids"])
            tok_s.append(ntok / dt)
            old = frozen.get((tid, arm))
            old_lat.append(old["latency_s"] if old else None)
            if old is not None and old["text"] == text:
                identical += 1
            elif old is not None:
                print(f"  DIFF {tid} {arm}:")
                print(f"    old tail: {old['text'][-100:]!r}")
                print(f"    new tail: {text[-100:]!r}")
        checked = sum(1 for tid in ids if (tid, arm) in frozen)
        def pxx(v, q):
            v = sorted(x for x in v if x is not None)
            return v[min(len(v) - 1, int(q * len(v)))] if v else None
        report[arm] = {
            "n_checked": checked, "byte_identical": identical,
            "new_latency_p50": pxx(new_lat, .5),
            "new_latency_p95": pxx(new_lat, .95),
            "old_latency_p50": pxx(old_lat, .5),
            "old_latency_p95": pxx(old_lat, .95),
            "new_tokens_per_s_mean": round(sum(tok_s) / len(tok_s), 1),
            "speedup_p50": (round(pxx(old_lat, .5) / pxx(new_lat, .5), 2)
                            if pxx(old_lat, .5) and pxx(new_lat, .5) else None),
        }
        r = report[arm]
        print(f"{arm}: identical {identical}/{checked} | "
              f"P50 old->new {r['old_latency_p50']}->{r['new_latency_p50']}s "
              f"(x{r['speedup_p50']}) | {r['new_tokens_per_s_mean']} tok/s")
    with open("results/phase7/fast_mask_verification.json", "w") as f:
        json.dump({"seed": seed, "n": len(ids), "report": report}, f, indent=2)
    print("saved results/phase7/fast_mask_verification.json")
    print("VERIFY-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", default="s43")
    ap.add_argument("--arms", default="G1,G2")
    args = ap.parse_args()
    main(args.n, args.seed, tuple(args.arms.split(",")))
