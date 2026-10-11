"""Phase 7C-1 Task 0: archive per-sample raw generations for 7C-0.1 (30x3).

Greedy decoding is deterministic, so re-running reproduces the exact
completions evaluated in constrained_generation_v2.json. This archives, per
sample: prompt hash, raw completion, extracted TaskIR text, parse/validator
errors, and D0-D3 per-sample outcomes (incl. assembled module text).

Writes results/phase7/constrained_generation_v2_samples.jsonl (NEW file —
does not touch constrained_generation_v2.json).
"""
import json, os, sys, hashlib, time

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.ir.parser import parse_text
from src.validator.validator import validate
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.constrained_assembly import apply_constraints, node_key

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_constrained_v2 import load_test_data, generate_for_seed, op_sequence, POLICY_OPS

ARMS = [("D0_normal", 0), ("D1_grammar", 1), ("D2_typed_ssa", 2), ("D3_control", 3)]


def prompt_hash(user_text):
    return hashlib.sha256(
        (SYSTEM_PROMPT + "\x00" + user_text).encode("utf-8")).hexdigest()[:16]


def main(n=30, seeds=("s42", "s43", "s44")):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = load_test_data(n)
    out_path = "results/phase7/constrained_generation_v2_samples.jsonl"
    os.makedirs("results/phase7", exist_ok=True)
    f = open(out_path, "w", encoding="utf-8")

    for seed in seeds:
        if not os.path.exists(f"runs/phase6b/composer_C2_{seed}/final"):
            print(f"[skip {seed}]")
            continue
        print(f"=== {seed} ===", flush=True)
        outputs, lat = generate_for_seed(seed, data, tok)

        for s, raw in zip(data, outputs):
            user = f"{s['task']}\n\nSelected capabilities:\n{s['cap_blocks']}"
            rec = {
                "id": s["id"], "seed": seed,
                "prompt_hash": prompt_hash(user),
                "raw_completion": raw,
                "taskir_text": extract_taskir_text(raw),
                "parse_error": None, "validator_errors": [],
                "arms": {},
            }
            try:
                mod = parse_text(rec["taskir_text"])
            except Exception as e:
                rec["parse_error"] = f"{type(e).__name__}: {e}"[:300]
                for arm, _lvl in ARMS:
                    rec["arms"][arm] = {"valid": False, "corrections": [],
                                        "rejections": [{"reason": "parse"}],
                                        "module_text": None, "opseq": None}
            else:
                rep = validate(mod)
                rec["validator_errors"] = [str(e) for e in rep.errors][:10]
                for arm, lvl in ARMS:
                    if lvl == 0:
                        res = {"valid": rep.valid, "corrections": [],
                               "rejections": []}
                        out_mod = mod
                    else:
                        r = apply_constraints(mod, lvl)
                        res = {"valid": r["valid"],
                               "corrections": r["corrections"],
                               "rejections": r["rejections"]}
                        out_mod = r["module"]
                    from src.ir.taskir import to_text
                    res["module_text"] = to_text(out_mod) if out_mod else None
                    res["opseq"] = op_sequence(out_mod)
                    rec["arms"][arm] = res
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        print(f"  archived {len(data)} samples ({lat:.0f}s gen)")

    f.close()
    print(f"saved {out_path}")
    print("ARCHIVE-DONE")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seeds", type=str, default="s42,s43,s44")
    args = ap.parse_args()
    main(args.n, tuple(args.seeds.split(",")))
