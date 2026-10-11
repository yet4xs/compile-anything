"""Phase 7C-0.1: Parser-faithful SSA constraint evaluation (v2).

Fixes over v1 (per advisor review):
1. src/ir/parser.py::parse_text() is the SOLE parsing entry — no custom regex.
2. Constraints operate on parsed Node/Program objects (AST level), not text.
3. Type inference calls spec.output_type(input_types); checking via types.is_compatible.
4. Each seed loads a FRESH Qwen base (no PEFT adapter sharing / contamination).
5. Per-sample statistics recorded independently (no cumulative counters).
6. D1 fidelity test: on programs that are already D0-valid, D1 must reproduce
   the identical AST (lossless assembly).

Framing (honest): this is POST-HOC constrained assembly — the model generates a
free-form completion once; constraints then parse + repair + re-validate it.
It is NOT token-level constrained decoding. D0-D3 share the SAME raw model
outputs per seed; only the post-processing pipeline differs.

Arms:
  D0: parse_text() -> validate()                       (baseline)
  D1: D0 + grammar/structure checks                    (lossless on valid)
  D2: D1 + typed SSA auto-fix (undef ref / arity / output retarget)
  D3: D2 + control-flow preservation (after/guard/retry)
"""
import json, os, sys, copy, time, gc
import argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.ir.parser import parse_text, TaskIRSyntaxError
from src.ir.taskir import Module, Node, ID_RE
from src.ir.types import is_compatible, normalize
from src.validator.validator import validate
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.constrained_assembly import (
    apply_constraints, module_ast_equal, node_key)
from src.isa.registry import get as get_skill

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}


# ---------------------------------------------------------------- data

def load_test_data(n=30):
    """Common eval tasks + gold capability blocks + reference TaskIR."""
    refs = {}
    for line in open("data/compiler_corpus_v4/test.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        refs[r["id"]] = r.get("plan_text", "") or r.get("plan_target", "")

    data = []
    for line in open("data/phase6b_resolver/common_eval_tasks.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        gold = [c for c in t.get("candidates", []) if c.get("is_gold")]
        if not gold:
            continue
        data.append({
            "id": t["id"],
            "task": t["task"],
            "cap_blocks": "\n\n".join(c.get("block", "") for c in gold[:5]),
            "ref_text": refs.get(t["id"], ""),
        })
        if len(data) >= n:
            break
    return data


def op_sequence(mod: Module):
    """Policy-op-excluded op sequence for opseq accuracy."""
    return [n.op for n in mod.program.nodes if n.op not in POLICY_OPS]


# ---------------------------------------------------------------- generation

def generate_for_seed(seed, data, tok, base_path="weights/Qwen2.5-3B-Instruct"):
    """Fresh base + adapter per seed; unload fully before returning."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(
        base, f"runs/phase6b/composer_C2_{seed}/final")
    model.eval()

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": f"{s['task']}\n\nSelected capabilities:\n{s['cap_blocks']}"}],
        tokenize=False, add_generation_prompt=True) for s in data]

    outputs = []
    t0 = time.time()
    for off in range(0, len(prompts), 8):
        if off and off % 24 == 0:
            print(f"    {off}/{len(prompts)}", flush=True)
        batch = tok(prompts[off:off + 8], return_tensors="pt", padding=True,
                    truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outputs.extend(tok.decode(g[batch["input_ids"].shape[1]:],
                                  skip_special_tokens=True) for g in gen)
    latency = time.time() - t0

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, latency


# ---------------------------------------------------------------- main

ARMS = [("D0_normal", 0), ("D1_grammar", 1),
        ("D2_typed_ssa", 2), ("D3_control", 3)]


def main(n=30, seeds=("s42", "s43", "s44")):
    print("=" * 64)
    print(f"Phase 7C-0.1  Parser-faithful constraint eval  n={n}")
    print("post-hoc constrained assembly (NOT token-level constrained decoding)")
    print("=" * 64)

    data = load_test_data(n)
    print(f"test samples: {len(data)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    summary, details = {}, []

    for seed in seeds:
        if not os.path.exists(f"runs/phase6b/composer_C2_{seed}/final"):
            print(f"\n[skip {seed}: adapter not found]")
            continue
        print(f"\n=== seed {seed} (fresh base load) ===")
        outputs, lat = generate_for_seed(seed, data, tok)
        print(f"  generated {len(outputs)} completions in {lat:.0f}s")

        seed_stats = {}
        # parse each completion ONCE with the official parser
        parsed = []
        for s, out in zip(data, outputs):
            try:
                parsed.append(parse_text(extract_taskir_text(out)))
            except Exception:
                parsed.append(None)

        n_parse_ok = sum(p is not None for p in parsed)
        print(f"  parse_text() success: {n_parse_ok}/{len(data)}")

        for arm, level in ARMS:
            n_valid = n_opseq = n_ast_keep = 0
            n_d0_valid = 0
            corr_kinds = {}

            for s, mod in zip(data, parsed):
                row = {"id": s["id"], "seed": seed, "arm": arm,
                       "parse_ok": mod is not None}
                if mod is None:
                    row.update(valid=False, opseq=False, corrections=0,
                               rejections=1, ast_preserved=None)
                    details.append(row)
                    continue

                if level == 0:
                    try:
                        v = validate(mod).valid
                    except Exception:
                        v = False
                    res = {"valid": v, "corrections": [], "rejections": [],
                           "module": mod}
                else:
                    res = apply_constraints(mod, level)

                n_valid += res["valid"]

                # D1 fidelity: valid-in-D0 must stay AST-identical under D1
                if level == 1:
                    try:
                        d0v = validate(mod).valid
                    except Exception:
                        d0v = False
                    if d0v:
                        n_d0_valid += 1
                        keep = (res["valid"] and
                                module_ast_equal(mod, res["module"]))
                        n_ast_keep += keep
                        row["d1_lossless"] = keep
                    else:
                        row["d1_lossless"] = None

                # opseq vs reference
                row_opseq = False
                if res["valid"] and s["ref_text"]:
                    try:
                        ref = parse_text(s["ref_text"])
                        row_opseq = (op_sequence(res["module"]) ==
                                     op_sequence(ref))
                    except Exception:
                        pass
                n_opseq += row_opseq
                for c in res["corrections"]:
                    corr_kinds[c["kind"]] = corr_kinds.get(c["kind"], 0) + 1
                row.update(valid=res["valid"], opseq=row_opseq,
                           corrections=len(res["corrections"]),
                           rejections=len(res["rejections"]))
                details.append(row)

            denom = max(1, len(data))
            seed_stats[arm] = {
                "n": len(data),
                "parse_ok": n_parse_ok,
                "valid": n_valid, "valid_pct": round(100 * n_valid / denom, 1),
                "opseq": n_opseq, "opseq_pct": round(100 * n_opseq / denom, 1),
                "correction_kinds": corr_kinds,
            }
            if level == 1:
                seed_stats[arm]["d1_lossless"] = n_ast_keep
                seed_stats[arm]["d1_lossless_of"] = n_d0_valid
            print(f"  {arm:<14s} valid={seed_stats[arm]['valid_pct']:>5.1f}% "
                  f"opseq={seed_stats[arm]['opseq_pct']:>5.1f}% "
                  f"corr={corr_kinds}"
                  + (f" d1_lossless={n_ast_keep}/{n_d0_valid}"
                     if level == 1 else ""))

        summary[seed] = seed_stats

    print("\n" + "=" * 64)
    print("SUMMARY  (valid% / opseq%)")
    print("=" * 64)
    for seed, st in summary.items():
        cells = "  ".join(
            f"{a}={st[a]['valid_pct']:.1f}/{st[a]['opseq_pct']:.1f}"
            for a, _ in ARMS)
        print(f"{seed:<6s} {cells}")

    os.makedirs("results/phase7", exist_ok=True)
    out = {"design": "post_hoc_constrained_assembly_v2",
           "parser": "src/ir/parser.py::parse_text (sole entry)",
           "n": len(data), "summary": summary, "details": details}
    with open("results/phase7/constrained_generation_v2.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved results/phase7/constrained_generation_v2.json")
    print("CONSTRAINED-V2-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seeds", type=str, default="s42,s43,s44")
    args = ap.parse_args()
    main(args.n, tuple(args.seeds.split(",")))
