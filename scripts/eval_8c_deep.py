"""Phase 8C: paired M0/M1 evaluation on deep tasks (v3 evaluator)."""
import json, os, sys
ROOT = "/ccfa2026/compile-anything"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import time, re, gc, random
import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.compiler.plan_skeleton import skeleton_from_dict, validate_skeleton, hard_errors
from src.compiler.skeleton_lowering import lower_skeleton
from src.eval.semantic_eval_v3 import evaluate_v3
from scripts.train_m0_m1 import SYS_M0, SYS_M1, caps_block

SPLITS = {
    "test_sft": "data/semantic_skeleton_v4/test_sft.jsonl",
    "test_samefamily": "data/semantic_skeleton_v4/test_samefamily.jsonl",
    "test_deep": "data/semantic_skeleton_v4/test_deep.jsonl",
}
SEED_ADP = "43"
OUTPUT = "results/phase8c/m0_m1_deep_comparison.json"


def gen(model, tok, sys_p, user):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": sys_p},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    t0 = time.time()
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=700, do_sample=False,
                             temperature=None, pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text, round(time.time() - t0, 2), int(out.shape[1] - ids["input_ids"].shape[1])


def main():
    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    all_recs = {}
    for k, path in SPLITS.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if k == "test_samefamily" and len(rows) > 200:
            random.Random(7).shuffle(rows)
            rows = rows[:200]
        all_recs[k] = rows

    results = {"details": []}
    for arm, adapter, sys_p in (
            ("M0", f"runs/phase8b/m0_s{SEED_ADP}/final", SYS_M0),
            ("M1", f"runs/phase8b/m1_s{SEED_ADP}/final", SYS_M1)):
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        for sp, recs in all_recs.items():
            print(f"  {arm}/{sp}: {len(recs)} tasks", flush=True)
            for rec in recs:
                user = (f"Task:\n{rec['instruction']}\n\n"
                        f"Available capabilities:\n{caps_block(rec)}")
                text, lat, ntok = gen(model, tok, sys_p, user)
                taskir = text
                if arm == "M1":
                    try:
                        sk = skeleton_from_dict(json.loads(
                            re.search(r"\{.*\}", text, re.S).group(0)))
                        reg = {t: ({"params": [{"name": x, "optional": False}
                                               for x in m.get("required", [])]}
                               if m.get("properties") else {"params": None})
                               for t, m in rec["capability_registry"].items()}
                        errs = hard_errors(validate_skeleton(sk, None, reg))
                        low = lower_skeleton(sk, None, reg, strict=True) if not errs else {"taskir": None}
                        taskir = low.get("taskir") or ""
                    except Exception:
                        taskir = ""
                v3 = evaluate_v3(taskir, rec)
                v3.update({"arm": arm, "split": sp, "task_id": rec["task_id"],
                           "depth": ("1" if len(rec["target_skeleton"]["nodes"]) == 1
                                     else "2-3" if len(rec["target_skeleton"]["nodes"]) <= 3
                                     else "4-6" if len(rec["target_skeleton"]["nodes"]) <= 6
                                     else "7+"),
                           "latency_s": lat, "gen_tokens": ntok})
                v3["raw_output"] = text[:4000]
                v3["taskir_text"] = taskir[:4000]
                results["details"].append(v3)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{arm} done")

    # summary + paired stats
    def agg(rows):
        n = len(rows)
        if not n:
            return None
        def m(k):
            v = [x.get(k) for x in rows if x.get(k) is not None]
            return round(sum(v) / len(v), 4) if v else None
        return {"n": n,
                "valid%": round(100 * sum(x["parse"] for x in rows) / n, 1),
                "exact_joint_v3%": round(100 * sum(x["exact_joint_v3"] for x in rows) / n, 1),
                "identity_ok%": round(100 * sum(x["identity_ok"] for x in rows) / n, 1),
                "order_ok%": round(100 * sum(x["order_ok"] for x in rows) / n, 1),
                "args_ok%": round(100 * sum(x["args_ok"] for x in rows) / n, 1),
                "dag_data_f1": m("dag_data_f1"),
                "dag_order_f1": m("dag_order_f1"),
                "missing_params": m("missing_params_total"),
                "wrong_value": m("wrong_value_total"),
                "extra_calls": m("extra_calls")}

    summary = {}
    groups = {"ALL": lambda x: True,
              "depth:4-6": lambda x: x["depth"] == "4-6",
              "depth:7+": lambda x: x["depth"] == "7+",
              "depth:2-3": lambda x: x["depth"] == "2-3",
              "depth:1": lambda x: x["depth"] == "1"}
    for sp in SPLITS:
        for gname, gf in groups.items():
            for arm in ("M0", "M1"):
                rows = [x for x in results["details"]
                        if x["arm"] == arm and x["split"] == sp and gf(x)]
                if rows:
                    summary[f"{sp}|{gname}|{arm}"] = agg(rows)
        # paired
        m0 = {x["task_id"]: x for x in results["details"]
              if x["arm"] == "M0" and x["split"] == sp}
        m1 = {x["task_id"]: x for x in results["details"]
              if x["arm"] == "M1" and x["split"] == sp}
        for key in ("exact_joint_v3", "args_ok", "identity_ok"):
            deltas = []
            fixed = broken = 0
            for tid in m0:
                if tid not in m1:
                    continue
                a = 1 if m1[tid][key] else 0
                b = 1 if m0[tid][key] else 0
                deltas.append(a - b)
                if a and not b:
                    fixed += 1
                elif b and not a:
                    broken += 1
            if deltas:
                rng = random.Random(13)
                boots = []
                for _ in range(10000):
                    s = [deltas[rng.randrange(len(deltas))] for _ in deltas]
                    boots.append(sum(s) / len(s))
                boots.sort()
                summary[f"{sp}|paired|{key}"] = {
                    "n": len(deltas),
                    "M0%": round(100 * sum(1 for t in m0 if m0[t][key]) / len(m0), 1),
                    "M1%": round(100 * sum(1 for t in m1 if t in m1 and m1[t][key]) / len(m1), 1),
                    "diff_pp": round(100 * sum(deltas) / len(deltas), 1),
                    "ci95": [round(100 * boots[250], 1), round(100 * boots[9750], 1)],
                    "fixed": fixed, "broken": broken}

    results["summary"] = summary
    os.makedirs("results/phase8c", exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for sp in SPLITS:
        for g in ("ALL", "depth:4-6", "depth:7+"):
            k0, k1 = f"{sp}|{g}|M0", f"{sp}|{g}|M1"
            if k0 in summary and k1 in summary:
                s0, s1 = summary[k0], summary[k1]
                print(f"{sp}|{g}: joint {s0['exact_joint_v3%']}->{s1['exact_joint_v3%']} "
                      f"args {s0['args_ok%']}->{s1['args_ok%']} "
                      f"dag_data {s0['dag_data_f1']}->{s1['dag_data_f1']}")
        if f"{sp}|paired|exact_joint_v3" in summary:
            print(f"  paired: {summary[f'{sp}|paired|exact_joint_v3']}")
    print("8C-EVAL-DONE")


if __name__ == "__main__":
    main()
