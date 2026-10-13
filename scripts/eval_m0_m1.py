"""Phase 8B-1B Task 8: paired M0/M1 evaluation (frozen evaluator).

Same tasks, same CapabilityIR. M0 -> TaskIR directly; M1 -> skeleton
JSON -> strict deterministic lowering -> TaskIR. Reports the eight
frozen metrics with paired M1-M0 differences, bootstrap CIs, and
fixed/broken transitions, grouped by split protocol (unseen-family /
same-family), depth (1 / 2-3 / 4-6 / 7+) and mutation presence.
"""
import json, os, sys, time, re, gc, argparse, random, math

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.ir.parser import parse_text
from src.validator.validator import validate
from src.compiler.plan_skeleton import skeleton_from_dict, validate_skeleton, hard_errors
from src.compiler.skeleton_lowering import lower_skeleton
from src.compiler.skeleton_lowering import lower_skeleton as _low
from scripts.train_m0_m1 import SYS_M0, SYS_M1, caps_block

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}


def metrics(taskir_text, sk_text_or_none, rec):
    """Eight frozen metrics vs the record's target_skeleton reference."""
    ref = rec["target_skeleton"]
    ref_nodes = ref["nodes"]
    ref_skills = [n["skill"] for n in ref_nodes]
    ref_tools = sorted({n["capability_id"] for n in ref_nodes})
    out = {"valid": False, "parse": False, "plan_f1": 0.0,
           "tool_f1": 0.0, "arg_acc": None, "dag_acc": 0.0,
           "effect_ok": None, "exact_joint": False, "n_pred": 0}
    if sk_text_or_none is not None:
        # skeleton channel: DAG accuracy measured on the skeleton itself
        try:
            pred_sk = skeleton_from_dict(json.loads(
                re.search(r"\{.*\}", sk_text_or_none, re.S).group(0)))
        except Exception:
            return out
        ref_edges = {(i, d) for i, n in enumerate(ref_nodes)
                     for d in n["depends_on"]}
        id_pos = {n["id"]: i for i, n in enumerate(ref_nodes)}
        pred_edges = set()
        for n in pred_sk.nodes:
            for d in n.depends_on:
                if n.id in id_pos and d in id_pos:
                    pred_edges.add((id_pos[n.id], id_pos[d]))
        if ref_edges or pred_edges:
            inter = len(ref_edges & pred_edges)
            p = inter / len(pred_edges) if pred_edges else 0.0
            r = inter / len(ref_edges) if ref_edges else 0.0
            out["dag_acc"] = round(2 * p * r / (p + r) if (p + r) else 0.0, 4)
        out["n_pred"] = len(pred_sk.nodes)
    if not taskir_text:
        return out
    try:
        mod = parse_text(taskir_text)
        out["parse"] = True
        out["valid"] = validate(mod).valid
    except Exception:
        return out
    pred_seq = [n.op for n in mod.program.nodes if n.op not in POLICY_OPS]

    def _f1(a, b):
        ca, cb = Counter(a), Counter(b)
        inter = sum((ca & cb).values())
        if not inter:
            return 0.0
        p, r = inter / max(1, sum(ca.values())), inter / max(1, sum(cb.values()))
        return round(2 * p * r / (p + r), 4)

    # ordered plan F1 (LCS)
    def lcs(a, b):
        if not a or not b:
            return 0
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b, 1):
                cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
            prev = cur
        return prev[-1]
    L = lcs(pred_seq, ref_skills)
    p = L / len(pred_seq) if pred_seq else 0.0
    r = L / len(ref_skills) if ref_skills else 0.0
    out["plan_f1"] = round(2 * p * r / (p + r) if (p + r) else 0.0, 4)
    if sk_text_or_none is None:
        out["n_pred"] = len(pred_seq)

    # tool identity
    bound = set()
    for n in mod.program.nodes:
        if n.op == "EXEC_ACTION" and n.params.get("action"):
            bound.add(str(n.params["action"]).lower())
        else:
            t = (n.hints or {}).get("tool")
            if t:
                bound.add(str(t).lower())
    out["tool_f1"] = _f1(sorted(bound), [t.lower() for t in ref_tools])

    # argument accuracy: ALL ref args of matched tools
    reg = rec["capability_registry"]
    tot = ok = 0
    tool_params = {}
    for n in mod.program.nodes:
        key = str(n.params.get("action") or (n.hints or {}).get("tool") or "").lower()
        if key:
            tool_params.setdefault(key, {}).update(
                {k: v for k, v in n.params.items() if k != "action"})
    for rn in ref_nodes:
        key = rn["capability_id"].lower()
        meta = reg.get(rn["capability_id"], {})
        if key not in bound:
            # unmatched tool: its required args count as missing
            if meta.get("schema") == "props":
                tot += len(meta.get("required") or [])
            continue
        for k, v in rn["arguments"].items():
            tot += 1
            pv = tool_params.get(key, {}).get(k)
            if pv is not None and str(pv).strip().lower() == str(v).strip().lower():
                ok += 1
    out["arg_acc"] = round(ok / tot, 4) if tot else None

    # effect proxy on corpus: mutating-class tools are those whose ref
    # skill is EXEC_ACTION; correct if identity-bound
    mut_refs = [n for n in ref_nodes if n["skill"] == "EXEC_ACTION"]
    if mut_refs:
        out["effect_ok"] = round(
            sum(1 for n in mut_refs
                if n["capability_id"].lower() in bound) / len(mut_refs), 4)
    else:
        out["effect_ok"] = 1.0 if out["valid"] else 0.0

    # exact joint: valid + ordered exact + tools exact + all args correct
    exact_seq = (pred_seq == ref_skills)
    exact_tools = (bound == {t.lower() for t in ref_tools})
    out["exact_joint"] = bool(out["valid"] and exact_seq and exact_tools
                              and (out["arg_acc"] in (None, 1.0)))
    return out


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
                             temperature=None,
                             pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text, round(time.time() - t0, 2), \
        int(out.shape[1] - ids["input_ids"].shape[1])


def main(splits=("dev_sft", "test_sft", "test_samefamily"), limit=0):
    rows_by_split = {k: [json.loads(l) for l in
                         open(f"data/semantic_skeleton_v3/{k}.jsonl",
                              encoding="utf-8") if l.strip()]
                     for k in splits}
    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    results = {}
    details = []
    for arm, adapter, sys_p in (
            ("M0", "runs/phase8b/m0_s42/final", SYS_M0),
            ("M1", "runs/phase8b/m1_s42/final", SYS_M1)):
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        per = []
        for sp in splits:
            rows = rows_by_split[sp]
            if limit:
                rows = rows[:limit]
            for rec in rows:
                user = (f"Task:\n{rec['instruction']}\n\n"
                        f"Available capabilities:\n{caps_block(rec)}")
                text, lat, ntok = gen(model, tok, sys_p, user)
                taskir = text
                sk_text = None
                low_ms = 0.0
                if arm == "M1":
                    sk_text = text
                    t0 = time.time()
                    try:
                        sk = skeleton_from_dict(json.loads(
                            re.search(r"\{.*\}", text, re.S).group(0)))
                        reg = {t: ({"params": [{"name": x, "optional": False}
                                               for x in m["required"]]}
                               if m.get("schema") == "props"
                               else {"params": None})
                               for t, m in rec["capability_registry"].items()}
                        errs = hard_errors(validate_skeleton(sk, None, reg))
                        low = lower_skeleton(sk, None, reg, strict=True) \
                            if not errs else {"taskir": None}
                        taskir = low.get("taskir") or ""
                    except Exception:
                        taskir = ""
                    low_ms = round((time.time() - t0) * 1000, 1)
                m = metrics(taskir, sk_text, rec)
                m.update({"arm": arm, "split": sp, "task_id": rec["task_id"],
                          "latency_s": lat, "gen_tokens": ntok,
                          "lowering_ms": low_ms,
                          "depth": ("1" if len(rec["target_skeleton"]["nodes"]) == 1
                                    else "2-3" if len(rec["target_skeleton"]["nodes"]) <= 3
                                    else "4-6" if len(rec["target_skeleton"]["nodes"]) <= 6
                                    else "7+"),
                          "has_mutation": any(n["skill"] == "EXEC_ACTION"
                                              for n in rec["target_skeleton"]["nodes"])})
                per.append(m)
                details.append(m)
        results[arm] = per
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{arm}: {len(per)} records")

    # aggregate + pairing
    def agg(rows):
        n = len(rows)
        def m(k):
            v = [x[k] for x in rows if x.get(k) is not None]
            return round(sum(v) / len(v), 4) if v else None
        return {"n": n,
                "valid%": round(100 * sum(x["valid"] for x in rows) / n, 1),
                "parse%": round(100 * sum(x["parse"] for x in rows) / n, 1),
                "plan_f1": m("plan_f1"), "tool_f1": m("tool_f1"),
                "arg_acc": m("arg_acc"), "dag_acc": m("dag_acc"),
                "effect": m("effect_ok"),
                "exact_joint%": round(100 * sum(x["exact_joint"] for x in rows) / n, 1),
                "tokens_mean": round(sum(x["gen_tokens"] for x in rows) / n, 1),
                "latency_p50": sorted(x["latency_s"] for x in rows)[n // 2],
                "lowering_ms_mean": m("lowering_ms")}

    summary = {}
    groups = {"ALL": lambda x: True,
              "depth:1": lambda x: x["depth"] == "1",
              "depth:2-3": lambda x: x["depth"] == "2-3",
              "depth:4-6+": lambda x: x["depth"] in ("4-6", "7+"),
              "mutation": lambda x: x["has_mutation"],
              "readonly": lambda x: not x["has_mutation"]}
    for sp in splits:
        for gname, gf in groups.items():
            for arm in ("M0", "M1"):
                rows = [x for x in results[arm] if x["split"] == sp and gf(x)]
                if rows:
                    summary[f"{sp}|{gname}|{arm}"] = agg(rows)
        # paired diffs on exact_joint + plan_f1
        m0 = {x["task_id"]: x for x in results["M0"] if x["split"] == sp}
        m1 = {x["task_id"]: x for x in results["M1"] if x["split"] == sp}
        for key in ("exact_joint",):
            diffs = []
            fixed = broken = 0
            for tid in m0:
                if tid not in m1:
                    continue
                a, b = m1[tid][key], m0[tid][key]
                diffs.append(1 if a else 0)
                if a and not b:
                    fixed += 1
                elif b and not a:
                    broken += 1
            rng = random.Random(13)
            boots = []
            for _ in range(10000):
                s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
                boots.append(sum(s) / len(s) if s else 0)
            boots.sort()
            base_rate = sum(diffs) / len(diffs) if diffs else 0
            m0_rate = sum(1 for t in m0 if m0[t][key]) / max(1, len(m0))
            summary[f"{sp}|paired|{key}"] = {
                "n": len(diffs), "M0%": round(100 * m0_rate, 1),
                "M1%": round(100 * base_rate, 1),
                "diff_pp": round(100 * (base_rate - m0_rate), 1),
                "ci95": [round(100 * boots[int(0.025 * 10000)], 1),
                         round(100 * boots[int(0.975 * 10000)], 1)],
                "fixed": fixed, "broken": broken}

    os.makedirs("results/phase8b", exist_ok=True)
    with open("results/phase8b/m0_m1_comparison.json", "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2,
                  ensure_ascii=False)
    for sp in splits:
        for g in ("ALL", "depth:2-3", "depth:4-6+", "mutation"):
            k0 = f"{sp}|{g}|M0"
            k1 = f"{sp}|{g}|M1"
            if k0 in summary and k1 in summary:
                s0, s1 = summary[k0], summary[k1]
                print(f"{sp}|{g}: valid {s0['valid%']}->{s1['valid%']} "
                      f"planF1 {s0['plan_f1']}->{s1['plan_f1']} "
                      f"tool {s0['tool_f1']}->{s1['tool_f1']} "
                      f"arg {s0['arg_acc']}->{s1['arg_acc']} "
                      f"dag {s0['dag_acc']}->{s1['dag_acc']} "
                      f"joint {s0['exact_joint%']}->{s1['exact_joint%']}")
        if f"{sp}|paired|exact_joint" in summary:
            print(f"  paired: {summary[f'{sp}|paired|exact_joint']}")
    print("EVAL-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev_sft,test_sft,test_samefamily")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(tuple(a.splits.split(",")), a.limit)
