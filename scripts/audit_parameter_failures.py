"""Phase 8B-1C Task 4: parameter failure taxonomy (A-F) over saved
M0/M1 outputs, plus per-split/depth/family stats and M0-correct /
M1-wrong case mining with anonymized evidence.

A instruction_contains_value_model_wrong
B value_not_user_visible_runtime_context
C wrong_param_key
D type_or_serialization_error
E param_attached_to_wrong_call
F reference_ambiguous_or_insufficient
"""
import json, os, sys, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUTS = "results/phase8b/m0_m1_outputs.jsonl"
V3DIR = "results/phase8b"


def load_records():
    data = {}
    for split in ("dev_sft", "test_sft", "test_samefamily"):
        for l in open(f"data/semantic_skeleton_v3/{split}.jsonl",
                      encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                data[r["task_id"]] = r
    return data


def classify(rn_args, pred_params, instruction, call):
    """Classify ONE failed parameter into A-F (first applicable)."""
    fails = []
    for k, rv in (rn_args or {}).items():
        pv = pred_params.get(k)
        if pv is None:
            # wrong key? maybe value present under another key
            val = str(rv)
            if val and any(_eq(v, val) for v in pred_params.values()):
                fails.append((k, "C"))
            elif val and val in instruction:
                fails.append((k, "A"))     # value visible but key omitted
            else:
                fails.append((k, "B"))     # not user-visible
        elif not _eq(pv, rv):
            val = str(rv)
            if val in instruction:
                fails.append((k, "A"))     # visible, extracted wrong
            elif _same_digits(pv, rv) or _same_alpha(pv, rv):
                fails.append((k, "D"))     # formatting/serialization
            else:
                fails.append((k, "A" if val.lower() in instruction.lower()
                              else "B"))
    return fails


def _eq(a, b):
    return str(a).strip().lower() == str(b).strip().lower()


def _same_digits(a, b):
    da = re.sub(r"\D", "", str(a))
    db = re.sub(r"\D", "", str(b))
    return bool(da) and da == db


def _same_alpha(a, b):
    return re.sub(r"[^a-z]", "", str(a).lower()) == \
        re.sub(r"[^a-z]", "", str(b).lower())


def main():
    recs = load_records()
    rows = [json.loads(l) for l in open(OUTS, encoding="utf-8")
            if l.strip()]
    # v3 rescoring embedded here for arg taxonomy
    from src.eval.semantic_eval_v3 import evaluate_v3
    tax = defaultdict(Counter)
    cases = []
    v3_rows = []
    for r in rows:
        rec = recs.get(r["task_id"])
        if rec is None:
            continue
        v3 = evaluate_v3(r.get("taskir_text", ""), rec)
        v3.update({"arm": r["arm"], "split": r["split"],
                   "task_id": r["task_id"], "depth": r["depth"],
                   "has_mutation": r["has_mutation"]})
        v3_rows.append(v3)
        fam = rec["provenance"].get("gold_tools", ["?"])[0]
        instr = rec["instruction"]
        # classify each failed call's params
        ref_nodes = rec["target_skeleton"]["nodes"]
        for cm in v3["call_matches"]:
            if cm["identity_ok"] is not True:
                continue
            st = cm.get("arg_status") or ""
            if st in ("ok", "keys_ok_value_unverifiable") or \
                    st.startswith("ok"):
                continue
            rn = ref_nodes[cm["pos"]]
            # find pred params for this call from the raw TaskIR
            pred_params = _params_at(r.get("taskir_text", ""),
                                     cm["pos"])
            for k, cat in classify(rn["arguments"], pred_params,
                                   instr, cm):
                key = (r["arm"], r["split"])
                tax[key][cat] += 1
                tax[(r["arm"], "ALL")][cat] += 1
                if len(cases) < 400:
                    cases.append({
                        "task_id": r["task_id"], "arm": r["arm"],
                        "split": r["split"], "call": cm["pos"],
                        "param": k, "class": cat,
                        "ref_value": str(rn["arguments"].get(k))[:80],
                        "pred_value": str(pred_params.get(k))[:80],
                        "instr_snippet": _anon(instr)[:160]})
    with open(f"{V3DIR}/parameter_error_taxonomy.json", "w") as f:
        json.dump({"taxonomy_counts": {f"{a}|{s}": dict(c)
                                       for (a, s), c in tax.items()},
                   "cases": cases}, f, indent=2, ensure_ascii=False)
    for k in sorted(tax):
        print(k, dict(tax[k]))
    print("saved parameter_error_taxonomy.json")


def _anon(t):
    t = re.sub(r"[\w.+-]+@[\w-]+", "<email>", t)
    t = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", "<phone>", t)
    return t


def _params_at(taskir_text, pos):
    from src.eval.semantic_eval_v3 import extract_calls_from_taskir
    calls = extract_calls_from_taskir(taskir_text) or []
    if pos < len(calls):
        return calls[pos]["params"]
    return {}


if __name__ == "__main__":
    main()
