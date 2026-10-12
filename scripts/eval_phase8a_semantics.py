"""Phase 8A Tasks 4-5: semantic correctness across attribution conditions.

Conditions (all oracle conditions explicitly labeled; only A is an
end-to-end learned result):
  A Fully Learned      : learned canonicalizer -> learned resolver -> composer
  B Canonicalizer Oracle: oracle tool labels -> learned resolver -> composer
  C Selected Tool Oracle: gold reference tool set + oracle labels -> composer
  D Ref-plan Upper Bound: oracle reference plan bound to TaskIR directly
                          (binding/expressiveness diagnosis, no model)

Task 5: condition A additionally runs G0/G1/G2 to measure whether online
constraints improve SEMANTIC layers or only legalize wrong programs.

Task source: tau3 tasks.json (full reference actions with arguments);
tasks overlapping the 7C-2.1 sample are excluded so the confirmation set
did not participate in this round's scorer development.
"""
import json, os, sys, time, gc, random, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.skill_grounder import build_probe_prompt, parse_label
from src.compiler.online_constrained_decoder import generate_constrained
from src.eval.taskir_semantic_eval import evaluate_sample
from src.eval.tau3_skill_oracle import get_tool_oracle
from src.eval.tau3_tool_registry import build_registry, runtime_pool

SEED = 42
TAU3_TASKS = ("third_party/tau3-bench/data/tau2/domains"
              "/{d}/tasks.json")
DOMAINS = ["airline", "banking_knowledge", "retail", "telecom"]
SAMPLING_SEED = 20261013


def load_tasks(n_per_domain, exclude_case_ids):
    tasks = []
    for d in DOMAINS:
        path = TAU3_TASKS.format(d=d)
        data = json.load(open(path, encoding="utf-8"))
        usable = []
        for t in data:
            acts = (t.get("evaluation_criteria") or {}).get("actions") or []
            if not acts:
                continue                    # refusal/communicate-only tasks
            case_id = f"{d}-{t['id']}"
            if case_id in exclude_case_ids:
                continue
            usable.append({"case_id": case_id, "domain": d,
                           "instruction": t["user_scenario"]["instructions"],
                           "ref_actions": [
                               {"name": a["name"],
                                "arguments": a.get("arguments") or {}}
                               for a in acts if a.get("name")]})
        rng = random.Random(SAMPLING_SEED)
        rng.shuffle(usable)
        tasks.extend(usable[:n_per_domain])
    return tasks


def resolve_selection(resolver, tok, instruction, tools, labels):
    """Learned resolver over (tool,label) blocks; returns selected list.
    NONE decision is preserved (empty list) — no forced top-1 fallback."""
    SYSTEM_R = ("You are a capability resolver for a task compiler. Given a "
                "task and capability descriptions, decide which capabilities "
                "the task requires. Answer with the label ONLY.")
    r_ids = tok.encode("relevant", add_special_tokens=False)
    i_ids = tok.encode("irrelevant", add_special_tokens=False)

    def score_pair(tool_name, skill):
        block = f"[c0] name: {tool_name}\ncanonical_skill: {skill}"
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_R},
             {"role": "user", "content":
              f"Task:\n{instruction}\n\nCapability:\n{block}\n\n"
              "Is this capability required by the task? "
              "Answer 'relevant' or 'irrelevant'."}],
            tokenize=False, add_generation_prompt=True)
        seqs = [tok(prompt + "relevant", return_tensors="pt"),
                tok(prompt + "irrelevant", return_tensors="pt")]
        maxlen = max(s["input_ids"].shape[1] for s in seqs)
        ii, am = [], []
        for s in seqs:
            pad = maxlen - s["input_ids"].shape[1]
            ii.append([tok.pad_token_id] * pad + s["input_ids"][0].tolist())
            am.append([0] * pad + s["attention_mask"][0].tolist())
        ii = torch.tensor(ii).to(resolver.device)
        am = torch.tensor(am).to(resolver.device)
        with torch.no_grad():
            logits = resolver(input_ids=ii, attention_mask=am,
                              num_logits_to_keep=4).logits
        K = 4

        def seq_lp(j, lab_ids):
            total = 0.0
            for tp in range(maxlen - len(lab_ids), maxlen):
                idx = tp - 1 - (maxlen - K)
                row = torch.log_softmax(logits[j, idx].float(), -1)
                total += float(row[ii[j, tp]])
            return total

        return seq_lp(0, r_ids) - seq_lp(1, i_ids)

    sel = []
    for tool in tools:
        s = score_pair(tool["name"], labels.get(tool["name"], "UNKNOWN"))
        if s > 0:
            sel.append((tool["name"], labels.get(tool["name"], "UNKNOWN")))
    return sel                      # may be empty = NONE decision


def ideal_taskir(domain, ref_actions):
    """Condition D: bind the reference plan directly as TaskIR."""
    lines = ["; TaskIR v0.1  module=ref_plan", "inputs: @task: Str", ""]
    prev = "@task"
    for i, a in enumerate(ref_actions):
        o = get_tool_oracle(domain, a["name"])
        skill = o["semantic_skill"] or "EXEC_ACTION"
        args = a.get("arguments") or {}
        if skill == "EXEC_ACTION":
            parts = [f"action={a['name']}"] + [
                f"{k}={json.dumps(v)}" for k, v in args.items()]
        else:
            parts = [prev] + [f"{k}={json.dumps(v)}"
                              for k, v in args.items()]
        lines.append(f"%c{i} = {skill}({', '.join(parts)})")
        prev = f"%c{i}"
    lines += ["", f"return {prev}"]
    return "\n".join(lines) + "\n"


def agg(rows):
    n = max(1, len(rows))
    def m(key):
        vals = [_dig(r, key) for r in rows]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "n": len(rows),
        "l0_valid": round(100 * sum(r["l0"]["valid"] for r in rows) / n, 1),
        "l0_parse": round(100 * sum(r["l0"]["parse_ok"] for r in rows) / n, 1),
        "l1_f1": m(("l1", "f1")), "l1_exact": round(
            100 * sum(r["l1"]["exact_set"] for r in rows) / n, 1),
        "l2_exact_f1": m(("l2", "exact_f1")),
        "l2_seq_match": round(
            100 * sum(r["l2"]["exact_sequence_match"] for r in rows) / n, 1),
        "l2_coarse_f1": m(("l2", "coarse_f1")),
        "l3_ordered_recall": m(("l3", "ordered_recall")),
        "l3_arg_acc": m(("l3", "arg_value_acc")),
        "l3_req_cov": m(("l3", "required_param_coverage")),
        "l4_effect": m(("l4", "effect_consistency")),
        "l4_bindability": m(("l4", "bindability")),
        "l4_unsafe_retry": sum(len(r["l4"]["unsafe_retry"]) for r in rows),
        "pred_steps": m(("l3", "pred_steps")),
        "ref_steps": m(("l3", "ref_steps")),
    }


def _dig(r, path):
    a, b = path
    return r.get(a, {}).get(b) if isinstance(r.get(a), dict) else None


def main(n_per_domain=30):
    reg = build_registry()
    pool = runtime_pool(reg)
    schemas = {d: {t["name"]: t for t in ts} for d, ts in pool.items()}

    exclude = set()
    try:
        cs = json.load(open("results/phase7/online_cross_source.json"))
        for r in cs.get("details", []):
            exclude.add(r["case_id"])
    except FileNotFoundError:
        pass
    tasks = load_tasks(n_per_domain, exclude)
    print(f"tasks: {len(tasks)} "
          f"({dict(Counter(t['domain'] for t in tasks))}), "
          f"excluded 7C-2.1 overlap: {len(exclude)}")

    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"          # batched canonicalizer probes
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    # learned canonicalizer labels for pool tools
    canon = PeftModel.from_pretrained(
        base, f"runs/phase6/grounder_g1g2_s{SEED}/final")
    canon.eval()
    learned_labels = {}
    for d, tools in pool.items():
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": build_probe_prompt(
                {"instruction": "", "capability": t}, "A")["system"]},
             {"role": "user", "content": build_probe_prompt(
                 {"instruction": "", "capability": t}, "A")["user"]}],
            tokenize=False, add_generation_prompt=True) for t in tools]
        for bs in range(0, len(prompts), 16):
            inputs = tok(prompts[bs:bs + 16], return_tensors="pt",
                         padding=True, truncation=True,
                         max_length=1024).to(canon.device)
            with torch.no_grad():
                out = canon.generate(**inputs, max_new_tokens=8,
                                     do_sample=False, temperature=None,
                                     pad_token_id=tok.pad_token_id)
            for t, o in zip(tools[bs:bs + 16], out):
                learned_labels[(d, t["name"])] = parse_label(tok.decode(
                    o[inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True))
    del canon
    gc.collect()
    torch.cuda.empty_cache()
    print("canonicalizer labels done")

    oracle_labels = {(d, t["name"]): get_tool_oracle(d, t["name"])["semantic_skill"]
                     for d, ts in pool.items() for t in ts}
    none_decisions = Counter()

    resolver = PeftModel.from_pretrained(
        base, f"runs/phase6b/resolver_R2_s{SEED}/final")
    resolver.eval()
    selections = {"A": {}, "B": {}}
    for i, t in enumerate(tasks):
        if i % 10 == 0:
            print(f"  resolver {i}/{len(tasks)}", flush=True)
        d = t["domain"]
        tools = pool[d]
        selA = resolve_selection(resolver, tok, t["instruction"], tools,
                                 {x["name"]: learned_labels[(d, x["name"])]
                                  for x in tools})
        selB = resolve_selection(resolver, tok, t["instruction"], tools,
                                 {x["name"]: oracle_labels[(d, x["name"])]
                                  for x in tools})
        selections["A"][t["case_id"]] = selA
        selections["B"][t["case_id"]] = selB
        if not selA:
            none_decisions["A"] += 1
        if not selB:
            none_decisions["B"] += 1
    del resolver
    gc.collect()
    torch.cuda.empty_cache()
    print(f"resolver done (NONE decisions: {dict(none_decisions)})")

    composer = PeftModel.from_pretrained(
        base, f"runs/phase6b/composer_C2_s{SEED}/final")
    composer.eval()

    def compose(blocks, level):
        user = (f"{instruction_cache[case_id]}\n\nSelected capabilities:\n"
                + "\n\n".join(blocks))
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        text, _st, _p = generate_constrained(composer, tok, prompt, level,
                                             max_new_tokens=512)
        return text

    instruction_cache = {t["case_id"]: t["instruction"] for t in tasks}
    conditions = defaultdict(list)          # cond-arm -> rows
    details = []
    for ci, t in enumerate(tasks):
        d, case_id = t["domain"], t["case_id"]
        schemas_d = schemas[d]

        def run_cond(name, sel, level):
            blocks = [f"[c{i}] name: {n}\ncanonical_skill: {s}"
                      for i, (n, s) in enumerate(sel)]
            text = compose(blocks, level) if blocks else ""
            if not blocks:                  # NONE decision preserved
                text = ""
            r = evaluate_sample(d, extract_taskir_text(text) if text else "",
                                t["ref_actions"], schemas_d)
            r["condition"] = name
            r["case_id"] = case_id
            r["text"] = text
            details.append(r)
            conditions[f"{name}"].append(r)
            return r

        if ci % 10 == 0:
            print(f"  composer {ci}/{len(tasks)}", flush=True)
        # A: learned, three arms
        for arm, level in (("A_G0", 0), ("A_G1", 1), ("A_G2", 3)):
            run_cond(arm, selections["A"][case_id], level)
        # B: canonicalizer oracle
        run_cond("B_G0", selections["B"][case_id], 0)
        # C: selected tool oracle (gold tools + oracle labels)
        gold = [(a["name"], oracle_labels.get((d, a["name"]),
                                              get_tool_oracle(d, a["name"])["semantic_skill"] or "EXEC_ACTION"))
                for a in t["ref_actions"]]
        gold = [(n, s or "EXEC_ACTION") for n, s in gold]
        seen = set()
        gold_sel = [(n, s) for n, s in gold
                    if not (n in seen or seen.add(n))]
        run_cond("C_G0", gold_sel, 0)
        # D: reference plan bound directly
        d_text = ideal_taskir(d, t["ref_actions"])
        r = evaluate_sample(d, d_text, t["ref_actions"], schemas_d)
        r["condition"] = "D_refplan"
        r["case_id"] = case_id
        r["text"] = d_text
        details.append(r)
        conditions["D_refplan"].append(r)

    del composer
    gc.collect()
    torch.cuda.empty_cache()

    summary = {k: agg(v) for k, v in conditions.items()}

    # ---- Task 4 attribution ----
    by = {}
    for r in details:
        by.setdefault((r["condition"], r["case_id"]), r)
    def get(cond, cid):
        return by.get((cond, cid))
    attr = Counter()
    task_flags = defaultdict(list)
    for t in tasks:
        cid = t["case_id"]
        a = get("A_G0", cid)
        b = get("B_G0", cid)
        c = get("C_G0", cid)
        dd = get("D_refplan", cid)
        sem_ok = lambda r: (r and r["l0"]["valid"]
                            and r["l2"]["exact_f1"] >= 0.99)
        if a is not None:
            if a["l0"]["valid"] and not sem_ok(a):
                task_flags[cid].append("validator_only_error")
                attr["validator_only_error"] += 1
            if not sem_ok(a) and sem_ok(b):
                task_flags[cid].append("canonicalizer_error")
                attr["canonicalizer_error"] += 1
            elif not sem_ok(b) and sem_ok(c):
                task_flags[cid].append("resolver_error")
                attr["resolver_error"] += 1
            elif not sem_ok(c):
                if c and c["l2"]["exact_f1"] < 0.99:
                    task_flags[cid].append("composer_planning_error")
                    attr["composer_planning_error"] += 1
                if c and c["l1"]["f1"] < 0.99 and c["l2"]["exact_f1"] >= 0.99:
                    task_flags[cid].append("tool_binding_error")
                    attr["tool_binding_error"] += 1
            if c and sem_ok(c) and c["l1"]["f1"] < 0.99:
                task_flags[cid].append("tool_binding_error")
                attr["tool_binding_error"] += 1
        if dd is not None and not sem_ok(dd):
            task_flags[cid].append("binding_upper_bound_gap")
            attr["binding_upper_bound_gap"] += 1
    # overlaps preserved (a task can appear in several flags)
    n_multi = sum(1 for v in task_flags.values() if len(v) > 1)

    os.makedirs("results/phase8a", exist_ok=True)
    with open("results/phase8a/semantic_layers.json", "w") as f:
        json.dump({"summary": summary,
                   "none_decisions": dict(none_decisions),
                   "n_tasks": len(tasks)}, f, indent=2, ensure_ascii=False)
    with open("results/phase8a/error_attribution.json", "w") as f:
        json.dump({"counts": dict(attr),
                   "n_tasks_flagged": len(task_flags),
                   "n_multi_cause": n_multi,
                   "task_flags": {k: v for k, v in task_flags.items()}},
                  f, indent=2, ensure_ascii=False)
    with open("results/phase8a/semantic_details.jsonl", "w") as f:
        for r in details:
            f.write(json.dumps(r, ensure_ascii=False,
                               default=str) + "\n")

    print("\n" + "=" * 70)
    for k in ("A_G0", "A_G1", "A_G2", "B_G0", "C_G0", "D_refplan"):
        s = summary[k]
        print(f"{k:<10} valid={s['l0_valid']:>5}% L1f1={s['l1_f1']} "
              f"L2exact={s['l2_exact_f1']} seq={s['l2_seq_match']}% "
              f"ordRec={s['l3_ordered_recall']} argAcc={s['l3_arg_acc']} "
              f"eff={s['l4_effect']} bind={s['l4_bindability']} "
              f"steps={s['pred_steps']}/{s['ref_steps']}")
    print("\nattribution:", dict(attr))
    print("multi-cause tasks:", n_multi)
    print("PHASE8A-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-domain", type=int, default=30)
    args = ap.parse_args()
    main(args.n_per_domain)
