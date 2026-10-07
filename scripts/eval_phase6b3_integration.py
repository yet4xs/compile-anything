"""Phase 6B-3: modular neural compiler integration — internal oracle decomposition.

Arms (corpus-v4 matched tool-use test, 994 tasks):
  I0  E1-A monolithic baseline (bare prompt, frozen)
  I1  Oracle Canonicalizer + Oracle Resolver + C2 Composer   [UPPER-BOUND DIAGNOSTIC]
  I2  Learned Canonicalizer + Oracle Resolver + C2 Composer
  I3  Oracle Canonicalizer + Learned Resolver + C2 Composer
  I4  Learned Canonicalizer + Learned Resolver + C2 Composer
  I4R1 Learned Canonicalizer + R1 generative Resolver + C2 Composer (positive only)

Seed pairing (frozen, no post-hoc mixing): s42 primary, s44 replicate.
Learned Canonicalizer = frozen G1+G2 grounder (schema-only probe A).
Learned Resolver run order: explicit NONE gate first; if NONE -> frontend refusal;
else per-capability R2 score>0 selection; accepted-but-empty recorded separately
(Task 7 — threshold untouched).
Canonicalizer only predicts canonical_skill (Task 6): concrete ids/names/
description/parameters always pass through raw.
Waterfall per-task record (Task 8) + per-arm metrics (Task 10) + error
contribution I1->I2/I3/I4 (Task 9).
"""
import json, os, sys, re, gc
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.skill_grounder import build_probe_prompt, parse_label
from src.ir.parser import parse_text
from src.validator.validator import validate

D = "data/phase6b_resolver"
POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
SYSTEM_R = ("You are a capability resolver for a task compiler. Given a task and "
            "capability descriptions, decide which capabilities the task requires. "
            "Answer with the label ONLY.")


def depth_metrics(pred_ops, ref_ops):
    if ref_ops is None:
        return None
    pc, rc = Counter(pred_ops), Counter(ref_ops)
    inter = sum((pc & rc).values())
    prec = inter / max(1, sum(pc.values()))
    rec = inter / max(1, sum(rc.values()))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"prec": prec, "rec": rec, "f1": f1,
            "ea_recall": (pc & rc)["EXEC_ACTION"] / rc["EXEC_ACTION"]
            if rc.get("EXEC_ACTION") else None,
            "unsupported": sum((pc - rc).values()) / max(1, sum(pc.values()))}


def main(seed=42):
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tasks = [json.loads(l) for l in open(f"{D}/common_eval_tasks.jsonl",
                                         encoding="utf-8") if l.strip()]
    nones = [json.loads(l) for l in open(f"{D}/none_tasks_common.jsonl",
                                         encoding="utf-8") if l.strip()]
    clean_none = [t for t in nones if not t["potential_collision"]]
    print(f"tasks={len(tasks)} clean_none={len(clean_none)} seed={seed}")

    # ── Stage 1: canonicalizer + resolver runs (load each model once) ──
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    # learned canonicalizer: probe-A skill per candidate block
    canon_model = PeftModel.from_pretrained(
        base, f"runs/phase6/grounder_g1g2_s{seed}/final")
    canon_model.eval()
    learned_skill = {}   # (task_id, capability_id) -> skill
    canon_prompts, canon_keys = [], []
    for t in tasks:
        for c in t["candidates"]:
            name_m = re.search(r"name: (\S+)", c["block"].split("\n")[0])
            desc = ""
            dm = re.search(r"description: (.*)", c["block"])
            if dm:
                desc = dm.group(1)[:200]
            p = build_probe_prompt({"instruction": "",
                                    "capability": {"name": name_m.group(1),
                                                   "description": desc,
                                                   "parameters": {}}}, "A")
            canon_prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": p["system"]},
                 {"role": "user", "content": p["user"]}],
                tokenize=False, add_generation_prompt=True))
            canon_keys.append((t["id"], c["capability_id"]))
    for bs in range(0, len(canon_prompts), 32):
        inputs = tok(canon_prompts[bs:bs+32], return_tensors="pt", padding=True,
                     truncation=True, max_length=1024).to(canon_model.device)
        with torch.no_grad():
            o = canon_model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                     temperature=None, pad_token_id=tok.pad_token_id)
        outs = [parse_label(tok.decode(x[inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)) for x in o]
        for k, s in zip(canon_keys[bs:bs+32], outs):
            learned_skill[k] = s
    del canon_model
    gc.collect(); torch.cuda.empty_cache()
    print("canonicalizer done", flush=True)

    # learned resolver R2: explicit NONE gate + per-pair scores
    res_model = PeftModel.from_pretrained(base, f"runs/phase6b/resolver_R2_s{seed}/final")
    res_model.eval()

    def gen(model, prompts, max_new=24, batch=32):
        outs = []
        for bs in range(0, len(prompts), batch):
            inputs = tok(prompts[bs:bs+batch], return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                o = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                   temperature=None, pad_token_id=tok.pad_token_id)
            outs.extend(tok.decode(x[inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True) for x in o)
        return outs

    def pair_scores(model, task, cands):
        """Sequence-likelihood scores (same method as the 6B-2.1 rescore)."""
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_R},
             {"role": "user", "content":
              "Task:\n" + task + "\n\nCapability:\n" + c["block"] +
              "\n\nIs this capability required by the task? "
              "Answer 'relevant' or 'irrelevant'."}],
            tokenize=False, add_generation_prompt=True) for c in cands]
        scores = []
        K = 4  # keep logits for only the last K positions (labels occupy <=2)
        for bs in range(0, len(prompts), 4):
            chunk = prompts[bs:bs+4]
            seqs, kinds = [], []
            for p in chunk:
                seqs.append(tok(p + "relevant", return_tensors="pt"))
                seqs.append(tok(p + "irrelevant", return_tensors="pt"))
                kinds.extend(["r", "i"])
            maxlen = max(s["input_ids"].shape[1] for s in seqs)
            pad = tok.pad_token_id
            ii, am = [], []
            for s in seqs:
                n = maxlen - s["input_ids"].shape[1]
                ii.append([pad] * n + s["input_ids"][0].tolist())
                am.append([0] * n + s["attention_mask"][0].tolist())
            ii = torch.tensor(ii).to(model.device)
            am = torch.tensor(am).to(model.device)
            with torch.no_grad():
                logits = model(input_ids=ii, attention_mask=am,
                               num_logits_to_keep=K).logits
            # returned logits cover global positions [maxlen-K, maxlen-1];
            # logits[idx] predicts the token at global position maxlen-K+idx+1
            r_ids = tok.encode("relevant", add_special_tokens=False)
            i_ids = tok.encode("irrelevant", add_special_tokens=False)
            seq_lp = []
            for j in range(len(seqs)):
                lab_len = len(r_ids) if kinds[j] == "r" else len(i_ids)
                total = 0.0
                for t_pos in range(maxlen - lab_len, maxlen):
                    # predictor position p = t_pos-1; returned index:
                    idx = t_pos - 1 - (maxlen - K)
                    row_lp = torch.log_softmax(logits[j, idx].float(), -1)
                    total += float(row_lp[ii[j, t_pos]])
                seq_lp.append(total)
            for j in range(0, len(seqs), 2):
                scores.append(seq_lp[j] - seq_lp[j + 1])
        return scores

    gate_prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_R},
         {"role": "user", "content":
          f"Task:\n{t['task']}\n\nAvailable capabilities:\n" +
          "\n\n".join(c["block"] for c in t["candidates"][:15]) +
          "\n\nWhich capabilities does the task require? "
          "Answer as a comma-separated id list (e.g. c3,c7) or NONE."}],
        tokenize=False, add_generation_prompt=True) for t in tasks]
    gate_outs = gen(res_model, gate_prompts)
    learned_selected = {}
    gate_refused = set()
    accepted_but_empty = set()
    # 6B-2.1 verdict: the explicit gate is NOT deployable (format shortcut,
    # 99% false refusal on positives). Per the frozen decision, the learned
    # resolver chain runs WITHOUT the gate: pairwise selection on every task.
    # Gate outputs are recorded for the waterfall only.
    for t, o in zip(tasks, gate_outs):
        ol = o.lower()
        if "none" in ol and not re.findall(r"\bc\d+\b", ol):
            gate_refused.add(t["id"])  # recorded, NOT enforced
    for t in tasks:
        cands = [c for c in t["candidates"]]
        scores = pair_scores(res_model, t["task"], cands)
        sel = {c["capability_id"] for c, s in zip(cands, scores) if s > 0}
        if not sel:
            accepted_but_empty.add(t["id"])
        learned_selected[t["id"]] = sel
    print(f"resolver done: refused={len(gate_refused)} accepted_empty="
          f"{len(accepted_but_empty)}", flush=True)

    # NONE-side gate results
    none_gate_prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_R},
         {"role": "user", "content":
          f"Task:\n{t['task']}\n\nAvailable capabilities:\n" +
          "\n\n".join(c["block"] for c in t["candidates"][:15]) +
          "\n\nWhich capabilities does the task require? "
          "Answer as a comma-separated id list (e.g. c3,c7) or NONE."}],
        tokenize=False, add_generation_prompt=True) for t in clean_none]
    none_gate_outs = gen(res_model, none_gate_prompts)
    none_refusals = sum(1 for o in none_gate_outs
                        if "none" in o.lower() and not re.findall(r"\bc\d+\b", o.lower()))
    pos_false_refusals = len(gate_refused)
    del res_model
    gc.collect(); torch.cuda.empty_cache()

    # R1 resolver (for I4R1)
    r1_model = PeftModel.from_pretrained(base, f"runs/phase6b/resolver_R1_s{seed}/final")
    r1_model.eval()
    r1_selected = {}
    r1_outs = gen(r1_model, gate_prompts)
    for t, o in zip(tasks, r1_outs):
        ol = o.lower()
        r1_selected[t["id"]] = set() if "none" in ol else set(re.findall(r"\bc\d+\b", ol))
    del r1_model
    gc.collect(); torch.cuda.empty_cache()

    # ── Stage 2: compose per arm (composer loaded once per arm config) ──
    def composer_prompt(t_dict, chosen, canon_mode):
        blocks = []
        for c in chosen:
            if canon_mode == "oracle":
                skill_m = re.search(r"canonical_skill: (\S+)", c["block"])
                skill = skill_m.group(1) if skill_m else "UNKNOWN"
            else:
                skill = learned_skill.get((t_dict["id"], c["capability_id"])) or "UNKNOWN"
            nm = re.search(r"name: (\S+)", c["block"].split("\n")[0]).group(1)
            blocks.append(f"[{c['capability_id']}] name: {nm}\ncanonical_skill: {skill}")
        return f"{t_dict['task']}\n\nSelected capabilities:\n" + "\n\n".join(blocks)

    composer = PeftModel.from_pretrained(base, f"runs/phase6b/composer_C2_s{seed}/final")
    composer.eval()

    def compose_and_score(chosen_fn, canon_mode, arm_name):
        prompts, kept = [], []
        for t in tasks:
            chosen = chosen_fn(t)
            if not chosen:
                continue
            kept.append(t)
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": composer_prompt(t, chosen, canon_mode)}],
                tokenize=False, add_generation_prompt=True))
        outs = gen(composer, prompts, max_new=512, batch=16)
        g = Counter()
        wf = []
        for t, o in zip(kept, outs):
            gold = set(t["gold_ids"])
            # resolver metrics computed by caller via chosen set
            try:
                ref_ops = [x.op for x in parse_text(t.get("plan_target", "")).program.nodes
                           if x.op not in POLICY_OPS]
            except Exception:
                ref_ops = None
            pred_ops = None
            row = {"id": t["id"], "resolver_exact": None, "composer_parse": False,
                   "composer_valid": False, "composer_opseq": False,
                   "semantic_skill_match": False, "exec_action_match": None,
                   "final_success": False}
            try:
                mod = parse_text(extract_taskir_text(o))
                row["composer_parse"] = True
                pred_ops = [x.op for x in mod.program.nodes if x.op not in POLICY_OPS]
            except Exception:
                pass
            if pred_ops is not None:
                try:
                    row["composer_valid"] = validate(mod).valid
                except Exception:
                    pass
            g["n"] += 1
            g["parse"] += row["composer_parse"]
            g["valid"] += row["composer_valid"]
            if row["composer_valid"] and ref_ops is not None:
                row["composer_opseq"] = pred_ops == ref_ops
                g["opseq"] += row["composer_opseq"]
                dm = depth_metrics(pred_ops, ref_ops)
                g["skill_f1_sum"] += dm["f1"]
                g["skill_micro_inter"] += dm["prec"] * sum(Counter(pred_ops).values())
                g["skill_micro_pred"] += sum(Counter(pred_ops).values())
                g["skill_micro_ref"] += sum(Counter(ref_ops).values())
                if dm["ea_recall"] is not None:
                    g["ea_tasks"] += 1
                    g["ea_sum"] += dm["ea_recall"]
                    row["exec_action_match"] = dm["ea_recall"] >= 1.0
                g["unsupported_sum"] += dm["unsupported"]
                unsup_ok = dm["unsupported"] <= 0.05
                row["semantic_skill_match"] = dm["f1"] >= 0.99
                row["final_success"] = (row["composer_opseq"]
                                        and row["semantic_skill_match"]
                                        and (dm["ea_recall"] is None or dm["ea_recall"] >= 1.0))
                g["final"] += row["final_success"]
            wf.append(row)
        n = max(1, g["n"])
        res = {
            "n": g["n"],
            "parse_pct": round(100 * g["parse"] / n, 2),
            "valid_pct": round(100 * g["valid"] / n, 2),
            "opseq_pct": round(100 * g["opseq"] / n, 2),
            "skill_f1_macro": round(g["skill_f1_sum"] / n, 4),
            "ea_recall_taskavg_pct": round(100 * g["ea_sum"] / max(1, g["ea_tasks"]), 2),
            "unsupported_avg": round(g["unsupported_sum"] / n, 4),
            "final_success_pct": round(100 * g["final"] / n, 2),
        }
        print(f"[{arm_name}] {json.dumps(res)}", flush=True)
        return res, wf

    results = {}
    waterfalls = {}

    def chosen_oracle(t):
        return [c for c in t["candidates"] if c["is_gold"]]

    def chosen_learned(t):
        return [c for c in t["candidates"] if c["capability_id"] in learned_selected.get(t["id"], set())]

    def chosen_r1(t):
        return [c for c in t["candidates"] if c["capability_id"] in r1_selected.get(t["id"], set())]

    # resolver metrics for learned arm
    res_exact = res_f1_sum = 0
    for t in tasks:
        sel = learned_selected.get(t["id"], set())
        gold = set(t["gold_ids"])
        tp = len(sel & gold)
        p = tp / max(1, len(sel))
        r = tp / max(1, len(gold))
        res_f1_sum += 2 * p * r / max(1e-9, p + r)
        res_exact += sel == gold
    results["_resolver_learned"] = {"set_f1": round(res_f1_sum / len(tasks), 4),
                                     "exact": round(res_exact / len(tasks), 4),
                                     "gate_refused_positive": pos_false_refusals,
                                     "accepted_but_empty": len(accepted_but_empty)}
    results["_gate_none"] = {"clean_none_n": len(clean_none),
                              "correct_refusals": none_refusals,
                              "note": "synthetic refusal diagnostic"}

    for arm, cfn, cmode in (("I1_oracle_oracle", chosen_oracle, "oracle"),
                             ("I2_learnedcanon_oracle", chosen_oracle, "learned"),
                             ("I3_oraclecanon_learnedres", chosen_learned, "oracle"),
                             ("I4_learned_learned", chosen_learned, "learned"),
                             ("I4R1_learned_r1", chosen_r1, "oracle")):
        results[arm], waterfalls[arm] = compose_and_score(
            lambda t, f=cfn: f(t), cmode, arm)

    # I0: E1-A monolithic on the same tasks (bare)
    e1a = PeftModel.from_pretrained(base, "runs/phase5b1/e1_3b_qlora/final")
    e1a.eval()
    prompts0 = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": t["task"]}],
        tokenize=False, add_generation_prompt=True) for t in tasks]
    outs0 = gen(e1a, prompts0, max_new=512, batch=16)
    g = Counter()
    for t, o in zip(tasks, outs0):
        g["n"] += 1
        try:
            ref_ops = [x.op for x in parse_text(t.get("plan_target", "")).program.nodes
                       if x.op not in POLICY_OPS]
        except Exception:
            ref_ops = None
        try:
            mod = parse_text(extract_taskir_text(o))
            g["parse"] += 1
            pred_ops = [x.op for x in mod.program.nodes if x.op not in POLICY_OPS]
            if validate(mod).valid:
                g["valid"] += 1
                if ref_ops is not None:
                    g["opseq"] += pred_ops == ref_ops
                    dm = depth_metrics(pred_ops, ref_ops)
                    g["skill_f1_sum"] += dm["f1"]
                    if dm["ea_recall"] is not None:
                        g["ea_tasks"] += 1
                        g["ea_sum"] += dm["ea_recall"]
        except Exception:
            pass
    n = max(1, g["n"])
    results["I0_e1a_monolithic"] = {
        "n": n, "parse_pct": round(100*g["parse"]/n, 2),
        "valid_pct": round(100*g["valid"]/n, 2), "opseq_pct": round(100*g["opseq"]/n, 2),
        "skill_f1_macro": round(g["skill_f1_sum"]/n, 4),
        "ea_recall_taskavg_pct": round(100*g["ea_sum"]/max(1, g["ea_tasks"]), 2),
    }
    print(f"[I0_e1a_monolithic] {json.dumps(results['I0_e1a_monolithic'])}", flush=True)

    os.makedirs("results/phase6b", exist_ok=True)
    out_name = f"results/phase6b/integration_s{seed}.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump({"results": results, "waterfalls": waterfalls}, f, indent=2, ensure_ascii=False)
    print(f"saved {out_name}")
    print(f"INTEGRATION-S{seed}-DONE")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
