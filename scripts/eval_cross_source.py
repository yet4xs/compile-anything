"""Phase 7C-2.1 Task 4: cross-source validation on tau3 (never used for
constraint development).

Protocols:
  A (oracle capabilities): capability blocks = the case's gold reference
     action tools, labeled by the frozen learned canonicalizer. Composer
     arms G0/G1/G2.
  B (learned capabilities): full frozen modular chain as in phase 6B-3 —
     canonicalizer over env-visible tools -> resolver selection -> Composer
     arms G0/G1/G2. No task-specific gold anywhere in the pipeline.

tau3 has no TaskIR references, so metrics are: Parse%, Valid%,
EXEC_ACTION output rate, and EXEC_ACTION reference recall (via
src/eval/tau3_skill_oracle, same as the 6B-3 60.23% baseline).
Composer uses batch=1 greedy + the optimized online decoder.
"""
import json, os, sys, re, gc, time, random, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.skill_grounder import build_probe_prompt, parse_label
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.compiler.online_constrained_decoder import generate_constrained

TAU3_SRC = "/ccfa2026/compile-anything/third_party/tau3-bench/src/tau2/domains"
SEED = 42                     # frozen primary adapters (as in 6B-3)
SAMPLING_SEED = 20261012
ARMS = [("G0", 0), ("G1", 1), ("G2", 3)]


def extract_domain_tools(domain):
    """Env-visible tools (source @is_tool + ref-name union) — 6B-3 logic."""
    ref_names = set()
    with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["case_id"].split("-")[0] == domain:
                ref_names.update(r.get("ref_action_names") or [])
    tools = [{"name": n, "description": ""} for n in sorted(ref_names)]
    path = os.path.join(TAU3_SRC, domain, "tools.py")
    if os.path.exists(path):
        src = open(path, encoding="utf-8").read()
        seen = {t["name"] for t in tools}
        for m in re.finditer(
                r"@is_tool[^\n]*\n(?:[^\n]*\n){0,4}?\s*def (\w+)\(", src):
            if m.group(1) not in seen:
                tools.append({"name": m.group(1), "description": ""})
    return tools


def exec_action_recall(text, ref_action_names):
    """EXEC_ACTION recall vs gold tool names, via the 6B-3 oracle lowering
    (semantic_match on lowered references); count-limited output rate."""
    try:
        mod = parse_text(extract_taskir_text(text))
    except Exception:
        return None, 0
    n_exec = sum(1 for n in mod.program.nodes if n.op == "EXEC_ACTION")
    refs = [r for r in (ref_action_names or []) if r]
    if not refs:
        return None, n_exec
    from src.eval.tau3_skill_oracle import (lower_reference_actions,
                                             semantic_match)
    lowered = lower_reference_actions(refs)
    hits = 0
    for ref in lowered:
        if any(n.op == "EXEC_ACTION" and semantic_match(
                "EXEC_ACTION", dict(n.params), ref)
                for n in mod.program.nodes):
            hits += 1
    return round(hits / len(lowered), 4), n_exec


def judge(text, ref_action_names):
    out = {"parse_ok": False, "valid": False, "error": None,
           "exec_recall": None, "n_exec": 0}
    try:
        mod = parse_text(extract_taskir_text(text))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:150]
        return out
    out["parse_ok"] = True
    rep = validate(mod)
    out["valid"] = rep.valid
    if not rep.valid:
        out["error"] = "; ".join(str(e) for e in rep.errors[:2])[:150]
    er, n_exec = exec_action_recall(text, ref_action_names)
    out["exec_recall"] = er
    out["n_exec"] = n_exec
    return out


def main(n=100):
    print("=" * 66)
    print(f"Phase 7C-2.1 Task 4: cross-source tau3  n={n}")
    print("=" * 66)

    # ---- tasks (stratified by domain) ----
    all_tasks = []
    with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            all_tasks.append({"case_id": r["case_id"],
                              "domain": r["case_id"].split("-")[0],
                              "instruction": r["instruction"],
                              "ref_action_names": r.get("ref_action_names") or []})
    by_dom = defaultdict(list)
    for t in all_tasks:
        by_dom[t["domain"]].append(t)
    rng = random.Random(SAMPLING_SEED)
    tasks = []
    for d in sorted(by_dom):
        members = sorted(by_dom[d], key=lambda x: x["case_id"])
        rng.shuffle(members)
        tasks.extend(members[:round(n * len(members) / len(all_tasks))])
    tasks = tasks[:n]
    print(f"sampled {len(tasks)} tau3 tasks: "
          f"{dict(Counter(t['domain'] for t in tasks))}")

    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    # ---- Stage 1: canonicalizer labels for env tools ----
    domains = sorted({t["domain"] for t in tasks})
    domain_tools = {d: extract_domain_tools(d) for d in domains}
    canon = PeftModel.from_pretrained(
        base, f"runs/phase6/grounder_g1g2_s{SEED}/final")
    canon.eval()
    tool_skill = {}
    for d, tools in domain_tools.items():
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
            preds = [parse_label(tok.decode(
                o[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)) for o in out]
            for t, s in zip(tools[bs:bs + 16], preds):
                tool_skill[(d, t["name"])] = s
    del canon
    gc.collect()
    torch.cuda.empty_cache()
    print("canonicalizer done")

    # ---- Protocol A blocks: gold reference tools ----
    protoA = {}
    for t in tasks:
        d = t["domain"]
        names = [n for n in dict.fromkeys(t["ref_action_names"])
                 if (d, n) in tool_skill] or list(t["ref_action_names"])[:5]
        protoA[t["case_id"]] = [(n, tool_skill.get((d, n), "UNKNOWN"))
                                for n in names]

    # ---- Protocol B: learned resolver selection (6B-3 logic) ----
    SYSTEM_R = ("You are a capability resolver for a task compiler. Given a "
                "task and capability descriptions, decide which capabilities "
                "the task requires. Answer with the label ONLY.")
    resolver = PeftModel.from_pretrained(
        base, f"runs/phase6b/resolver_R2_s{SEED}/final")
    resolver.eval()
    r_ids = tok.encode("relevant", add_special_tokens=False)
    i_ids = tok.encode("irrelevant", add_special_tokens=False)

    def score_pair(task_text, tool_name, skill):
        block = f"[c0] name: {tool_name}\ncanonical_skill: {skill}"
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_R},
             {"role": "user", "content":
              f"Task:\n{task_text}\n\nCapability:\n{block}\n\n"
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

    protoB = {}
    for i, t in enumerate(tasks):
        if i % 20 == 0:
            print(f"  resolver {i}/{len(tasks)}", flush=True)
        d = t["domain"]
        scored = []
        for tool in domain_tools[d]:
            skill = tool_skill.get((d, tool["name"]), "UNKNOWN")
            scored.append((score_pair(t["instruction"], tool["name"], skill),
                           tool["name"], skill))
        pos = [(n, s) for sc, n, s in scored if sc > 0]
        if not pos:
            scored.sort(reverse=True)
            pos = [(scored[0][1], scored[0][2])] if scored else []
        protoB[t["case_id"]] = pos
    del resolver
    gc.collect()
    torch.cuda.empty_cache()
    print("resolver done")

    # ---- Composer arms ----
    composer = PeftModel.from_pretrained(
        base, f"runs/phase6b/composer_C2_s{SEED}/final")
    composer.eval()

    results = {"n": len(tasks), "sampling_seed": SAMPLING_SEED,
               "protocols": {}}
    details = []
    for proto_name, sel_map in [("A_oracle", protoA), ("B_learned", protoB)]:
        print(f"\n=== Protocol {proto_name} ===", flush=True)
        proto_res = {}
        per_arm_case = {}                 # arm -> {case_id: judge}
        for arm, level in ARMS:
            stats = Counter()
            lat = []
            per_arm_case[arm] = {}
            for t in tasks:
                sel = sel_map[t["case_id"]]
                blocks = [f"[c{i}] name: {n}\ncanonical_skill: {s}"
                          for i, (n, s) in enumerate(sel)]
                user = (f"{t['instruction']}\n\nSelected capabilities:\n"
                        + "\n\n".join(blocks))
                prompt = tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True)
                t0 = time.time()
                text, cst, _ = generate_constrained(
                    composer, tok, prompt, level, max_new_tokens=512)
                lat.append(round(time.time() - t0, 2))
                j = judge(text, t["ref_action_names"])
                stats["n"] += 1
                stats["parse"] += j["parse_ok"]
                stats["valid"] += j["valid"]
                stats["exec_out"] += j["n_exec"] > 0
                if j["exec_recall"] is not None:
                    stats["rec_n"] += 1
                    stats["rec_sum"] += j["exec_recall"]
                per_arm_case[arm][t["case_id"]] = j
                details.append({"protocol": proto_name, "arm": arm,
                                "case_id": t["case_id"], "text": text,
                                "judge": j})
            dn = max(1, stats["n"])
            proto_res[arm] = {
                "n": stats["n"],
                "parse_pct": round(100 * stats["parse"] / dn, 1),
                "valid_pct": round(100 * stats["valid"] / dn, 1),
                "exec_action_output_pct": round(100 * stats["exec_out"] / dn, 1),
                "exec_recall_mean": (round(stats["rec_sum"] / stats["rec_n"], 4)
                                     if stats["rec_n"] else None),
                "latency_p50": sorted(lat)[len(lat) // 2],
            }
            r = proto_res[arm]
            print(f"  {arm}: parse={r['parse_pct']}% valid={r['valid_pct']}% "
                  f"exec_out={r['exec_action_output_pct']}% "
                  f"rec={r['exec_recall_mean']} p50={r['latency_p50']}s",
                  flush=True)
        trans = {}
        for arm, _ in ARMS[1:]:
            fixed = broken = 0
            for cid, ja in per_arm_case[arm].items():
                j0 = per_arm_case["G0"].get(cid)
                if j0 is None:
                    continue
                if ja["valid"] and not j0["valid"]:
                    fixed += 1
                elif j0["valid"] and not ja["valid"]:
                    broken += 1
            trans[arm] = {"fixed": fixed, "broken": broken}
        proto_res["transitions_vs_G0"] = trans
        results["protocols"][proto_name] = proto_res

    results["details"] = details
    os.makedirs("results/phase7", exist_ok=True)
    with open("results/phase7/online_cross_source.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nsaved results/phase7/online_cross_source.json")
    print("CROSS-SOURCE-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    main(args.n)
