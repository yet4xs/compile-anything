"""Phase 8A.1 Task 4 + 8B-0 Task 7 generations (one GPU pass).

Generates and stores TEXTS + selections only; scoring happens offline
with the corrected v2 evaluator (no rescoring-on-GPU).

Grid:
  8A.1 label counterfactuals:
    A1 = learned selection + learned labels   (exists: 8A run texts)
    A2 = SAME learned selection + ORACLE labels          [NEW]
    C  = gold tool ids + LEARNED labels                  [NEW]
    D  = gold tool ids + oracle labels (old C_G0)        [exists]
  8B-0 arms (composer s42 frozen, base model for P2):
    P2 = base Qwen3B emits JSON skeleton -> deterministic lowering
    P3 = composer + explicit tool-identity instruction
    P4 = composer + length/coverage instruction
"""
import json, os, sys, time, gc, random, argparse, re

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.compiler.prompt_format import SYSTEM_PROMPT
from src.compiler.online_constrained_decoder import generate_constrained
from src.compiler.skill_grounder import build_probe_prompt, parse_label
from src.compiler.plan_skeleton import skeleton_from_dict
from src.compiler.skeleton_lowering import lower_skeleton
from src.eval.tau3_skill_oracle import get_tool_oracle
from src.eval.tau3_tool_registry import build_registry, runtime_pool

SEED = "42"
SAMPLING_SEED = 20261013
DOMS = ["airline", "banking_knowledge", "retail", "telecom"]


def load_tasks(n_per_domain):
    exclude = set()
    try:
        cs = json.load(open("results/phase7/online_cross_source.json"))
        for r in cs.get("details", []):
            exclude.add(r["case_id"])
    except FileNotFoundError:
        pass
    tasks = []
    for d in DOMS:
        data = json.load(open(
            f"third_party/tau3-bench/data/tau2/domains/{d}/tasks.json",
            encoding="utf-8"))
        usable = []
        for t in data:
            acts = (t.get("evaluation_criteria") or {}).get("actions") or []
            if not acts:
                continue
            cid = f"{d}-{t['id']}"
            if cid in exclude:
                continue
            usable.append({"case_id": cid, "domain": d,
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
        def seq_lp(j, lab):
            total = 0.0
            for tp in range(maxlen - len(lab), maxlen):
                idx = tp - 1 - (maxlen - K)
                row = torch.log_softmax(logits[j, idx].float(), -1)
                total += float(row[ii[j, tp]])
            return total
        return seq_lp(0, r_ids) - seq_lp(1, i_ids)

    return [(t["name"], labels.get(t["name"], "UNKNOWN"))
            for t in tools if score_pair(t["name"],
                                         labels.get(t["name"], "UNKNOWN")) > 0]


def skeleton_prompt(task, tools_with_labels):
    sys_p = (
        "You are a task planner. Output ONLY a JSON object with key "
        '"nodes": a list of steps, each {"id": "p0", "capability_id": '
        "<tool>, \"skill\": <skill>, \"depends_on\": [<ids>], "
        '"arguments": {<param>: <value>}}. Plan the COMPLETE ordered '
        "sequence of tool calls. Use ONLY the listed capabilities. "
        "Mutating tools must have skill EXEC_ACTION.")
    caps = []
    for name, skill, eff, params in tools_with_labels:
        ps = ", ".join(f"{p['name']}{'?' if p.get('optional') else ''}"
                       for p in params)
        caps.append(f"- {name} | skill: {skill} | effect: {eff} | "
                    f"params: {ps}")
    user = (f"Task:\n{task}\n\nAvailable capabilities:\n"
            + "\n".join(caps) + "\n\nJSON plan:")
    return sys_p, user


def parse_skeleton_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main(n_per_domain=30):
    reg = build_registry()
    pool = runtime_pool(reg)
    schemas = {d: {t["name"]: t for t in ts} for d, ts in pool.items()}
    tasks = load_tasks(n_per_domain)
    print(f"tasks: {len(tasks)}")

    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    # learned labels (deterministic, matches 8A run)
    canon = PeftModel.from_pretrained(
        base, f"runs/phase6/grounder_g1g2_s{SEED}/final")
    canon.eval()
    learned = {}
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
                learned[(d, t["name"])] = parse_label(tok.decode(
                    o[inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True))
    del canon
    gc.collect()
    torch.cuda.empty_cache()
    print("labels done")

    oracle = {(d, t["name"]): get_tool_oracle(d, t["name"])["semantic_skill"]
              for d, ts in pool.items() for t in ts}

    resolver = PeftModel.from_pretrained(
        base, f"runs/phase6b/resolver_R2_s{SEED}/final")
    resolver.eval()
    selA = {}
    for i, t in enumerate(tasks):
        if i % 10 == 0:
            print(f"  resolver {i}/{len(tasks)}", flush=True)
        d = t["domain"]
        labels = {x["name"]: learned[(d, x["name"])] for x in pool[d]}
        selA[t["case_id"]] = resolve_selection(resolver, tok,
                                               t["instruction"],
                                               pool[d], labels)
    del resolver
    gc.collect()
    torch.cuda.empty_cache()
    print("resolver done")

    composer = PeftModel.from_pretrained(
        base, f"runs/phase6b/composer_C2_s{SEED}/final")
    composer.eval()

    fout = open("results/phase8a/grid_texts.jsonl", "a", encoding="utf-8")

    def emit(rec):
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

    def compose(user, model, level=0, maxtok=512):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        text, _st, _ = generate_constrained(model, tok, prompt, level,
                                            max_new_tokens=maxtok)
        return text, round(time.time() - t0, 2)

    for i, t in enumerate(tasks):
        d, cid = t["domain"], t["case_id"]
        sel = selA[cid]
        oracle_labels = {x["name"]: oracle[(d, x["name"])] for x in pool[d]}
        gold = []
        seen = set()
        for a in t["ref_actions"]:
            n = a["name"]
            if n not in seen:
                seen.add(n)
                gold.append((n, oracle.get((d, n))
                             or get_tool_oracle(d, n)["semantic_skill"]
                             or "EXEC_ACTION"))

        def blocks_of(sel_list):
            return [f"[c{i}] name: {n}\ncanonical_skill: {s}"
                    for i, (n, s) in enumerate(sel_list)]

        if i % 10 == 0:
            print(f"  gen {i}/{len(tasks)}", flush=True)

        # A2: learned selection + oracle labels
        selA2 = [(n, oracle_labels.get(n, "UNKNOWN")) for n, _ in sel]
        if selA2:
            text, lat = compose(
                f"{t['instruction']}\n\nSelected capabilities:\n"
                + "\n\n".join(blocks_of(selA2)), composer)
            emit({"arm": "A2_G0", "case_id": cid, "text": text,
                  "latency_s": lat, "selected": [n for n, _ in selA2]})
        else:
            emit({"arm": "A2_G0", "case_id": cid, "text": "",
                  "latency_s": 0.0, "selected": []})

        # C: gold tools + learned labels
        goldC = [(n, learned.get((d, n), "UNKNOWN")) for n, _ in gold]
        text, lat = compose(
            f"{t['instruction']}\n\nSelected capabilities:\n"
            + "\n\n".join(blocks_of(goldC)), composer)
        emit({"arm": "C_learned_labels_G0", "case_id": cid, "text": text,
              "latency_s": lat, "selected": [n for n, _ in goldC]})

        # P3: identity instruction on the A1 selection
        if sel:
            user = (f"{t['instruction']}\n\nSelected capabilities:\n"
                    + "\n\n".join(blocks_of(sel))
                    + "\n\nIMPORTANT: Emit every tool invocation as "
                    "EXEC_ACTION(action=<capability_name>, <parameters>), "
                    "preserving the exact capability name.")
            text, lat = compose(user, composer)
            emit({"arm": "P3_identity", "case_id": cid, "text": text,
                  "latency_s": lat, "selected": [n for n, _ in sel]})

        # P4: length/coverage instruction on the A1 selection
        if sel:
            user = (f"{t['instruction']}\n\nSelected capabilities:\n"
                    + "\n\n".join(blocks_of(sel))
                    + "\n\nIMPORTANT: Produce a COMPLETE multi-step plan "
                    "with one node for EVERY capability that the task "
                    "requires, in correct execution order. Do not stop "
                    "after the first step.")
            text, lat = compose(user, composer)
            emit({"arm": "P4_length", "case_id": cid, "text": text,
                  "latency_s": lat, "selected": [n for n, _ in sel]})

        # P2: base model emits skeleton JSON -> deterministic lowering
        tools_meta = [(x["name"], oracle_labels.get(x["name"], "UNKNOWN"),
                       get_tool_oracle(d, x["name"])["effect_class"]
                       or "read_only", x["params"]) for x in pool[d]]
        sys_p, user = skeleton_prompt(t["instruction"], tools_meta)
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=2048).to(base.device)
        with torch.no_grad():
            out = base.generate(**inputs, max_new_tokens=512,
                                do_sample=False, temperature=None,
                                pad_token_id=tok.pad_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        lat = round(time.time() - t0, 2)
        sk_json = parse_skeleton_json(raw)
        lowered = None
        sk_err = None
        if sk_json is not None:
            sk = skeleton_from_dict(sk_json)
            low = lower_skeleton(sk, d, schemas[d])
            lowered = low.get("taskir")
            sk_err = low["errors"][:2] or None
        emit({"arm": "P2_skeleton", "case_id": cid,
              "text": lowered or "", "skeleton_json_raw": raw[:2000],
              "skeleton_parse_ok": sk_json is not None,
              "lower_errors": sk_err,
              "latency_s": lat, "selected": None})

    fout.close()
    print("GRID-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-domain", type=int, default=30)
    args = ap.parse_args()
    main(args.n_per_domain)
