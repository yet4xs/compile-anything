"""Phase 8D-1B: GPU training + evaluation for M0/M1 dataflow pilot.

Runs on the server: training (QLoRA) then paired evaluation.
Usage: nohup python scripts/run_dataflow_pilot.py > pilot.log 2>&1 &
"""
import json, os, sys, time, re, gc, random, math, hashlib, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, PeftModel

from src.ir.parser import parse_text
from src.validator.validator import validate
from src.compiler.plan_skeleton import skeleton_from_dict, validate_skeleton, hard_errors
from src.compiler.skeleton_lowering import lower_skeleton
from src.runtime.mock_capability_registry import MockCapabilityRegistry, MockCapability, DataflowRuntime

DATA = "data/runtime_dataflow_v2"
OUTR = "results/phase8d"

SYS_M0 = ("You are a task compiler. Given a task and capabilities with "
          "schemas, emit TaskIR v0.2 using EXTRACT_FIELD and bind() for "
          "runtime parameter binding. Output ONLY TaskIR.\n"
          '%id = EXEC_ACTION(action="tool", literals...)\n'
          '%fid = EXTRACT_FIELD(%producer, field="name") -> Type\n'
          '%id = EXEC_ACTION(action="tool")  bind(param=%fid)\n'
          'return %last')
SYS_M1 = ("You are a semantic task planner. Given a task and capabilities "
          "with schemas, emit ONLY a JSON skeleton. Use runtime_ref for "
          "params from upstream outputs:\n"
          '{"nodes": [{"id":"p0", "capability_id":"...", '
          '"skill":"EXEC_ACTION", "depends_on":[], "arguments":{'
          '"param": value or {"kind":"runtime_ref", "producer":"pX", '
          '"field":"output_field"}}}]}')


def caps_block(rec, protocol="distractor"):
    ids = rec.get("offered_capability_ids", [])
    if protocol == "oracle":
        ids = rec.get("gold_capability_ids", ids)
    lines = []
    for tid in ids:
        meta = rec["capability_registry"].get(tid, {})
        ins = meta.get("input_schema", {})
        outs = meta.get("output_schema", {})
        ins_s = ", ".join(f"{k}:{v}" for k, v in ins.items()) or "-"
        outs_s = ", ".join(f"{k}:{v}" for k, v in outs.items()) or "-"
        lines.append(f"- {tid} | in: {ins_s} | out: {outs_s}")
    return "\n".join(lines)


def build_prompt(rec, protocol="distractor"):
    return (f"Task:\n{rec['instruction']}\n\n"
            f"Available capabilities:\n{caps_block(rec, protocol)}")


def build_targets(rec):
    """M0 = TaskIR, M1 = Skeleton JSON."""
    m0_target = rec.get("m0_taskir", "")
    m1_target = json.dumps(rec["target_skeleton"], ensure_ascii=False)
    return m0_target, m1_target


class SFTData(Dataset):
    def __init__(self, rows, tok, sys_prompt, max_len=2048):
        self.rows, self.tok, self.sys, self.ml = rows, tok, sys_prompt, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": self.sys},
             {"role": "user", "content": r["user"]}],
            tokenize=False, add_generation_prompt=True)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tok(r["target"], add_special_tokens=False)["input_ids"] \
            + [self.tok.eos_token_id]
        ids = (p_ids + t_ids)[-self.ml:]
        labels = ([-100] * len(p_ids) + t_ids)[-self.ml:]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}


def collate(batch, pad_id):
    mx = max(len(b["input_ids"]) for b in batch)
    ii, ll, aa = [], [], []
    for b in batch:
        n = mx - len(b["input_ids"])
        ii.append(b["input_ids"] + [pad_id] * n)
        ll.append(b["labels"] + [-100] * n)
        aa.append(b["attention_mask"] + [0] * n)
    return {"input_ids": torch.tensor(ii),
            "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(aa)}


def train_model(model_name, train_rows, tok, base, seed, out_dir):
    sys_map = {"m0": SYS_M0, "m1": SYS_M1}
    sys_p = sys_map[model_name]

    rows = []
    for rec in train_rows:
        user = build_prompt(rec, protocol="distractor")
        m0_t, m1_t = build_targets(rec)
        target = m0_t if model_name == "m0" else m1_t
        rows.append({"user": user, "target": target,
                     "task_id": rec["task_id"]})

    t_tokens = sum(len(tok(r["target"], add_special_tokens=False)["input_ids"])
                   for r in rows)
    print(f"{model_name}: {len(rows)} samples, {t_tokens} target tokens")

    bnb = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        quantization_config=bnb, device_map="auto")
    base.config.use_cache = False
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                       task_type="CAUSAL_LM",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()

    eff_batch = 16
    per_dev = 1
    accum = eff_batch
    steps_per_epoch = math.ceil(len(rows) / eff_batch)
    max_steps = int(steps_per_epoch * 3)  # 3 epochs (small dataset)

    targs = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=accum, num_train_epochs=1.0,
        max_steps=max_steps, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_strategy="no", bf16=True,
        optim="paged_adamw_32bit", report_to=[], seed=seed,
        max_grad_norm=1.0)
    ds = SFTData(rows, tok, sys_p)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                       data_collator=lambda b: collate(b, tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    os.makedirs(f"{out_dir}/final", exist_ok=True)
    model.save_pretrained(f"{out_dir}/final")
    losses = [h.get("loss") for h in trainer.state.log_history
               if "loss" in h]
    log = {"model": model_name, "seed": seed, "n_samples": len(rows),
           "target_tokens": t_tokens, "max_steps": max_steps,
           "wall_min": round(dt / 60, 1),
           "first_loss": losses[0] if losses else None,
           "last_loss": losses[-1] if losses else None}
    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"saved {out_dir}/final | {dt/60:.1f} min | loss "
          f"{losses[0]:.3f}→{losses[-1]:.3f}")
    del model, base, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return log


def generate_and_parse(model, tok, rec, arm, protocol):
    """Generate one completion and convert to TaskIR."""
    sys_p = SYS_M0 if arm == "M0" else SYS_M1
    user = build_prompt(rec, protocol)
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
    lat = round(time.time() - t0, 2)
    ntok = int(out.shape[1] - ids["input_ids"].shape[1])

    taskir = text
    if arm == "M1":
        # Parse skeleton JSON → lower to TaskIR
        try:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                sk = skeleton_from_dict(json.loads(m.group(0)))
                schemas = {}
                for tid in rec["capability_registry"]:
                    meta = rec["capability_registry"][tid]
                    schemas[tid] = {
                        "params": [{"name": k, "optional": False, "type": v}
                                    for k, v in meta.get("input_schema", {}).items()],
                        "output_fields": meta.get("output_schema", {}),
                        "effect_class": meta.get("effect_class", "read_only")}
                errs = hard_errors(validate_skeleton(sk, None, schemas))
                if errs:
                    taskir = ""
                else:
                    low = lower_skeleton(sk, None, schemas, strict=True)
                    taskir = low.get("taskir") or ""
            else:
                taskir = ""
        except Exception:
            taskir = ""
    return taskir, text, lat, ntok


def extract_bindings_skeleton(nodes):
    b = set()
    for n in nodes:
        for pk, v in (n.get("arguments") or {}).items():
            if isinstance(v, dict) and v.get("kind") == "runtime_ref":
                b.add((v["producer"], v["field"], n["id"], pk))
    return b


def extract_bindings_taskir(taskir_text):
    if not taskir_text:
        return set()
    try:
        mod = parse_text(taskir_text)
    except Exception:
        return set()
    ext_map = {}
    for n in mod.program.nodes:
        if n.op == "EXTRACT_FIELD" and n.inputs:
            ext_map[n.id] = (n.inputs[0], str(n.params.get("field", "")))
    orig = [n for n in mod.program.nodes if n.op != "EXTRACT_FIELD"]
    ssa2p = {n.id: f"p{i}" for i, n in enumerate(orig)}
    b = set()
    for n in orig:
        for pk, ref in n.param_refs.items():
            if ref in ext_map:
                prod, field = ext_map[ref]
                b.add((ssa2p.get(prod, prod), field,
                       ssa2p.get(n.id, n.id), pk))
    return b


def evaluate_one(rec, taskir, protocol):
    res = {"task_id": rec["task_id"], "topology": rec["topology"],
           "parse": False, "validator": False, "tool_match": False,
           "binding_exact": False, "binding_f1": 0.0,
           "execution": False, "strict_semantic": False,
           "n_ref_b": 0, "n_pred_b": 0, "n_ok_b": 0}
    ref_b = extract_bindings_skeleton(rec["target_skeleton"]["nodes"])
    res["n_ref_b"] = len(ref_b)
    gold = sorted(rec.get("gold_capability_ids", []))

    if not taskir:
        return res
    try:
        mod = parse_text(taskir)
        res["parse"] = True
    except Exception:
        return res
    v2 = {t: {"input_schema": rec["capability_registry"][t].get("input_schema", {}),
               "output_schema": rec["capability_registry"][t].get("output_schema", {})}
           for t in rec["capability_registry"]}
    res["validator"] = validate(mod, capability_registry=v2).valid
    pred_tools = sorted({n.params.get("action", "") for n in mod.program.nodes
                          if n.op == "EXEC_ACTION" and n.params.get("action")})
    res["tool_match"] = pred_tools == gold
    pred_b = extract_bindings_taskir(taskir)
    res["n_pred_b"] = len(pred_b)
    res["n_ok_b"] = len(pred_b & ref_b)
    res["binding_exact"] = pred_b == ref_b
    inter = len(pred_b & ref_b)
    if inter:
        p = inter / max(1, len(pred_b))
        r = inter / max(1, len(ref_b))
        res["binding_f1"] = round(2 * p * r / (p + r) if (p + r) else 0, 4)
    if res["validator"]:
        reg = MockCapabilityRegistry()
        for tid in rec["capability_registry"]:
            meta = rec["capability_registry"][tid]
            outs = meta.get("output_schema", {})
            def mk(tid, outs):
                def fn(params):
                    h = int(hashlib.md5(json.dumps(
                        params, sort_keys=True, default=str).encode()
                    ).hexdigest()[:6], 16)
                    r = {}
                    for k, t in outs.items():
                        if t == "Int": r[k] = h % 1000
                        elif t == "Float": r[k] = float(h % 10000) / 100
                        elif t == "Bool": r[k] = h % 2 == 0
                        else: r[k] = f"{tid}_{k}_{h % 9999}"
                    return r
                return fn
            reg.register(MockCapability(
                tid, meta.get("input_schema", {}), outs,
                meta.get("effect_class", "read_only"), mk(tid, outs)))
        rt = DataflowRuntime(reg)
        tr = rt.execute(mod)
        res["execution"] = tr.success
    res["strict_semantic"] = bool(
        res["validator"] and res["tool_match"]
        and res["binding_exact"] and res["execution"])
    return res


def run_evaluation(m0_dir, m1_dir, seed):
    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    test = [json.loads(l) for l in open(f"{DATA}/test.jsonl",
                                         encoding="utf-8") if l.strip()]
    all_results = {"details": []}
    for arm, adapter in (("M0", m0_dir), ("M1", m1_dir)):
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        for protocol in ("oracle", "distractor"):
            for rec in test:
                taskir, raw, lat, ntok = generate_and_parse(
                    model, tok, rec, arm, protocol)
                ev = evaluate_one(rec, taskir, protocol)
                ev.update({"arm": arm, "protocol": protocol,
                           "latency_s": lat, "gen_tokens": ntok})
                ev["raw"] = raw[:2000]
                all_results["details"].append(ev)
            print(f"  {arm}/{protocol} done", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # summary
    by = defaultdict(list)
    for r in all_results["details"]:
        by[(r["arm"], r["protocol"])].append(r)
    summary = {}
    for (arm, proto), rows in sorted(by.items()):
        n = len(rows)
        def m(k):
            v = [x.get(k) for x in rows if x.get(k) is not None]
            return round(sum(v) / len(v), 4) if v else None
        def pct(k):
            return round(100 * sum(1 for x in rows if x.get(k)) / n, 1)
        summary[f"{proto}|{arm}"] = {
            "n": n, "validator%": pct("validator"),
            "tool_match%": pct("tool_match"),
            "binding_exact%": pct("binding_exact"),
            "binding_f1": m("binding_f1"),
            "execution%": pct("execution"),
            "strict_semantic%": pct("strict_semantic"),
            "tokens_mean": round(sum(x["gen_tokens"] for x in rows) / n, 1),
            "latency_p50": sorted(x["latency_s"] for x in rows)[n // 2]}
    # paired
    for proto in ("oracle", "distractor"):
        m0 = {x["task_id"]: x for x in all_results["details"]
               if x["arm"] == "M0" and x["protocol"] == proto}
        m1 = {x["task_id"]: x for x in all_results["details"]
               if x["arm"] == "M1" and x["protocol"] == proto}
        for key in ("binding_exact", "strict_semantic"):
            deltas = []
            fx = bk = 0
            for tid in m0:
                if tid not in m1:
                    continue
                a = 1 if m1[tid][key] else 0
                b = 1 if m0[tid][key] else 0
                deltas.append(a - b)
                if a and not b:
                    fx += 1
                elif b and not a:
                    bk += 1
            if deltas:
                rng = random.Random(13)
                boots = sorted(
                    sum(deltas[rng.randrange(len(deltas))]
                        for _ in deltas) / len(deltas)
                    for _ in range(5000))
                summary[f"{proto}|paired|{key}"] = {
                    "n": len(deltas),
                    "diff_pp": round(100 * sum(deltas) / len(deltas), 1),
                    "ci95": [round(100 * boots[125], 1),
                              round(100 * boots[4875], 1)],
                    "fixed": fx, "broken": bk}
    all_results["summary"] = summary
    os.makedirs(OUTR, exist_ok=True)
    with open(f"{OUTR}/m0_m1_dataflow_pilot.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    for k in sorted(summary):
        v = summary[k]
        if "paired" in k:
            print(f"  {k}: diff={v['diff_pp']}pp CI={v['ci95']} "
                  f"f/b={v['fixed']}/{v['broken']}")
        else:
            print(f"  {k}: bem={v['binding_exact%']}% strict={v['strict_semantic%']}% "
                  f"val={v['validator%']}% exec={v['execution%']}%")
    print("PILOT-EVAL-DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    train = [json.loads(l) for l in open(f"{DATA}/train.jsonl",
                                          encoding="utf-8") if l.strip()]
    if args.smoke:
        train = train[:20]

    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    suffix = "_smoke" if args.smoke else ""
    m0_dir = f"runs/phase8d/m0_df_s{args.seed}{suffix}"
    m1_dir = f"runs/phase8d/m1_df_s{args.seed}{suffix}"

    print("=== Training M0 ===", flush=True)
    train_model("m0", train, tok, None, args.seed, m0_dir)

    print("=== Training M1 ===", flush=True)
    train_model("m1", train, tok, None, args.seed, m1_dir)

    if not args.smoke:
        print("=== Evaluation ===", flush=True)
        run_evaluation(f"{m0_dir}/final", f"{m1_dir}/final", args.seed)
    print("PILOT-DONE")


if __name__ == "__main__":
    main()
