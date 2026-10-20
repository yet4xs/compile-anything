"""Phase 8D-2B: Three-arm representation ablation (M0-T / M0-J / M1).

Single script: build targets → train all arms × seeds → evaluate.
Usage: nohup python scripts/train_three_arm.py > 3arm.log 2>&1 &
"""
import json, os, sys, time, re, gc, random, math, hashlib, argparse

ROOT = "/ccfa2026/compile-anything" if os.path.isdir(
    "/ccfa2026/compile-anything") else os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.getcwd() != ROOT:
    os.chdir(ROOT)

import torch
from torch.utils.data import Dataset
from collections import defaultdict
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, PeftModel

from src.ir.parser import parse_text
from src.ir.taskir import to_text, module_to_dict, module_from_dict
from src.validator.validator import validate

DATA = "data/runtime_dataflow_v2"
OUTR = "results/phase8d"
RUNS = "runs/phase8d"

# ── System prompts ──

SYS_M0T = (
    "You are a task compiler. Given a task and capabilities with schemas, "
    "emit TaskIR v0.2 text using EXTRACT_FIELD and bind() for runtime "
    "parameter binding. Output ONLY TaskIR.\n"
    '%id = EXEC_ACTION(action="tool", literals...)\n'
    '%fid = EXTRACT_FIELD(%producer, field="name") -> Type\n'
    '%id = EXEC_ACTION(action="tool")  bind(param=%fid)\n'
    'return %last')

SYS_M0J = (
    "You are a task compiler. Given a task and capabilities with schemas, "
    "emit a complete TaskIR v0.2 JSON AST. Output ONLY valid JSON.\n"
    'Format: {"taskir_version": "0.2", "program": {"name": "...", '
    '"inputs": [...], "nodes": [{"id": "%n", "op": "EXEC_ACTION", '
    '"inputs": [...], "params": {...}, "param_refs": {...}, '
    '"output_type": null, "after": [...], "hints": {...}}], '
    '"output": "%last"}}\n'
    "Use op=EXTRACT_FIELD for field extraction and param_refs for "
    "runtime parameter binding.")

SYS_M1 = (
    "You are a semantic task planner. Given a task and capabilities "
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
        eff = meta.get("effect_class", "read_only")
        lines.append(f"- {tid} | in: {ins_s} | out: {outs_s} | eff: {eff}")
    return "\n".join(lines)


def build_user(rec, protocol="distractor"):
    return (f"Task:\n{rec['instruction']}\n\n"
            f"Available capabilities:\n{caps_block(rec, protocol)}")


def build_targets(rec):
    """Build M0-T, M0-J, M1 targets from a single record."""
    # M0-T: TaskIR text (already lowered)
    m0t = rec.get("m0_taskir", "")
    # M0-J: TaskIR JSON AST (from M0-T via module_to_dict)
    try:
        mod = parse_text(m0t)
        m0j = json.dumps(module_to_dict(mod), ensure_ascii=False)
        # verify roundtrip
        mod2 = module_from_dict(json.loads(m0j))
        assert validate(mod2).valid, "M0-J roundtrip validation failed"
    except Exception:
        m0j = None  # skip this record for M0-J if roundtrip fails
    # M1: Skeleton JSON
    m1 = json.dumps(rec["target_skeleton"], ensure_ascii=False)
    return m0t, m0j, m1


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


def train_arm(arm, train_rows, tok, seed, out_dir):
    sys_map = {"m0t": SYS_M0T, "m0j": SYS_M0J, "m1": SYS_M1}
    sys_p = sys_map[arm]

    # Set random state
    torch.manual_seed(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

    t_tokens = sum(len(tok(r["target"], add_special_tokens=False)["input_ids"])
                   for r in train_rows)
    print(f"{arm}/s{seed}: {len(train_rows)} samples, {t_tokens} target tokens")

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

    eff_batch = 16
    per_dev = 1
    accum = eff_batch
    steps_per_epoch = math.ceil(len(train_rows) / eff_batch)
    max_steps = int(steps_per_epoch * 3)  # 3 epochs

    targs = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=accum, num_train_epochs=1.0,
        max_steps=max_steps, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_strategy="no", bf16=True,
        optim="paged_adamw_32bit", report_to=[], seed=seed,
        max_grad_norm=1.0)
    ds = SFTData(train_rows, tok, sys_p)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                       data_collator=lambda b: collate(b, tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    os.makedirs(f"{out_dir}/final", exist_ok=True)
    model.save_pretrained(f"{out_dir}/final")
    losses = [h.get("loss") for h in trainer.state.log_history if "loss" in h]
    log = {"arm": arm, "seed": seed, "n_samples": len(train_rows),
           "target_tokens": t_tokens, "max_steps": max_steps,
           "wall_min": round(dt / 60, 1),
           "first_loss": losses[0] if losses else None,
           "last_loss": losses[-1] if losses else None}
    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  saved {out_dir}/final | {dt/60:.1f}min | "
          f"loss {losses[0]:.3f}→{losses[-1]:.3f}")
    del model, base, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return log


# ── Evaluation ──

def gen_and_eval(model, tok, rec, arm, protocol):
    """Generate one completion and evaluate."""
    sys_p = {"M0-T": SYS_M0T, "M0-J": SYS_M0J, "M1": SYS_M1}[arm]
    user = build_user(rec, protocol)
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

    # Convert to TaskIR
    taskir = ""
    if arm == "M0-T":
        taskir = text
    elif arm == "M0-J":
        try:
            d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            mod = module_from_dict(d)
            taskir = to_text(mod)
        except Exception:
            taskir = ""
    elif arm == "M1":
        try:
            from src.compiler.plan_skeleton import skeleton_from_dict, \
                validate_skeleton, hard_errors
            from src.compiler.skeleton_lowering import lower_skeleton
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
                if not errs:
                    low = lower_skeleton(sk, None, schemas, strict=True)
                    taskir = low.get("taskir") or ""
        except Exception:
            taskir = ""

    # Evaluate
    from src.eval.canonical_dataflow_matcher import (
        canonical_from_skeleton, canonical_from_taskir,
        canonical_call_binding_match, canonical_dag_match)

    result = {"task_id": rec["task_id"], "arm": arm,
              "protocol": protocol, "topology": rec["topology"],
              "parse": False, "validator": False,
              "binding_exact": False, "binding_f1": 0.0,
              "tool_match": False, "execution": False,
              "latency_s": lat, "gen_tokens": ntok,
              "hit_generation_cap": ntok >= 698,
              # Full, untruncated model output (audit requirement 9B-2.2 T0.3);
              # the 2000-char preview kept separately for log readability.
              "raw": text,
              "raw_preview": text[:2000],
              "raw_log_truncated": len(text) > 2000}

    # Stage-separated compilation outcomes (audit requirement 9B-2.2 T0.4):
    # json_syntax / skeleton_valid / lowering_ok / taskir_valid are recorded
    # independently so offline rescoring never has to infer them from raw.
    result["json_syntax"] = bool(taskir) if arm in ("M0-J", "M1") else None
    result["skeleton_valid"] = None
    result["lowering_ok"] = None
    if arm == "M1":
        result["skeleton_valid"] = taskir != ""
    if arm in ("M0-J", "M1"):
        result["lowering_ok"] = bool(taskir)

    if not taskir:
        return result
    try:
        mod = parse_text(taskir)
        result["parse"] = True
    except Exception:
        return result
    v2 = {t: {"input_schema": rec["capability_registry"][t].get("input_schema", {}),
               "output_schema": rec["capability_registry"][t].get("output_schema", {})}
           for t in rec["capability_registry"]}
    result["validator"] = validate(mod, capability_registry=v2).valid
    gold = sorted(rec.get("gold_capability_ids", []))
    pred_tools = sorted({n.params.get("action", "") for n in mod.program.nodes
                          if n.op == "EXEC_ACTION" and n.params.get("action")})
    result["tool_match"] = pred_tools == gold

    ref = canonical_from_skeleton(rec["target_skeleton"]["nodes"])
    pred = canonical_from_taskir(taskir)
    if pred:
        a = canonical_call_binding_match(ref, pred)
        result["binding_exact"] = a["bindings_match"] and a["calls_match"]
        n_ref_b = max(1, a["n_ref_bindings"])
        n_pred_b = max(1, a["n_pred_bindings"])
        result["binding_f1"] = round(
            2 * a["n_correct_bindings"] / (n_ref_b + n_pred_b), 4)
    # Execution
    if result["validator"]:
        from src.runtime.mock_capability_registry import (
            MockCapabilityRegistry, MockCapability, DataflowRuntime)
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
        result["execution"] = tr.success
    result["strict_semantic"] = bool(
        result["validator"] and result["tool_match"]
        and result["binding_exact"] and result["execution"])
    return result


def main():
    train = [json.loads(l) for l in open(f"{DATA}/train.jsonl",
                                          encoding="utf-8") if l.strip()]
    test = [json.loads(l) for l in open(f"{DATA}/test.jsonl",
                                         encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build training data per arm
    arms_data = {}
    for arm, sys_p in (("m0t", SYS_M0T), ("m0j", SYS_M0J), ("m1", SYS_M1)):
        rows = []
        skipped = 0
        for rec in train:
            m0t, m0j, m1 = build_targets(rec)
            target = {"m0t": m0t, "m0j": m0j, "m1": m1}[arm]
            if target is None:
                skipped += 1
                continue
            rows.append({"user": build_user(rec, "distractor"),
                         "target": target, "task_id": rec["task_id"]})
        arms_data[arm] = rows
        print(f"{arm}: {len(rows)} samples ({skipped} skipped)")

    # Train all arms × seeds
    SEEDS = [44, 45, 46]
    train_logs = {}
    for seed in SEEDS:
        for arm in ("m0t", "m0j", "m1"):
            out_dir = f"{RUNS}/{arm}_s{seed}"
            if os.path.exists(f"{out_dir}/final"):
                print(f"  skip {arm}/s{seed} (exists)")
                continue
            print(f"\n=== Training {arm}/s{seed} ===", flush=True)
            train_logs[f"{arm}_s{seed}"] = train_arm(
                arm, arms_data[arm], tok, seed, out_dir)

    # Evaluate
    print("\n=== Evaluation ===", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    all_results = []
    for seed in SEEDS:
        for arm, arm_label in (("m0t", "M0-T"), ("m0j", "M0-J"), ("m1", "M1")):
            adapter = f"{RUNS}/{arm}_s{seed}/final"
            if not os.path.exists(adapter):
                continue
            model = PeftModel.from_pretrained(base, adapter)
            model.eval()
            for protocol in ("oracle", "distractor"):
                for rec in test:
                    ev = gen_and_eval(model, tok, rec, arm_label, protocol)
                    ev["seed"] = seed
                    all_results.append(ev)
            print(f"  {arm_label}/s{seed} eval done", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Aggregate
    by = defaultdict(list)
    for r in all_results:
        by[(r["arm"], r["protocol"], r["seed"])].append(r)
        by[(r["arm"], r["protocol"], "ALL")].append(r)

    summary = {}
    for (arm, proto, seed), rows in sorted(by.items()):
        n = len(rows)
        def pct(k):
            return round(100 * sum(1 for x in rows if x.get(k)) / n, 1)
        def avg(k):
            v = [x.get(k, 0) for x in rows]
            return round(sum(v) / len(v), 4) if v else 0
        summary[f"{proto}|{seed}|{arm}"] = {
            "n": n, "parse%": pct("parse"), "validator%": pct("validator"),
            "tool_match%": pct("tool_match"),
            "binding_exact%": pct("binding_exact"),
            "binding_f1": avg("binding_f1"),
            "execution%": pct("execution"),
            "strict_semantic%": pct("strict_semantic"),
            "tokens_mean": round(sum(x["gen_tokens"] for x in rows) / n, 1),
            "latency_p50": sorted(x["latency_s"] for x in rows)[n // 2]}

    # Paired comparisons (across seeds, per protocol)
    for proto in ("oracle", "distractor"):
        for pair in (("M1", "M0-J"), ("M0-J", "M0-T"), ("M1", "M0-T")):
            a_arm, b_arm = pair
            deltas = []
            fx = bk = 0
            for seed in SEEDS:
                a = {x["task_id"]: x for x in all_results
                     if x["arm"] == a_arm and x["protocol"] == proto
                     and x["seed"] == seed}
                b = {x["task_id"]: x for x in all_results
                     if x["arm"] == b_arm and x["protocol"] == proto
                     and x["seed"] == seed}
                for tid in a:
                    if tid not in b:
                        continue
                    av = 1 if a[tid]["binding_exact"] else 0
                    bv = 1 if b[tid]["binding_exact"] else 0
                    deltas.append(av - bv)
                    if av and not bv:
                        fx += 1
                    elif bv and not av:
                        bk += 1
            if deltas:
                rng = random.Random(13)
                boots = sorted(
                    sum(deltas[rng.randrange(len(deltas))]
                        for _ in deltas) / len(deltas)
                    for _ in range(5000))
                summary[f"{proto}|paired|{a_arm}-vs-{b_arm}"] = {
                    "n": len(deltas),
                    "diff_pp": round(100 * sum(deltas) / len(deltas), 1),
                    "ci95": [round(100 * boots[125], 1),
                              round(100 * boots[4875], 1)],
                    "fixed": fx, "broken": bk}

    # Save
    os.makedirs(OUTR, exist_ok=True)
    with open(f"{OUTR}/three_arm_ablation.json", "w") as f:
        json.dump({"summary": summary, "train_logs": train_logs},
                  f, indent=2, ensure_ascii=False)
    with open(f"{OUTR}/three_arm_raw_predictions.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    for k in sorted(summary):
        v = summary[k]
        if "paired" in k:
            print(f"  {k}: diff={v['diff_pp']}pp CI={v['ci95']} "
                  f"f/b={v['fixed']}/{v['broken']}")
        elif "ALL" in k:
            print(f"  {k}: binding={v['binding_exact%']}% "
                  f"strict={v['strict_semantic%']}% val={v['validator%']}%")
    print("THREE-ARM-DONE")


if __name__ == "__main__":
    main()
