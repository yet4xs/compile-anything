"""v3 C-arm-only rerun (frozen amendment): array-capable parser + raw
persistence. A/B verdicts from run 2 stand. Executes C1+C2 for the 6
variants once. Usage ledger appended separately.
"""
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/serverless_eval_v1"))
sys.path.insert(0, str(REPO / "scripts"))
from glm_client import complete
from src.contract.aes_init_checker import AESInitChecker, Scenario, KEY_REGS
from aes_pilot_run import (PKT, OUTD, docs_block, task_stmt, plan_steps,
                            oracle, own_check, C_ENUM)

SYS_C1 = ("从文档抽取机器可检查的契约条目。输出 JSON 数组,"
          "每条 {id,kind,target,quote,file,line};kind 取 "
          "precondition/side_effect/ordering/protocol/"
          "completion;id 必须取自枚举 {INIT_IDLE, "
          "CTRL_BEFORE_KEY, RECHECK_AFTER_CTRL, "
          "CTRL_TWO_WRITE, KEY_ALL_REGS, "
          "READ_RESETS_PHASE} 之一。没有依据的不要写。")


def extract_json(text):
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None


def content_of(r):
    return ((r.get("response") or {}).get("choices", [{}])[0]
            .get("message", {}).get("content", ""))


def call(msgs, tag, raw_log, usage):
    payload = {"model": "glm-5.3-flash", "messages": msgs,
               "temperature": 0, "max_tokens": 8192,
               "thinking": {"type": "enabled"}, "reasoning_effort": "low",
               "stream": False}
    r = complete(payload, timeout=180)
    raw_log.append({"tag": tag, "content": content_of(r)[:20000]})
    usage.append({"tag": tag, "http_status": r["http_status"],
                  "seconds": round(r["seconds"], 2),
                  "usage": (r.get("response") or {}).get("usage", {}),
                  "error": r["error"] or None})
    return r


def main():
    doc = docs_block()
    results, usage, raws = [], [], []
    t0 = time.time()
    for v in PKT["variants"]:
        rC1 = call([{"role": "system", "content": SYS_C1},
                    {"role": "user", "content": doc}],
                   f"C1v3/{v['id']}", raws, usage)
        entries = extract_json(content_of(rC1))
        if not isinstance(entries, list):
            entries = []
        ids = {e.get("id") for e in entries if isinstance(e, dict)}
        flags = {f: (eid in ids) for eid, f in C_ENUM.items()}
        rC2 = call([{"role": "system", "content":
                     "你是硬件软件接口专家。严格遵守你拥有的契约条目,"
                     "输出操作计划 JSON {'plan':[...]}。"},
                    {"role": "user", "content": doc + "\n\n" +
                     "抽取到的契约条目:\n" +
                     json.dumps(entries, ensure_ascii=False)[:4000] +
                     "\n\n" + task_stmt(v)}], f"C2v3/{v['id']}", raws, usage)
        planC = plan_steps(extract_json(content_of(rC2)))
        vd = {"task": v["id"], "method": "C_v3", "flags": flags,
              "entries_n": len(entries),
              "entry_ids": sorted(x for x in ids if x)}
        if planC is None:
            vd.update({"judgment": "UNPARSEABLE", "submitted": False})
        else:
            ok, why = own_check(flags, planC)
            vd["self_check"] = why
            if not ok:
                vd.update({"judgment": "NO_VALID_PLAN", "submitted": False})
            else:
                o = oracle(v, planC)
                vd.update({"judgment": o.judgment, "contract": o.contract,
                           "submitted": True})
        results.append(vd)
        print(v["id"], json.dumps(flags), flush=True)

    tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage:
        for k in tot:
            tot[k] += (u.get("usage") or {}).get(k, 0)
    ld = REPO / "results/usage_ledger_v5/aes_pilot_v3_c_20261024"
    ld.mkdir(parents=True, exist_ok=True)
    (ld / "usage.json").write_text(json.dumps(
        {"schema": "raw_api_usage_ledger_v5",
         "batch": {"planned": 12, "attempts": len(usage),
                   "http200": sum(1 for u in usage
                                  if u["http_status"] == 200),
                   "reported_tokens": tot}, "calls": usage},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (OUTD / "results_c_v3.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUTD / "raw_outputs_c_v3.json").write_text(
        json.dumps(raws, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(tot), "wall_min", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
