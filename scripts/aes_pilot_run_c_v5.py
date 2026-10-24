"""v5 修复轮(冻结于 v5 修正案): 对 v4 判为可修复 decline 的任务各
执行一次 C3 修复调用,输出计划经同一 v4 深度+结构自检后决定提交。
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/serverless_eval_v1"))
sys.path.insert(0, str(REPO / "scripts"))
from glm_client import complete
from aes_pilot_run import PKT, OUTD, docs_block, task_stmt, extract_json, \
    plan_steps, oracle
from aes_pilot_run_c_v4 import depth_check


def content_of(r):
    return ((r.get("response") or {}).get("choices", [{}])[0]
            .get("message", {}).get("content", ""))


def main():
    raws = json.loads((OUTD / "raw_outputs_c_v3.json").read_text(
        encoding="utf-8"))
    c2 = {r["tag"].split("/")[1]: r["content"] for r in raws
          if r["tag"].startswith("C2v3/")}
    v4 = {r["task"]: r for r in json.loads(
        (OUTD / "results_c_v4_rescore.json").read_text(encoding="utf-8"))}

    doc = docs_block()
    results, usage, rawlog = [], [], []
    t0 = time.time()
    for v in PKT["variants"]:
        vid = v["id"]
        if not v4.get(vid, {}).get("fixable"):
            results.append({"task": vid, "action": "no-repair",
                            "v4": v4.get(vid, {}).get("v4_depth")})
            continue
        plan = plan_steps(extract_json(c2.get(vid, "")))
        ok, _, why = depth_check(v["scenario"], plan)
        # 结构自检沿用 v3 own_check(flags 全 6)
        from aes_pilot_run import own_check
        s_ok, s_why = own_check({f: True for f in
                                 ("initial_idle_gate", "ctrl_before_key",
                                  "recheck_after_ctrl", "ctrl_two_write",
                                  "key_all_registers",
                                  "read_resets_phase")}, plan)
        msgs = [{"role": "system", "content":
                 "你是硬件软件接口专家。你的上一版操作计划未通过自动"
                 "检查。按检查器反馈修订计划,输出完整 JSON "
                 "{'plan':[...]}。"},
                {"role": "user", "content": doc + "\n\n" + task_stmt(v) +
                 "\n\n上一版计划:\n" +
                 json.dumps({"plan": plan}, ensure_ascii=False)[:4000] +
                 "\n\n检查器反馈: " + (why if not ok else s_why)}]
        payload = {"model": "glm-5.3-flash", "messages": msgs,
                   "temperature": 0, "max_tokens": 8192,
                   "thinking": {"type": "enabled"},
                   "reasoning_effort": "low", "stream": False}
        r = complete(payload, timeout=180)
        rawlog.append({"tag": f"C3v5/{vid}",
                       "content": content_of(r)[:20000]})
        usage.append({"tag": f"C3v5/{vid}",
                      "http_status": r["http_status"],
                      "seconds": round(r["seconds"], 2),
                      "usage": (r.get("response") or {})
                      .get("usage", {}),
                      "error": r["error"] or None})
        plan2 = plan_steps(extract_json(content_of(r)))
        rec = {"task": vid, "action": "repair"}
        if plan2 is None:
            rec.update({"final": "NO_VALID_PLAN(unparseable)"})
        else:
            ok2, fix2, why2 = depth_check(v["scenario"], plan2)
            s_ok2, s_why2 = own_check(
                {f: True for f in ("initial_idle_gate", "ctrl_before_key",
                                   "recheck_after_ctrl", "ctrl_two_write",
                                   "key_all_registers",
                                   "read_resets_phase")}, plan2)
            if ok2 and s_ok2:
                o = oracle(v, plan2)
                rec.update({"final": "submitted",
                            "oracle": o.judgment,
                            "contract": o.contract})
            else:
                rec.update({"final": "NO_VALID_PLAN",
                            "check": why2 if not ok2 else s_why2,
                            "still_fixable": fix2})
        results.append(rec)
        print(vid, json.dumps(rec, ensure_ascii=False)[:140], flush=True)

    tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage:
        for k in tot:
            tot[k] += (u.get("usage") or {}).get(k, 0)
    ld = REPO / "results/usage_ledger_v5/aes_pilot_v5_repair_20261024"
    ld.mkdir(parents=True, exist_ok=True)
    (ld / "usage.json").write_text(json.dumps(
        {"schema": "raw_api_usage_ledger_v5",
         "batch": {"planned": 4, "attempts": len(usage),
                   "http200": sum(1 for u in usage
                                  if u["http_status"] == 200),
                   "reported_tokens": tot}, "calls": usage},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (OUTD / "results_c_v5_repair.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False),
        encoding="utf-8")
    (OUTD / "raw_outputs_c_v5.json").write_text(
        json.dumps(rawlog, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(tot), "wall_min", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
