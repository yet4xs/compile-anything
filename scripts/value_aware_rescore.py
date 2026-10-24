"""v7 值感知自检:对 v6 已存 C2 计划零 API 重打分 + ≤2 次修复调用。

模拟自检(由抽取 flags + 任务给定窗口驱动,冻结于 v7 修正案):
- 轮询应答按场景窗口模拟;
- COMMAND 在最近观测未就绪时写 → 拒绝(可修复,反馈"观测为 0");
- 未就绪下停止且段数不足 → 允许提交(诚实停止)。
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
from spi_pilot_run import (PKT, extract_json, plan_steps, content_of,
                            oracle, correct)

OUTD = REPO / "results/spi_pilot_v6"
LEDGER = REPO / "results/usage_ledger_v5/spi_v7_20261024"


def value_aware_check(scenario, steps, flags):
    """返回 (action, reason): submit / reject-fixable。"""
    if not flags.get("cmd_ready_gate"):
        return "submit", "no gate extracted — checker abstains"
    windows = scenario.get("ready_windows", [0])
    never = scenario.get("never_ready", False)
    target = scenario.get("segments", 3)
    cmd_count, polls_since = 0, 0
    last_ready = True
    for st in steps:
        if st["op"] == "read" and st["reg"] == "STATUS":
            if cmd_count == 0:
                last_ready = True
            elif never:
                last_ready = False
            else:
                w = windows[min(cmd_count - 1, len(windows) - 1)]
                last_ready = polls_since >= w
                polls_since += 1
        elif st["op"] == "write" and st["reg"] == "COMMAND":
            if last_ready is False:
                return "reject", ("COMMAND submitted while the last "
                                  "observation was READY=0 (S1/S2 边界:"
                                  "不得带着已知未就绪前进)")
            cmd_count += 1
            polls_since = 0
            last_ready = False   # 提交后须重新观测(除窗口 0 需确认)
            if cmd_count == 1 and windows[0] == 0 and not never:
                last_ready = True  # 窗口 0:提交后仍就绪
    if cmd_count < target and last_ready is False:
        return "submit", "honest-stop plan (fewer segments, not-ready obs)"
    return "submit", "pass"


def main():
    raws = json.loads((OUTD / "raw_outputs.json").read_text(encoding="utf-8"))
    c2 = {r["tag"].split("/")[1]: r["content"] for r in raws
          if r["tag"].startswith("C2/")}
    res = json.loads((OUTD / "results.json").read_text(encoding="utf-8"))
    c_flags = {r["task"]: r.get("flags", {}) for r in res
               if r["method"] == "C"}

    out, usage, rawlog = [], [], []
    for v in PKT["variants"]:
        plan = plan_steps(extract_json(c2.get(v["id"], "")))
        if plan is None:
            out.append({"task": v["id"], "v7": "UNPARSEABLE"})
            continue
        act, why = value_aware_check(v["scenario"], plan,
                                     c_flags.get(v["id"], {}))
        rec = {"task": v["id"], "v7_action": act, "v7_reason": why}
        if act == "submit":
            o = oracle(v, plan)
            rec["oracle"] = o.judgment
            rec["correct"] = correct(v, o.judgment)
        else:
            rec["correct"] = False
        out.append(rec)
        print(v["id"], json.dumps(rec, ensure_ascii=False)[:150])

    # 修复调用: 仅 U4(若被拒)
    t0 = time.time()
    for rec in out:
        if rec.get("v7_action") != "reject":
            continue
        v = next(x for x in PKT["variants"] if x["id"] == rec["task"])
        plan = plan_steps(extract_json(c2.get(v["id"], "")))
        msgs = [{"role": "system", "content":
                 "你的计划未通过自动检查。按反馈修订;若场景在预算内"
                 "无法建立就绪,正确做法是有限次轮询后停止(允许提交"
                 "少于全部 segment)。输出完整 JSON {'plan':[...]}。"},
                {"role": "user", "content":
                 "任务: SPI_HOST 提交 3 个发送 segment(polling-only)。"
                 f"场景: {v['given']}\n上一版计划:\n" +
                 json.dumps({"plan": plan}, ensure_ascii=False)[:4000] +
                 "\n检查器反馈: " + rec["v7_reason"]}]
        payload = {"model": "glm-5.3-flash", "messages": msgs,
                   "temperature": 0, "max_tokens": 8192,
                   "thinking": {"type": "enabled"},
                   "reasoning_effort": "low", "stream": False}
        r = complete(payload, timeout=180)
        rawlog.append({"tag": f"C3v7/{rec['task']}",
                       "content": content_of(r)[:20000]})
        usage.append({"tag": f"C3v7/{rec['task']}",
                      "http_status": r["http_status"],
                      "usage": (r.get("response") or {})
                      .get("usage", {})})
        p2 = plan_steps(extract_json(content_of(r)))
        if p2 is None:
            rec.update({"repair": "UNPARSEABLE"})
        else:
            act2, why2 = value_aware_check(v["scenario"], p2,
                                           c_flags.get(v["id"], {}))
            if act2 == "submit":
                o = oracle(v, p2)
                rec.update({"repair": "submitted", "oracle2": o.judgment,
                            "correct": correct(v, o.judgment)})
            else:
                rec.update({"repair": "still-rejected", "why2": why2})
        print(rec["task"], "repair:", rec.get("repair"), flush=True)

    cc = sum(1 for r in out if r.get("correct"))
    print(f"v7 C-arm correct: {cc}/{len(out)}")
    tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage:
        for k in tot:
            tot[k] += (u.get("usage") or {}).get(k, 0)
    LEDGER.mkdir(parents=True, exist_ok=True)
    (LEDGER / "usage.json").write_text(json.dumps(
        {"schema": "raw_api_usage_ledger_v5",
         "batch": {"planned": 2, "attempts": len(usage),
                   "http200": sum(1 for u in usage
                                  if u["http_status"] == 200),
                   "reported_tokens": tot}, "calls": usage},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (OUTD / "results_c_v7.json").write_text(json.dumps(
        out, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUTD / "raw_outputs_v7.json").write_text(json.dumps(
        rawlog, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
