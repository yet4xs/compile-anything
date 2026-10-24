"""v8 执行器: 公平修复轮(A+/B+) + 第二模型 A 臂。冻结于 6b885717。
Run once: python scripts/v8_fairness_run.py
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
from aes_pilot_run import PKT as AES_PKT, task_stmt as aes_stmt, \
    plan_steps, oracle as aes_oracle, own_check as aes_own
from aes_pilot_run_c_v4 import depth_check
from spi_pilot_run import PKT as SPI_PKT, task_stmt as spi_stmt, \
    oracle as spi_oracle, own_check as spi_own

OUTD = REPO / "results/v8_fairness_20261024"
LEDGER = REPO / "results/usage_ledger_v5/v8_20261024"
OUTD.mkdir(parents=True, exist_ok=True)

AES_DOC = None
SPI_DOC = None


def docs_aes():
    global AES_DOC
    if AES_DOC is None:
        parts = [f"===== 文件 {k} ({m['sha256'][:12]}) =====\n" +
                 Path(m["path"]).read_text(encoding="utf-8", errors="replace")
                 for k, m in AES_PKT["documents"].items()]
        AES_DOC = "\n\n".join(parts)
    return AES_DOC


def docs_spi():
    global SPI_DOC
    if SPI_DOC is None:
        parts = [f"===== 文件 {k} ({m['sha256'][:12]}) =====\n" +
                 Path(m["path"]).read_text(encoding="utf-8", errors="replace")
                 for k, m in SPI_PKT["documents"].items()]
        SPI_DOC = "\n\n".join(parts)
    return SPI_DOC


def extract_json(text):
    import re
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


usage, raws = [], []


def call(msgs, tag, model="glm-5.3-flash"):
    payload = {"model": model, "messages": msgs, "temperature": 0,
               "max_tokens": 8192, "thinking": {"type": "enabled"},
               "reasoning_effort": "low", "stream": False}
    r = complete(payload, timeout=180)
    raws.append({"tag": tag, "model": model,
                 "content": content_of(r)[:20000]})
    usage.append({"tag": tag, "model": model,
                  "http_status": r["http_status"],
                  "seconds": round(r["seconds"], 2),
                  "usage": (r.get("response") or {}).get("usage", {}),
                  "error": r["error"] or None})
    return r


def generic_feedback(ip, scenario, plan):
    """通用结构检查(与 C 管线 v6 阶段同代码、全 flags;不含深度/取值)。"""
    if ip == "aes":
        flags = {f: True for f in ("initial_idle_gate", "ctrl_before_key",
                                   "recheck_after_ctrl", "ctrl_two_write",
                                   "key_all_registers",
                                   "read_resets_phase")}
        ok, why = aes_own(flags, plan)
        return ok, why
    flags = {f: True for f in ("cmd_ready_gate", "queue_one_more",
                               "wait_after_full", "txempty_done")}
    ok, why = spi_own(flags, plan)
    return ok, why


def run_family(ip, tasks, doc, stmt, oracle_fn, sysgen):
    results = []
    for v in tasks:
        vid = v["id"]
        base = [{"role": "user", "content": doc + "\n\n" + stmt(v)}]
        # ── A 生成 ──
        rA = call([{"role": "system", "content": sysgen}] + base, f"A/{vid}")
        pA = plan_steps(extract_json(content_of(rA)))
        rec = {"task": vid, "arm": "A"}
        planA_final = pA
        if pA is None:
            rec["init"] = "UNPARSEABLE"
        else:
            o = oracle_fn(v, pA)
            rec["init"] = o.judgment
            rec["init_contract"] = o.contract
            if not o.judgment == "SUCCESS" and \
                    o.judgment not in ("INSUFFICIENT_EVIDENCE",
                                       "BUDGET_INFEASIBLE"):
                ok, why = generic_feedback(ip, v["scenario"], pA)
                rec["generic_check"] = why
                if not ok:
                    rA2 = call([{"role": "system", "content":
                                 sysgen + " 你的计划未通过自动检查,按反馈"
                                 "修订,输出完整 JSON {'plan':[...]}。"}]
                                + base
                                + [{"role": "user", "content":
                                    "上一版计划:\n" + json.dumps(
                                        {"plan": pA}, ensure_ascii=False
                                    )[:4000] + "\n检查器反馈: " + why}],
                                f"A2/{vid}")
                    pA2 = plan_steps(extract_json(content_of(rA2)))
                    if pA2 is not None:
                        ok2, _ = generic_feedback(ip, v["scenario"], pA2)
                        if ok2:
                            planA_final = pA2
                            rec["repaired"] = True
                        else:
                            planA_final = None
                            rec["repaired"] = False
                    else:
                        planA_final = None
                        rec["repaired"] = False
        if planA_final is None and rec.get("init") != "UNPARSEABLE":
            rec["final"] = "NO_VALID_PLAN"
        elif planA_final is None:
            rec["final"] = "UNPARSEABLE"
        else:
            o = oracle_fn(v, planA_final)
            rec["final"] = o.judgment
            rec["final_contract"] = o.contract
        results.append(rec)

        # ── B 生成(同修复权) ──
        rB = call([{"role": "system", "content":
                    sysgen + " 计划外每步附出处。输出 {'plan':[...],"
                    "'sources':[...]}"}] + base, f"B/{vid}")
        pB = plan_steps(extract_json(content_of(rB)))
        rec = {"task": vid, "arm": "B"}
        planB_final = pB
        if pB is None:
            rec["init"] = "UNPARSEABLE"
        else:
            o = oracle_fn(v, pB)
            rec["init"] = o.judgment
            if not o.judgment == "SUCCESS" and \
                    o.judgment not in ("INSUFFICIENT_EVIDENCE",
                                       "BUDGET_INFEASIBLE"):
                ok, why = generic_feedback(ip, v["scenario"], pB)
                if not ok:
                    rB2 = call([{"role": "system", "content":
                                 sysgen + " 你的计划未通过自动检查,按反馈"
                                 "修订,输出完整 JSON {'plan':[...]}。"}]
                                + base
                                + [{"role": "user", "content":
                                    "上一版计划:\n" + json.dumps(
                                        {"plan": pB}, ensure_ascii=False
                                    )[:4000] + "\n检查器反馈: " + why}],
                                f"B2/{vid}")
                    pB2 = plan_steps(extract_json(content_of(rB2)))
                    if pB2 is not None and \
                            generic_feedback(ip, v["scenario"], pB2)[0]:
                        planB_final = pB2
                        rec["repaired"] = True
                    else:
                        planB_final = None
                        rec["repaired"] = False
        if planB_final is None:
            rec["final"] = "NO_VALID_PLAN" if rec.get("init") != \
                "UNPARSEABLE" else "UNPARSEABLE"
        else:
            o = oracle_fn(v, planB_final)
            rec["final"] = o.judgment
        results.append(rec)
        print(vid, "A/B done", flush=True)
    return results


def run_second_model(tasks, doc, stmt, oracle_fn, sysgen):
    results = []
    for v in tasks:
        r = call([{"role": "system", "content": sysgen},
                  {"role": "user", "content": doc + "\n\n" + stmt(v)}],
                 f"A53/{v['id']}", model="glm-5.3")
        p = plan_steps(extract_json(content_of(r)))
        rec = {"task": v["id"], "arm": "A-glm5.3"}
        if p is None:
            rec["final"] = "UNPARSEABLE"
        else:
            o = oracle_fn(v, p)
            rec["final"] = o.judgment
            rec["contract"] = o.contract
        results.append(rec)
        print(v["id"], "glm5.3 done", flush=True)
    return results


def main():
    t0 = time.time()
    allr = []
    sys_aes = "你是硬件软件接口专家,按文档输出操作计划。"
    sys_spi = "你是嵌入式软件专家,按文档输出操作计划。"
    allr += run_family("aes", AES_PKT["variants"], docs_aes(),
                       aes_stmt, aes_oracle, sys_aes)
    allr += run_family("spi", SPI_PKT["variants"], docs_spi(),
                       spi_stmt, spi_oracle, sys_spi)
    allr += run_second_model(AES_PKT["variants"], docs_aes(),
                             aes_stmt, aes_oracle, sys_aes)
    allr += run_second_model(SPI_PKT["variants"], docs_spi(),
                             spi_stmt, spi_oracle, sys_spi)

    def correct(tid, j):
        never = (tid == "T6" or tid == "U4")
        okset = ("BUDGET_INFEASIBLE", "INSUFFICIENT_EVIDENCE") if never \
            else ("SUCCESS",)
        return j in okset

    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in allr:
        arm = r["arm"]
        j = r.get("final") or r.get("init")
        agg[arm][1] += 1
        agg[arm][0] += correct(r["task"], j)
    for arm, (c, n) in sorted(agg.items()):
        print(f"{arm}: {c}/{n}")
    (OUTD / "results.json").write_text(json.dumps(
        allr, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUTD / "raw_outputs.json").write_text(json.dumps(
        raws, indent=1, ensure_ascii=False), encoding="utf-8")
    tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage:
        for k in tot:
            tot[k] += (u.get("usage") or {}).get(k, 0)
    LEDGER.mkdir(parents=True, exist_ok=True)
    (LEDGER / "usage.json").write_text(json.dumps(
        {"schema": "raw_api_usage_ledger_v5",
         "batch": {"attempts": len(usage),
                   "http200": sum(1 for u in usage
                                  if u["http_status"] == 200),
                   "reported_tokens": tot}, "calls": usage},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(tot), "wall_min", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
