"""SPI v6 pilot 执行器(A/B/C + 单轮修复,冻结协议)。
Run once: python scripts/spi_pilot_run.py
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
from src.contract.spi_submit_checker import SPISubmitChecker, SPIScenario

PKT = json.loads((REPO / "results/spi_pilot_v6/tasks.json")
                 .read_text(encoding="utf-8"))
OUTD = REPO / "results/spi_pilot_v6"
LEDGER = REPO / "results/usage_ledger_v5/spi_v6_20261024"

C_ENUM = {"CMD_READY_GATE": "cmd_ready_gate",
          "QUEUE_ONE_MORE": "queue_one_more",
          "WAIT_AFTER_FULL": "wait_after_full",
          "TXEMPTY_DONE": "txempty_done",
          "CONFLICT_RETAINED": "conflict_retained"}


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


def call(msgs, tag, usage, raws):
    payload = {"model": "glm-5.3-flash", "messages": msgs,
               "temperature": 0, "max_tokens": 8192,
               "thinking": {"type": "enabled"}, "reasoning_effort": "low",
               "stream": False}
    r = complete(payload, timeout=180)
    raws.append({"tag": tag, "content": content_of(r)[:20000]})
    usage.append({"tag": tag, "http_status": r["http_status"],
                  "seconds": round(r["seconds"], 2),
                  "usage": (r.get("response") or {}).get("usage", {}),
                  "error": r["error"] or None})
    return r


def plan_steps(j):
    if not isinstance(j, dict) or not isinstance(j.get("plan"), list):
        return None
    out = []
    for st in j["plan"]:
        if not isinstance(st, dict) or "op" not in st:
            return None
        out.append({"op": st.get("op"), "reg": st.get("reg", ""),
                    "field": st.get("field"), "value": st.get("value")})
    return out


def docs_block():
    parts = []
    for k, m in PKT["documents"].items():
        parts.append(f"===== 文件 {k} ({m['sha256'][:12]}) =====\n" +
                     Path(m["path"]).read_text(encoding="utf-8",
                                               errors="replace"))
    return "\n\n".join(parts)


def task_stmt(v):
    return f"""任务: SPI_HOST 处于复位后空闲状态。提交 3 个发送 segment(依次写
COMMAND),并在每步用 polling(读 STATUS)取得证据,最终以充分证据
宣告 3 个 segment 全部发送完成。不使用中断(EVENT_ENABLE 全程关闭)。
场景: {v['given']}
输出严格 JSON: {{"plan":[{{"op":"read","reg":"STATUS","field":"READY"}} 或
{{"op":"write","reg":"COMMAND","value":"0x..."}} ...]}}。只输出 JSON。"""


def oracle(v, steps):
    sc = SPIScenario(**{k: w for k, w in v["scenario"].items()
                        if k in ("segments", "ready_windows",
                                 "txempty_after", "never_ready")})
    return SPISubmitChecker(sc, max_steps=300).check(steps)


def own_check(flags, steps):
    """结构自检(冻结): cmd1 前有读;第3 cmd 前有新读;终点前有读。"""
    reads = 0
    cmd_count = 0
    reads_after_two = 0
    for st in steps:
        if st["op"] == "read" and st["reg"] == "STATUS":
            reads += 1
            if cmd_count >= 2:
                reads_after_two += 1
        elif st["op"] == "write" and st["reg"] == "COMMAND":
            cmd_count += 1
            if cmd_count == 1 and flags.get("cmd_ready_gate") and reads == 0:
                return False, "first COMMAND without any STATUS read"
            if cmd_count >= 3 and flags.get("wait_after_full") and \
                    reads_after_two == 0:
                return False, "3rd+ COMMAND without a fresh STATUS read"
    if flags.get("txempty_done") and steps and not any(
            st["op"] == "read" for st in steps[-3:]):
        return False, "plan ends without a completion observation"
    return True, "pass"


def correct(v, judgment):
    if v["id"] == "U4":
        return judgment in ("BUDGET_INFEASIBLE", "INSUFFICIENT_EVIDENCE")
    return judgment == "SUCCESS"


def main():
    doc = docs_block()
    results, usage, raws = [], [], []
    t0 = time.time()
    for v in PKT["variants"]:
        base_msgs = [{"role": "user", "content": doc + "\n\n" + task_stmt(v)}]
        # A
        rA = call([{"role": "system", "content":
                    "你是嵌入式软件专家,按文档输出操作计划。"}] + base_msgs,
                  f"A/{v['id']}", usage, raws)
        pA = plan_steps(extract_json(content_of(rA)))
        d = {"task": v["id"], "method": "A"}
        if pA is None:
            d.update({"judgment": "UNPARSEABLE"})
        else:
            o = oracle(v, pA)
            d.update({"judgment": o.judgment, "contract": o.contract})
        results.append(d)
        # B
        rB = call([{"role": "system", "content":
                    "你是嵌入式软件专家。计划外每步附出处 {'file','line',"
                    "'quote'}。输出 {'plan':[...],'sources':[...]}"}]
                  + base_msgs, f"B/{v['id']}", usage, raws)
        jB = extract_json(content_of(rB))
        pB = plan_steps(jB)
        d = {"task": v["id"], "method": "B"}
        if pB is None:
            d.update({"judgment": "UNPARSEABLE"})
        else:
            o = oracle(v, pB)
            d.update({"judgment": o.judgment, "contract": o.contract})
        results.append(d)
        # C: extraction + plan + own check (+ repair once)
        rC1 = call([{"role": "system", "content":
                     "从文档抽取机器可检查的契约条目。输出 JSON 数组,每条 "
                     "{id,kind,target,quote,file,line};id 必须取自枚举 "
                     "{CMD_READY_GATE, QUEUE_ONE_MORE, WAIT_AFTER_FULL, "
                     "TXEMPTY_DONE, CONFLICT_RETAINED} 之一。没有依据的"
                     "不要写。"}, {"role": "user", "content": doc}],
                    f"C1/{v['id']}", usage, raws)
        entries = extract_json(content_of(rC1))
        if not isinstance(entries, list):
            entries = []
        ids = {e.get("id") for e in entries if isinstance(e, dict)}
        flags = {f: (eid in ids) for eid, f in C_ENUM.items()}
        rC2 = call([{"role": "system", "content":
                     "你是嵌入式软件专家。严格遵守你拥有的契约条目,输出"
                     "操作计划 JSON {'plan':[...]}。"},
                    {"role": "user", "content": doc + "\n\n契约条目:\n" +
                     json.dumps(entries, ensure_ascii=False)[:4000] +
                     "\n\n" + task_stmt(v)}], f"C2/{v['id']}", usage, raws)
        pC = plan_steps(extract_json(content_of(rC2)))
        d = {"task": v["id"], "method": "C", "flags": flags,
             "entry_ids": sorted(x for x in ids if x)}
        plan_final = pC
        if pC is not None:
            ok, why = own_check(flags, pC)
            d["self_check"] = why
            if not ok:
                # 单轮修复(冻结): 反馈后重写
                rC3 = call([{"role": "system", "content":
                             "你的计划未通过自动检查。按反馈修订,输出完整 "
                             "JSON {'plan':[...]}。"},
                            {"role": "user", "content": doc + "\n\n" +
                             task_stmt(v) + "\n\n上一版计划:\n" +
                             json.dumps({"plan": pC}, ensure_ascii=False)
                             [:4000] + "\n\n检查器反馈: " + why}],
                            f"C3/{v['id']}", usage, raws)
                pC3 = plan_steps(extract_json(content_of(rC3)))
                if pC3 is not None:
                    ok2, why2 = own_check(flags, pC3)
                    if ok2:
                        plan_final = pC3
                        d["repaired"] = True
                    else:
                        plan_final = None
                        d["repaired"] = False
                        d["self_check2"] = why2
                else:
                    plan_final = None
                    d["repaired"] = False
        if plan_final is None and pC is None:
            d.update({"judgment": "UNPARSEABLE"})
        elif plan_final is None:
            d.update({"judgment": "NO_VALID_PLAN"})
        else:
            o = oracle(v, plan_final)
            d.update({"judgment": o.judgment, "contract": o.contract})
        results.append(d)
        print(v["id"], "done", flush=True)

    for m in ("A", "B", "C"):
        sub = [r for r in results if r["method"] == m]
        cc = sum(correct(next(v for v in PKT["variants"]
                              if v["id"] == r["task"]), r["judgment"])
                 for r in sub)
        print(m, "correct", f"{cc}/{len(sub)}")
    tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage:
        for k in tot:
            tot[k] += (u.get("usage") or {}).get(k, 0)
    LEDGER.mkdir(parents=True, exist_ok=True)
    (LEDGER / "usage.json").write_text(json.dumps(
        {"schema": "raw_api_usage_ledger_v5",
         "batch": {"planned": 16, "attempts": len(usage),
                   "http200": sum(1 for u in usage
                                  if u["http_status"] == 200),
                   "reported_tokens": tot}, "calls": usage},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (OUTD / "results.json").write_text(json.dumps(
        results, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUTD / "raw_outputs.json").write_text(json.dumps(
        raws, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(tot), "wall_min", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
