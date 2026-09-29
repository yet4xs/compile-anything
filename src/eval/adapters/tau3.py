"""τ³-bench adapter — stateful agent benchmark → EvalSample.

τ³ tasks are NOT plain function calling: they carry a DB state, a policy,
a user scenario and expected outcomes (actions + NL assertions). The
adapter preserves stateful semantics instead of flattening to static call
chains:
- initial_state: task.initial_state if present, else the domain db.json
  snapshot (recorded with its sha256)
- capabilities: domain tools (retail ships tools.md; other domains define
  tools in repo code — recorded as a pointer, never fabricated)
- metadata.policy: policy.md path + sha256 (full text stays on disk)
- reference_actions: evaluation_criteria.actions (expected DB writes)
- reference_answer: nl_assertions / communicate_info
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import List

from ..schema import EvalSample

ROOT = pathlib.Path(__file__).resolve().parents[3]
TAU3_DIR = ROOT / "data" / "external_benchmarks" / "tau3_bench" / "domains"

DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load(domains_dir: pathlib.Path = None) -> List[EvalSample]:
    d = pathlib.Path(domains_dir) if domains_dir else TAU3_DIR
    samples: List[EvalSample] = []
    for dom in DOMAINS:
        dd = d / dom
        if not dd.is_dir():
            continue
        tasks = json.loads((dd / "tasks.json").read_text(encoding="utf-8"))
        splits = {}
        sp = dd / "split_tasks.json"
        if sp.exists():
            splits = json.loads(sp.read_text(encoding="utf-8"))
        split_of = {}
        for split_name, ids in (splits.items() if isinstance(splits, dict)
                                else []):
            for i in ids if isinstance(ids, list) else []:
                split_of[str(i)] = split_name
        db = {}
        for dbf in ("db.json", "db.toml"):
            p = dd / dbf
            if p.exists():
                db = {"file": dbf, "sha256": _sha(p)}
                break
        tools = None
        tools_src = "repo code (pointer: third_party/tau3-bench)"
        tp = dd / "tools.md"
        if tp.exists():
            tools_src = f"tools.md@{_sha(tp)}"
        policy = dd / "policy.md"
        for t in tasks:
            us = t.get("user_scenario") or {}
            instrs = us.get("instructions") or {}
            if isinstance(instrs, dict):
                instrs = instrs.get("task_instructions", "")
            if not isinstance(instrs, str):
                instrs = json.dumps(instrs, ensure_ascii=False)[:300]
            desc = t.get("description") or ""
            if not isinstance(desc, str):
                desc = json.dumps(desc, ensure_ascii=False)[:300]
            ec = t.get("evaluation_criteria") or {}
            samples.append(EvalSample(
                benchmark="tau3_bench",
                case_id=f"{dom}-{t.get('id', '')}",
                instruction=desc + (f"\n\n{instrs}" if instrs else ""),
                capabilities=tools,
                conversation=None,       # user simulator is dynamic
                initial_state=t.get("initial_state") or db,
                reference_actions=ec.get("actions") or [],
                reference_answer="; ".join(
                    (ec.get("nl_assertions") or [])
                    + (ec.get("communicate_info") or [])),
                metadata={
                    "domain": dom,
                    "policy": (f"policy.md@{_sha(policy)}"
                               if policy.exists() else None),
                    "tools_source": tools_src,
                    "reward_basis": ec.get("reward_basis"),
                    "split": split_of.get(str(t.get("id")), None),
                }))
    return samples
