"""Spider / BIRD style text-to-SQL samples -> TaskIR.

Input record schema:
    {"task_id", "db_id", "question", "query": "SELECT ..."}

Lowering policy (deliberately conservative):
  - the SQL always stays executable inside QUERY_DB (params.query,
    params.table) — the DB executor is the "hardware" for relational ops;
  - a simple SELECT column-list WITHOUT group-by/having/join/aggregates is
    additionally DECOMPOSED with an EXTRACT node (field projection);
  - GROUP BY / HAVING / JOIN / window functions / aggregates are NOT
    force-mapped onto inadequate ops — they are recorded as SQL features
    in meta.provenance.sql_features and feed docs/missing-skills.md
    (GROUP, TABLE_SCAN etc. are documented ISA gaps, not silent hacks).
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from ...ir.taskir import Module, Node, Program, Retry
from .base import BenchmarkLifter, register

_AGG_RE = re.compile(r"\b(avg|sum|count|min|max)\s*\(", re.I)


def _features(sql: str) -> Dict[str, bool]:
    s = sql.lower()
    return {
        "where": bool(re.search(r"\bwhere\b", s)),
        "group_by": bool(re.search(r"\bgroup\s+by\b", s)),
        "having": bool(re.search(r"\bhaving\b", s)),
        "order_by": bool(re.search(r"\border\s+by\b", s)),
        "limit": bool(re.search(r"\blimit\b", s)),
        "join": bool(re.search(r"\bjoin\b", s)),
        "aggregate": bool(_AGG_RE.search(s)),
        "distinct": "distinct" in s,
    }


@register
class SpiderLifter(BenchmarkLifter):
    name = "spider"
    dataset = "spider/bird"

    def can_handle(self, sample: Dict) -> bool:
        return bool(sample.get("query") and sample.get("question")) \
            and bool(re.match(r"(?is)^\s*select\b", sample.get("query", "")))

    def lift(self, sample: Dict) -> Optional[Module]:
        sql = (sample.get("query") or "").strip().rstrip(";")
        question = sample.get("question") or ""
        m = re.match(r"(?is)^select\s+(.+?)\s+from\s+([a-z_][\w]*)\b(.*)$",
                     sql.replace("\n", " "))
        if not m:
            return None
        select_list, table, rest = m.group(1).strip(), m.group(2), m.group(3)
        feats = _features(sql)
        star = select_list.strip() == "*"
        plain_cols = None
        if not star and not feats["aggregate"] and not feats["group_by"] \
                and not feats["join"] and not feats["having"] \
                and re.fullmatch(r"[a-z_][\w]*(\s*,\s*[a-z_][\w]*)*",
                                 select_list, re.I):
            plain_cols = [c.strip() for c in select_list.split(",")]

        nodes = [Node(id="%c0", op="QUERY_DB", inputs=["@task"],
                      params={"table": table, "query": sql})]
        if plain_cols:
            nodes.append(Node(id="%c1", op="EXTRACT", inputs=["%c0"],
                              params={"fields": plain_cols}))
        prev = nodes[-1].id
        gen = Node(id="%ans", op="GENERATE", inputs=[prev, "@task"],
                   params={"role": "final_answer"})
        ver = Node(id="%verify", op="VERIFY", inputs=["%ans"],
                   params={"check": "answer_matches_query_result"})
        gen.retry = Retry(max_attempts=2, on="%verify")
        nodes.extend([gen, ver])

        name = re.sub(r"[^0-9A-Za-z_]+", "_", question[:40]).strip("_").lower() \
            or f"sql_{sample.get('task_id', 'task')}"
        prog = Program(name=name, description=question,
                       inputs=[{"name": "@task", "type": "Str"}],
                       nodes=nodes, output="%ans")
        return Module(program=prog, meta={
            "name": name,
            "provenance": {
                "source": self.name,
                "task_id": str(sample.get("task_id", "")),
                "db_id": sample.get("db_id", ""),
                "sql_features": feats,
                "decomposed": plain_cols is not None,
            }})

    def lift_with_reason(self, sample: Dict) -> Tuple[Optional[Module], Optional[str]]:
        if not sample.get("query"):
            return None, "missing query"
        if not re.match(r"(?is)^\s*select\b", sample.get("query", "")):
            return None, "non-SELECT sql"
        if self.lift(sample) is None:
            return None, "unparseable FROM clause"
        return self.lift(sample), None
