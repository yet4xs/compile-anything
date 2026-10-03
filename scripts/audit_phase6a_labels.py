"""Phase 6A Task 1: audit EXEC_ACTION / capability->skill supervision in corpus v3.1.

Joins every lowering entry (tool, skill, mapping_kind) to the raw tool schema
(xLAM / toolbench_static), then aggregates by skill x source x mapping_kind with
unique tool names/families and EXEC_ACTION exact/verified-fallback/unverified-
fallback split per the frozen phase6a_manifest.json.
"""
import json, os, re, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data/compiler_corpus_v3_1/train.jsonl")
RAW_XLAM = os.path.join(ROOT, "data/raw/xlam/xlam_function_calling_60k.json")
RAW_TBS = os.path.join(ROOT, "data/raw/toolbench_static")

ACTION_VERB_RE = re.compile(
    r"\b(create|update|delete|remove|toggle|enable|disable|cancel|book|reserve|"
    r"order|purchase|buy|checkout|place|submit|send|post|add|edit|modify|set|"
    r"assign|grant|revoke|reset|activate|deactivate|close|open|transfer|withdraw|"
    r"deposit|apply|register|sign[- ]?up|subscribe|unsubscribe|publish|upload|"
    r"install|run|execute|start|stop|restart|connect|disconnect)\w*\b", re.I)

# frozen contamination check: retrieval-prefixed tool names labeled EXEC_ACTION
RETRIEVAL_PREFIX_RE = re.compile(
    r"^(get|list|search|find|fetch|query|lookup|retrieve|show|view|check|is_|has_)", re.I)
# description must START with an explicit mutation verb to override the prefix
MUTATION_START_RE = re.compile(
    r"^\s*(create|add|update|delete|remove|cancel|book|reserve|place|make|toggle|"
    r"enable|disable|grant|revoke|reset|activate|deactivate|transfer|submit|"
    r"publish|upload|install|sign\s?up|subscribe|send|post|order)", re.I)


def family_key(tool_name: str) -> str:
    return "_".join(tool_name.split("_")[:2]) if tool_name else "<none>"


def load_xlam_schemas():
    """name -> {description, parameters}"""
    idx = {}
    d = json.load(open(RAW_XLAM, encoding="utf-8"))
    for rec in d:
        tools = rec.get("tools")
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except Exception:
                continue
        for t in tools or []:
            if isinstance(t, dict) and t.get("name"):
                idx.setdefault(t["name"], {
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}) or t.get("parameter", {}),
                })
    return idx


def load_tbs_schemas():
    idx = {}
    if not os.path.isdir(RAW_TBS):
        return idx
    for fn in os.listdir(RAW_TBS):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(RAW_TBS, fn), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for t in rec.get("tools") or []:
                    if isinstance(t, dict) and t.get("name"):
                        idx.setdefault(t["name"], {
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                        })
                # toolbench_static records may carry api metadata in other fields
                for key in ("api_list", "apis", "functions"):
                    for t in rec.get(key) or []:
                        if isinstance(t, dict) and t.get("name"):
                            idx.setdefault(t["name"], {
                                "description": t.get("description", ""),
                                "parameters": t.get("parameters", {}),
                            })
    return idx


def main():
    xlam = load_xlam_schemas()
    tbs = load_tbs_schemas()
    print(f"schema index: xlam={len(xlam)} toolbench_static={len(tbs)}")

    def schema_for(tool, source):
        if tool in xlam:
            return xlam[tool], "xlam_raw"
        if tool in tbs:
            return tbs[tool], "tbs_raw"
        return None, None

    agg = defaultdict(Counter)          # (skill, source, mapping_kind) -> counts
    tool_names = defaultdict(set)       # (skill, source) -> unique tool names
    tool_fams = defaultdict(set)
    exec_split = Counter()              # exact_action_rule / verified_fallback / unverified_fallback
    exec_split_examples = defaultdict(list)
    join_hit = join_miss = 0
    total_calls = 0
    skill_support = Counter()
    per_call = []                        # rows for grounding dataset feasibility check

    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src = r.get("source", "")
            for low in r.get("lowering") or []:
                tool = low.get("tool", "")
                skill = low.get("skill", "")
                mk = low.get("mapping_kind", "")
                total_calls += 1
                skill_support[skill] += 1
                sch, sch_src = schema_for(tool, src)
                if sch:
                    join_hit += 1
                else:
                    join_miss += 1
                agg[(skill, src, mk)]["count"] += 1
                agg[(skill, src, mk)]["schema_available"] += 1 if sch else 0
                tool_names[(skill, src)].add(tool)
                tool_fams[(skill, src)].add(family_key(tool))
                per_call.append({
                    "tool": tool, "skill": skill, "mapping_kind": mk,
                    "matched_rule": low.get("matched_rule"),
                    "source": src, "schema": bool(sch),
                })
                if skill == "EXEC_ACTION":
                    if mk == "exact":
                        # contamination check: retrieval-prefixed name labeled action
                        if RETRIEVAL_PREFIX_RE.match(tool):
                            desc = (sch or {}).get("description", "") or ""
                            if MUTATION_START_RE.match(desc):
                                cat = "exact_action_rule"
                            else:
                                cat = "contradictory_exact"
                        else:
                            cat = "exact_action_rule"
                    elif mk == "fallback":
                        desc = (sch or {}).get("description", "") or ""
                        cat = ("verified_fallback"
                               if (sch and ACTION_VERB_RE.search(desc))
                               else "unverified_fallback")
                    else:  # heuristic EXEC_ACTION
                        cat = "exact_action_rule"  # verb rule matched explicitly
                    exec_split[cat] += 1
                    if len(exec_split_examples[cat]) < 3:
                        exec_split_examples[cat].append(
                            {"tool": tool, "desc": (sch or {}).get("description", "")[:120]})

    print(f"\ntotal tool calls: {total_calls}; schema join hit {join_hit} "
          f"({100*join_hit/max(1,total_calls):.1f}%), miss {join_miss}")

    print("\n=== skill support (top 15) ===")
    for s, c in skill_support.most_common(15):
        print(f"  {s:<20s} {c:6d}")

    print("\n=== EXEC_ACTION supervision split (frozen criteria) ===")
    for cat in ("exact_action_rule", "contradictory_exact", "verified_fallback", "unverified_fallback"):
        print(f"  {cat:24s} {exec_split[cat]:6d}")
        for ex in exec_split_examples[cat]:
            print(f"      e.g. {ex['tool']}: {ex['desc']}")

    rows = []
    for (skill, src, mk), c in sorted(agg.items()):
        rows.append({
            "skill": skill, "source": src, "mapping_kind": mk,
            "count": c["count"], "schema_available": c["schema_available"],
            "unique_tool_names": len(tool_names[(skill, src)]),
            "unique_tool_families": len(tool_fams[(skill, src)]),
        })

    out = {
        "total_tool_calls": total_calls,
        "schema_join": {"hit": join_hit, "miss": join_miss,
                        "hit_pct": round(100 * join_hit / max(1, total_calls), 2)},
        "skill_support": dict(skill_support.most_common()),
        "exec_action_split": dict(exec_split),
        "exec_action_examples": {k: v for k, v in exec_split_examples.items()},
        "cells": rows,
        "note": "corpus 'capabilities' field is semantic skill STRINGS; real tool "
                "schemas recovered by joining lowering.tool -> raw dataset definitions "
                "(xlam_function_calling_60k.json / toolbench_static)",
    }
    os.makedirs(os.path.join(ROOT, "results/phase6"), exist_ok=True)
    with open(os.path.join(ROOT, "results/phase6/grounding_label_audit.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved results/phase6/grounding_label_audit.json")

    # feasibility preview: boundary skills with schema
    boundary = {"SEARCH", "FETCH", "QUERY_DB", "EXEC_ACTION", "SEND", "SAVE"}
    b = Counter()
    for c in per_call:
        if c["skill"] in boundary and c["schema"] and c["mapping_kind"] in ("exact", "heuristic"):
            b[(c["skill"], c["mapping_kind"])] += 1
    print("\n=== boundary-skill G1/G2 samples WITH schema (dataset feasibility) ===")
    for (s, mk), c in sorted(b.items()):
        print(f"  {s:<14s} {mk:<10s} {c:6d}")


if __name__ == "__main__":
    main()
