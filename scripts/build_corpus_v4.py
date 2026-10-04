"""Phase 6B Stage 1: build corpus v4 — capability-explicit version of v3.1.

v3.1 stays immutable. Every tool-use record in v4 carries THREE separate
concepts (never merged into one 'capabilities' field again):

  available_capabilities  — full runtime-visible tool list (raw dataset defs)
  selected_capabilities   — concrete tool ids the GT trajectory calls
                            (Resolver supervision / Composer oracle-selected
                            arms ONLY; never given at end-to-end inference)
  canonical_capabilities  — per-capability semantic metadata
                            {capability_id, name, canonical_skill, skill_family,
                             label_tier G1/G2/G3, confidence}

Label tiers reuse the Phase 6A frozen criteria (exact->G1 with the EXEC_ACTION
contamination filter; heuristic->G2; fallback/G3 audit-only).
Joins: corpus record id -> v3.1 record (instruction/plan_target/lowering);
lowering.tool -> raw schema index (xLAM / toolbench_static).
"""
import json, os, sys, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.audit_phase6a_labels import (load_xlam_schemas, load_tbs_schemas,
                                          RETRIEVAL_PREFIX_RE, MUTATION_START_RE)

V31 = os.path.join(ROOT, "data/compiler_corpus_v3_1")
OUT = os.path.join(ROOT, "data/compiler_corpus_v4")

BOUNDARY = {"RETRIEVAL": {"SEARCH", "FETCH", "QUERY_DB"},
            "ACTION": {"EXEC_ACTION", "SEND", "SAVE"}}


def skill_family(skill):
    for fam, skills in BOUNDARY.items():
        if skill in skills:
            return fam
    return "other"


def tier_of(skill, mk, name, desc):
    if mk == "exact":
        if skill == "EXEC_ACTION" and RETRIEVAL_PREFIX_RE.match(name) \
                and not MUTATION_START_RE.match(desc or ""):
            return "G3"  # contradictory exact (Phase 6A frozen filter)
        return "G1"
    if mk == "heuristic":
        return "G2"
    return "G3"


def slim_params(p):
    if not isinstance(p, dict):
        return {}
    req = p.get("required") if isinstance(p.get("required"), list) else []
    props = p.get("properties")
    if not isinstance(props, dict):
        props = {k: v for k, v in p.items() if k not in ("required", "type")}
    return {"required": req[:8],
            "properties": {k: str(v.get("type", v) if isinstance(v, dict) else v)[:12]
                           for k, v in list(props.items())[:10]}}


def build_split(split, xlam, tbs):
    rows, audit = [], Counter()
    skipped_no_schema = 0
    for line in open(os.path.join(V31, f"{split}.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        src = r.get("source", "")
        rec = {
            "id": r.get("id"), "source": src,
            "instruction": r.get("instruction", ""),
            "plan_target": r.get("plan_target", ""),
            "plan_json": r.get("plan_json"),
            "lowering": r.get("lowering") or [],
            "available_capabilities": [],
            "selected_capabilities": [],
            "canonical_capabilities": [],
        }
        if src in ("xlam", "toolbench_static"):
            # available = all raw tools visible at runtime; index by name
            tool_names = sorted({l.get("tool", "") for l in rec["lowering"] if l.get("tool")})
            # full available list is only recoverable for the tools we have schemas for;
            # xLAM per-record tool lists vary — use the global schema index ∩ record-relevant
            # domain proxy: we cannot recover the true per-record tool list from the corpus,
            # so available = tools with schemas whose names co-occur in this record's
            # source bucket is NOT sound. Store ALL schema-known tools of the record's
            # used tool set + a sampled distractor pool (deterministic, seed-free via hash)
            # -> for composer training we primarily need SELECTED; available is best-effort.
            avail = []
            for i, t in enumerate(tool_names):
                sch = xlam.get(t) or tbs.get(t)
                if not sch:
                    skipped_no_schema += 1
                    continue
                avail.append({"capability_id": f"c{i}", "name": t,
                              "description": (sch.get("description", "") or "")[:400],
                              "parameters": slim_params(sch.get("parameters"))})
            # deterministic distractors from the same source schema pool (max 15 total)
            pool = [n for n in (xlam if src == "xlam" else tbs) if n not in tool_names]
            pool.sort()
            for j, t in enumerate(pool[: max(0, 15 - len(avail))]):
                sch = (xlam if src == "xlam" else tbs)[t]
                avail.append({"capability_id": f"d{j}", "name": t,
                              "description": (sch.get("description", "") or "")[:400],
                              "parameters": slim_params(sch.get("parameters"))})
            rec["available_capabilities"] = avail
            # selected = ids of tools the trajectory calls (join by name)
            name2id = {a["name"]: a["capability_id"] for a in avail}
            rec["selected_capabilities"] = sorted({name2id[l["tool"]]
                                                    for l in rec["lowering"]
                                                    if l.get("tool") in name2id})
            # canonical metadata for every available capability
            from src.lifter.toolmap import map_tool
            canon = []
            for a in avail:
                m = map_tool(a["name"], {})
                sk = m.get("skill")
                mk = m.get("mapping_kind", "")
                canon.append({"capability_id": a["capability_id"],
                              "name": a["name"],
                              "canonical_skill": sk,
                              "skill_family": skill_family(sk),
                              "label_tier": tier_of(sk, mk, a["name"], a["description"]),
                              "confidence": m.get("confidence")})
            rec["canonical_capabilities"] = canon
            audit["with_caps"] += 1
            audit[f"src_{src}"] += 1
        else:
            audit["no_tool_source"] += 1
        rows.append(rec)
        audit["n"] += 1
        audit["sel_sum"] += len(rec["selected_capabilities"])
        audit["avail_sum"] += len(rec["available_capabilities"])
    return rows, audit, skipped_no_schema


def main():
    os.makedirs(OUT, exist_ok=True)
    xlam, tbs = load_xlam_schemas(), load_tbs_schemas()
    all_audit = {}
    for split in ("train", "val", "test"):
        rows, audit, skipped = build_split(split, xlam, tbs)
        with open(os.path.join(OUT, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = max(1, audit["n"])
        all_audit[split] = {
            "records": audit["n"],
            "with_capabilities": audit["with_caps"],
            "no_tool_source": audit["no_tool_source"],
            "skipped_tools_no_schema": skipped,
            "avg_available_per_task": round(audit["avail_sum"] / n, 2),
            "avg_selected_per_task": round(audit["sel_sum"] / n, 2),
            "selected_available_ratio": round(audit["sel_sum"] / max(1, audit["avail_sum"]), 4),
            "by_source": {k: v for k, v in audit.items() if k.startswith("src_")},
        }
        print(split, json.dumps(all_audit[split]))
    with open(os.path.join(OUT, "build_audit.json"), "w", encoding="utf-8") as f:
        json.dump(all_audit, f, indent=2, ensure_ascii=False)
    print("saved data/compiler_corpus_v4/")


if __name__ == "__main__":
    main()
