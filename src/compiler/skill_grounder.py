"""Phase 6A skill grounder: prompts, parsing, metrics, counterfactuals.

Skill definitions are auto-built from src/isa/registry.py (no second
hand-written copy). Output format is a single bare skill label.
"""
from __future__ import annotations
import random
import re
from collections import Counter
from typing import Dict, List, Optional

from src.isa import registry

BOUNDARY = {"RETRIEVAL": {"SEARCH", "FETCH", "QUERY_DB"},
            "ACTION": {"EXEC_ACTION", "SEND", "SAVE"}}

LABEL_SET = sorted(registry.REGISTRY.keys())


def skill_family(skill: str) -> str:
    for fam, skills in BOUNDARY.items():
        if skill in skills:
            return fam
    return "OTHER"


def skill_definitions_block(skills: Optional[List[str]] = None) -> str:
    """Auto-built skill definitions from the ISA registry."""
    names = skills or LABEL_SET
    lines = []
    for name in sorted(names):
        spec = registry.get(name)
        if spec is None:
            continue
        ins = ",".join(spec.input_types) if spec.input_types else "-"
        out = spec.output_rule or "-"
        lines.append(f"- {name} ({spec.cls}): {spec.desc} [inputs: {ins}; output: {out}]")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a capability grounder for a task compiler. Given a tool/capability "
    "schema, map it to exactly ONE canonical Skill ISA label below. Answer with "
    "the label ONLY — no explanation, no punctuation.\n\n"
    "Skill ISA:\n{skills}"
)


def format_capability(cap: Dict, transform: str = "full") -> str:
    """Render a capability schema, optionally counterfactually transformed.

    transform: full | name_masked | desc_masked | name_perturbed
    """
    name = cap.get("name", "")
    desc = (cap.get("description") or "").strip()
    params = cap.get("parameters") or {}
    req = params.get("required") or []
    props = params.get("properties") or {}
    if transform == "name_masked":
        name = "tool_17"
    elif transform == "name_perturbed":
        name = "api_x7"
    elif transform == "desc_masked":
        desc = ""
    pstr = ", ".join(f"{k}:{props[k]}" for k in list(props)[:8]) or "-"
    rstr = ", ".join(req) or "-"
    block = f"name: {name}\nparameters: {pstr}\nrequired: {rstr}"
    if desc:
        block = f"name: {name}\ndescription: {desc}\nparameters: {pstr}\nrequired: {rstr}"
    return block


def build_probe_prompt(sample: Dict, probe: str = "A",
                       transform: str = "full",
                       skill_block: Optional[str] = None,
                       candidates: Optional[List[Dict]] = None) -> str:
    """Probe A: schema-only. Probe B: task+schema. Probe C: selection."""
    skills_txt = skill_block if skill_block is not None else skill_definitions_block()
    cap_txt = format_capability(sample["capability"], transform)
    task = (sample.get("instruction") or "").strip()
    if probe == "A":
        user = (f"Capability:\n{cap_txt}\n\n"
                "Which canonical Skill ISA label does this capability lower to? "
                "Answer with the label only.")
    elif probe == "B":
        user = (f"Task:\n{task}\n\nCapability:\n{cap_txt}\n\n"
                "Which canonical Skill ISA label should this capability lower to "
                "for this task? Answer with the label only.")
    elif probe == "C":
        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(f"[{i}] {format_capability(c, transform)}")
        user = (f"Task:\n{task}\n\nAvailable capabilities:\n" + "\n".join(lines) +
                "\n\nWhich capability does the task require, and what Skill ISA "
                "label does it lower to? Answer exactly as: capability_id=<n>; "
                "skill=<LABEL>")
    else:
        raise ValueError(probe)
    return {"system": SYSTEM_PROMPT.format(skills=skills_txt), "user": user}


_LABEL_RE = re.compile(
    r"\b(SEARCH|FETCH|QUERY_DB|EXEC_ACTION|SEND|SAVE|LOAD|CALCULATE|CODEGEN|"
    r"CLASSIFY|EXTRACT_ENTITIES|SUMMARIZE|CONVERT|TRANSLATE|FILTER|TRANSFORM|"
    r"EXTRACT|DEDUP|SORT|JOIN|MERGE|ARGMIN|ARGMAX|MIN|MAX|SUM|AVG|COUNT|"
    r"COMPARE|GENERATE|PLAN|VERIFY|SELECT|NO_CALL)\b")


def parse_label(text: str) -> Optional[str]:
    m = _LABEL_RE.search(text.upper())
    return m.group(1) if m else None


def parse_probe_c(text: str):
    m = re.search(r"capability_id\s*=\s*(\d+)\s*;\s*skill\s*=\s*([A-Z_]+)", text)
    if m:
        return int(m.group(1)), m.group(2)
    cid = re.search(r"\b(\d)\b", text)
    lab = parse_label(text)
    if cid and lab:
        return int(cid.group(1)), lab
    return None, lab


# ── metrics ──

def boundary_metrics(samples: List[Dict], preds: List[Optional[str]]):
    """Primary metric: RETRIEVAL vs ACTION binary on boundary-skill samples."""
    y, p = [], []
    for s, pred in zip(samples, preds):
        fam = skill_family(s["target_skill"])
        if fam == "OTHER":
            continue
        y.append(fam)
        p.append(skill_family(pred) if pred and skill_family(pred) != "OTHER"
                 else ("OTHER" if pred else "MISS"))
    n = len(y)
    if n == 0:
        return {"n": 0}
    acc = sum(a == b for a, b in zip(y, p)) / n
    per = {}
    conf = Counter()
    for a, b in zip(y, p):
        conf[(a, b)] += 1
    for cls in ("RETRIEVAL", "ACTION"):
        tp = conf[(cls, cls)]
        fp = sum(v for (a, b), v in conf.items() if b == cls and a != cls)
        fn = sum(v for (a, b), v in conf.items() if a == cls and b != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[cls] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": tp + fn}
    macro_f1 = sum(v["f1"] for v in per.values()) / len(per)
    return {
        "n": n,
        "binary_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per,
        "confusion": {
            "retrieval_to_action": conf[("RETRIEVAL", "ACTION")],
            "action_to_retrieval": conf[("ACTION", "RETRIEVAL")],
            "action_to_miss": conf[("ACTION", "MISS")],
            "retrieval_to_miss": conf[("RETRIEVAL", "MISS")],
        },
    }


def exact_metrics(samples: List[Dict], preds: List[Optional[str]],
                  min_support: int = 0):
    """Exact-skill multiclass over skills present in samples."""
    support = Counter(s["target_skill"] for s in samples)
    skills = {k for k, v in support.items() if v >= min_support}
    conf = Counter()
    n_in = 0
    for s, pred in zip(samples, preds):
        if s["target_skill"] not in skills:
            continue
        n_in += 1
        conf[(s["target_skill"], pred or "MISS")] += 1
    per = {}
    f1s = []
    for sk in sorted(skills):
        tp = conf[(sk, sk)]
        fp = sum(v for (a, b), v in conf.items() if b == sk and a != sk)
        fn = sum(v for (a, b), v in conf.items() if a == sk and b != sk)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[sk] = {"precision": round(prec, 4), "recall": round(rec, 4),
                   "f1": round(f1, 4), "support": tp + fn}
        f1s.append(f1)
    micro = sum(conf[(sk, sk)] for sk in skills) / n_in if n_in else 0.0
    return {"n": n_in,
            "micro_f1": round(micro, 4),
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "per_skill": per}


# ── counterfactual + NO_CALL construction ──

def make_no_call(sample: Dict, distractors: List[Dict]) -> Dict:
    """Synthetic irrelevance probe: correct capability removed, only distractors."""
    return {
        "sample_id": sample["sample_id"] + "-nc",
        "instruction": sample["instruction"],
        "candidates": distractors,
        "target": "NO_CALL",
        "synthetic_negative": True,
    }


def pick_hard_negatives(sample: Dict, train_pool: List[Dict], k: int = 4,
                        rng: Optional[random.Random] = None):
    """4 hard negatives from TRAIN groups: prefer same source, same lexical
    family (share first token), different semantic class."""
    if rng is None:
        rng = random.Random(0)
    fam_class = skill_family(sample["target_skill"])
    first_tok = sample["tool_name"].split("_")[0].lower()
    same_src = [c for c in train_pool if c["source"] == sample["source"]
                and c["sample_id"] != sample["sample_id"]
                and skill_family(c["target_skill"]) != fam_class]
    pref_ids = set()
    pref = []
    rest = []
    for c in same_src:
        if c["tool_name"].split("_")[0].lower() == first_tok:
            pref.append(c)
            pref_ids.add(c["sample_id"])
        else:
            rest.append(c)
    rng.shuffle(pref)
    rng.shuffle(rest)
    picked = (pref + rest)[:k]
    if len(picked) < k:
        picked_ids = {c["sample_id"] for c in picked}
        extra = [c for c in train_pool
                 if skill_family(c["target_skill"]) != fam_class
                 and c["sample_id"] not in picked_ids]
        rng.shuffle(extra)
        picked += extra[:k - len(picked)]
    return picked[:k]


import random  # noqa: E402  (used by pick_hard_negatives default)
