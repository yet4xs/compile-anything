"""Phase 9B-2.3 Task 3: score predictions against the RECONSTRUCTED gold
program (tau3_gold_compiled_v1) — no index alignment, bijection-based.

Gold program nodes carry per-param sourcing:
  {"class": "LITERAL", "value": v}
  {"class": "runtime_ref", "producer_step": j, "path": p, ...}
  {"class": "runtime_ref_multi", "elements": [{producer_step, path}, ...]}

Prediction side: calls [(tool, {param: literal})] plus bindings
[(consumer_idx, param, producer_idx, field_path)] — extractable from
TaskIR (param_refs via EXTRACT_FIELD chains) or a Skeleton (runtime_ref
arguments).

Matching:
  1. Call bijection: pred call <-> gold call matched by (tool, literal
     signature); same-tool repeated instances distinguished by their
     literal signatures; legal topological reordering is irrelevant
     because positions never enter the comparison.
  2. Binding comparison through the bijection: gold (consumer, param) ->
     (producer, path) vs pred mapped through the bijection. Root-scalar
     gold paths ("" — the tool's entire output is the value) match any
     top-level scalar path (no index/dot) and are reported as
     ROOT_SCALAR_EQUIV, not as an exact path match.
  3. Only SOURCED gold params (runtime_ref / runtime_ref_multi) are
     binding-graded; LITERAL params are graded by call-literal equality.
     Unsourceable gold params (AMBIGUOUS/USER_LATENT/ENV_STATE/UNKNOWN)
     are excluded and reported as coverage gaps, never silently passed.
"""

from typing import Optional


def _lit_signature(tool, literals):
    return (tool, tuple(sorted((k, str(v)) for k, v in literals.items())))


def gold_calls_from_program(program):
    """[(tool, {param: literal_value})] — sourced params contribute no
    literal (they are graded as bindings instead)."""
    calls = []
    for node in program:
        lits = {}
        for param, a in node["arguments"].items():
            if a["class"] == "LITERAL":
                lits[param] = a["value"]
        calls.append((node["tool"], lits))
    return calls


def gold_bindings_from_program(program):
    """[(consumer_step, param, producer_step, path)] with multi-refs
    expanded per element (element index appended to param for identity)."""
    out = []
    for node in program:
        for param, a in node["arguments"].items():
            if a["class"] == "runtime_ref":
                out.append((node["step"], param, a["producer_step"],
                            a.get("path", "")))
            elif a["class"] == "runtime_ref_multi":
                for i, e in enumerate(a.get("elements", [])):
                    out.append((node["step"], f"{param}[{i}]",
                                e["producer_step"], e.get("path", "")))
    return out


def bijection(pred_calls, gold_calls):
    """Greedy exact-signature bijection. Returns (pred_to_gold, unmatched)."""
    from collections import Counter
    gold_pool = Counter(_lit_signature(t, l) for t, l in gold_calls)
    p2g = {}
    unmatched = []
    for i, (t, lits) in enumerate(pred_calls):
        sig = _lit_signature(t, lits)
        if gold_pool.get(sig, 0) > 0:
            # find a gold index with this signature
            for j, (gt, gl) in enumerate(gold_calls):
                if _lit_signature(gt, gl) == sig and j not in p2g.values():
                    p2g[i] = j
                    gold_pool[sig] -= 1
                    break
        else:
            unmatched.append(i)
    return p2g, unmatched


def _root_scalar_equiv(gold_path, pred_path):
    return gold_path == "" and pred_path and \
        "[" not in pred_path and "." not in pred_path


def score_prediction(pred_calls, pred_bindings, gold_program):
    """pred_calls: [(tool, {param: literal})]
    pred_bindings: [(consumer_idx, param, producer_idx, field_path)]
    Returns the metric dict."""
    gcalls = gold_calls_from_program(gold_program)
    gbindings = gold_bindings_from_program(gold_program)
    sourced_params = {(node["step"], param)
                      for node in gold_program
                      for param, a in node["arguments"].items()
                      if a["class"] in ("runtime_ref", "runtime_ref_multi")}
    unsourced = sum(1 for node in gold_program
                    for a in node["arguments"].values()
                    if a["class"] not in ("LITERAL", "runtime_ref",
                                          "runtime_ref_multi"))

    p2g, unmatched_calls = bijection(pred_calls, gcalls)

    details = []
    n_exact = n_equiv = 0
    for (cidx, param, pidx, ppath) in pred_bindings:
        g = p2g.get(cidx)
        producer_g = p2g.get(pidx)
        matched, how = None, None
        if g is not None and producer_g is not None:
            cands = [(pr, pp) for (gc, prm, pr, pp) in gbindings
                     if gc == g and (prm == param or
                                     prm.startswith(param + "["))]
            for pr, pp in cands:
                if pr == producer_g and pp == ppath:
                    matched, how = True, "EXACT"
                    break
            if matched is None:
                for pr, pp in cands:
                    if pr == producer_g and _root_scalar_equiv(pp, ppath):
                        matched, how = True, "ROOT_SCALAR_EQUIV"
                        break
        if matched:
            if how == "EXACT":
                n_exact += 1
            else:
                n_equiv += 1
        details.append({
            "pred_consumer": cidx, "param": param,
            "pred_producer": pidx, "pred_path": ppath,
            "gold_consumer": g, "gold_producer_mapped": producer_g,
            "matched": matched, "match_kind": how})

    n_gold_b = len(gbindings)
    n_pred_b = len(pred_bindings)
    strict = (not unmatched_calls
              and len(p2g) == len(gcalls)
              and unsourced == 0
              and n_exact + n_equiv == n_gold_b
              and n_exact + n_equiv == n_pred_b
              and n_equiv == 0)

    return {
        "n_gold_calls": len(gcalls),
        "n_pred_calls": len(pred_calls),
        "calls_bijection_complete": len(p2g) == len(gcalls)
                                    and not unmatched_calls,
        "unmatched_pred_calls": unmatched_calls,
        "n_gold_bindings": n_gold_b,
        "n_pred_bindings": n_pred_b,
        "n_bindings_exact": n_exact,
        "n_bindings_root_scalar_equiv": n_equiv,
        "gold_unsourced_params": unsourced,
        "gold_sourced_params": len(sourced_params),
        "strict_binding_exact": strict,
        "details": details,
    }


def bindings_from_skeleton_nodes(nodes):
    """[(consumer_idx, param, producer_idx, path)] from skeleton nodes
    (list of dicts with id/capability_id/arguments runtime_ref)."""
    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    out = []
    for i, n in enumerate(nodes):
        for param, v in (n.get("arguments") or {}).items():
            if isinstance(v, dict) and v.get("kind") == "runtime_ref":
                p = id_to_idx.get(v.get("producer"))
                if p is not None:
                    out.append((i, param, p, v.get("field", "")))
    return out


def calls_from_skeleton_nodes(nodes):
    return [(n["capability_id"],
             {k: v for k, v in (n.get("arguments") or {}).items()
              if not (isinstance(v, dict) and v.get("kind") == "runtime_ref")})
            for n in nodes]
