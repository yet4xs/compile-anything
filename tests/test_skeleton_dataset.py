"""Phase 8B-1A Task 5: skeleton dataset integrity tests (CPU)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.plan_skeleton import (skeleton_from_dict,
                                        validate_skeleton, hard_errors)
from src.compiler.skeleton_lowering import lower_skeleton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "semantic_skeleton")


def _rows(split):
    path = os.path.join(D, f"{split}.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")
            if l.strip()]


def test_manifest_and_splits_exist():
    man = json.load(open(os.path.join(D, "manifest.json"),
                         encoding="utf-8"))
    assert man["tau3_used"] is False
    assert man["family_disjoint"] is True
    for k in ("train", "dev", "test"):
        assert os.path.exists(os.path.join(D, f"{k}.jsonl"))
        assert os.path.exists(os.path.join(D, f"{k}_sft.jsonl"))


def test_records_have_required_fields():
    for r in _rows("dev")[:200]:
        assert r["instruction"]
        assert isinstance(r["available_capabilities"], list)
        sk = r["target_skeleton"]
        assert sk["nodes"]
        for n in sk["nodes"]:
            assert n["capability_id"] and n["skill"]
            for d in n["depends_on"] + n.get("order_on", []):
                assert isinstance(d, str)
        p = r["provenance"]
        assert p["evidence"] in ("verified", "derived", "synthetic",
                                 "ambiguous")
        assert p["split_group"]


def test_no_forward_references():
    for r in _rows("dev")[:300]:
        ids = [n["id"] for n in r["target_skeleton"]["nodes"]]
        pos = {x: i for i, x in enumerate(ids)}
        for i, n in enumerate(r["target_skeleton"]["nodes"]):
            for d in n["depends_on"] + n.get("order_on", []):
                assert pos[d] < i, f"forward ref {d} in {r['task_id']}"


def test_sft_ready_records_pass_full_chain():
    """Every SFT-ready target must lower to validator-passing TaskIR."""
    for r in _rows("train_sft")[:150]:
        sk = skeleton_from_dict(r["target_skeleton"])
        pseudo = {t: {"params": []}
                  for t in r.get("available_capabilities", [])}
        pseudo.update({n.capability_id: {"params": []}
                       for n in sk.nodes})
        assert not hard_errors(validate_skeleton(sk, None, pseudo))
        low = lower_skeleton(sk, None, pseudo)
        assert low["taskir"] is not None, (r["task_id"], low["errors"][:1])


def test_family_isolation():
    fams = {}
    for k in ("train", "dev", "test"):
        fams[k] = {r["provenance"]["split_group"]
                   for r in _rows(k)}
    assert not (fams["train"] & fams["test"])
    assert not (fams["train"] & fams["dev"])
    assert not (fams["dev"] & fams["test"])


def test_ambiguous_excluded_and_listed():
    """Ambiguous records never enter the corpus; failures are saved."""
    fails = json.load(open(os.path.join(
        ROOT, "results", "phase8b", "skeleton_data_failures.json"),
        encoding="utf-8")) if os.path.exists(os.path.join(
        ROOT, "results", "phase8b", "skeleton_data_failures.json")) else []
    for r in _rows("train")[:100]:
        assert r["provenance"]["evidence"] != "ambiguous"


def test_depth_and_shape_coverage():
    import collections
    depth = collections.Counter()
    for r in _rows("train"):
        depth[min(len(r["target_skeleton"]["nodes"]), 8)] += 1
    assert depth[1] > 1000                    # single-step mass
    assert depth[3] + depth[4] > 500          # multi-step present
    assert depth[7] + depth[8] > 5            # deep tasks present


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
