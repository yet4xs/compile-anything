"""Near-duplicate detection and group-aware split (Phase 5B-0 Task 4).

- normalize(text): canonical instruction form
- minhash signatures + banded LSH -> candidate pairs -> true Jaccard check
- union-find families: identical normalized instructions OR Jaccard >= 0.8
  (optionally same op-sequence with Jaccard >= 0.5) never cross splits
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

N_PERM = 32
BANDS = 8          # rows per band = N_PERM // BANDS = 4
NEAR_THRESHOLD = 0.8
OPSEQ_THRESHOLD = 0.5

_STOP = set("a an the of to for in on at with and or is are do does my your "
            "me i you it this that from by as be".split())


def normalize(text: str) -> str:
    return " ".join(t for t in re.sub(r"[^a-z0-9 ]", " ",
                                      (text or "").lower()).split())


def tokens(text: str) -> List[str]:
    return [t for t in normalize(text).split() if t not in _STOP]


def _h(seed: int, tok: str) -> int:
    return int.from_bytes(
        hashlib.md5(f"{seed}|{tok}".encode()).digest()[:8], "big")


def minhash(toks: Iterable[str]) -> Tuple[int, ...]:
    ts = list(toks)
    if not ts:
        ts = ["<empty>"]
    sig = []
    for i in range(N_PERM):
        sig.append(min(_h(i, t) for t in ts))
    return tuple(sig)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def near_duplicate_families(texts: List[str],
                            op_seqs: List[str] = None,
                            near_threshold: float = NEAR_THRESHOLD,
                            opseq_threshold: float = OPSEQ_THRESHOLD
                            ) -> List[int]:
    """family id per text.

    Union rule: docs sharing ANY LSH band signature are unioned. This can
    over-merge (band collision without true Jaccard confirmation) — the
    conservative direction for split integrity (never splits a near-dup
    pair). True near-dup PAIR counts (Jaccard-verified) come from
    cross_split_leakage(). op_seqs, when provided, only widens families
    via bucket collisions of identical op sequences."""
    n = len(texts)
    uf = UnionFind()
    sigs = [minhash(tokens(t)) for t in texts]
    rows = N_PERM // BANDS
    for band in range(BANDS):
        buckets: Dict[tuple, List[int]] = defaultdict(list)
        lo = band * rows
        for i, sig in enumerate(sigs):
            buckets[sig[lo:lo + rows]].append(i)
        for idxs in buckets.values():
            for j in range(1, len(idxs)):
                uf.union(idxs[0], idxs[j])
    if op_seqs is not None:
        op_buckets: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(op_seqs):
            op_buckets[s].append(i)
        for idxs in op_buckets.values():
            # same semantic op sequence: union only if token sets overlap at
            # all (cheap pre-filter); keeps unrelated phrasings separate
            for j in range(1, len(idxs)):
                if _jaccard(set(tokens(texts[idxs[0]])),
                            set(tokens(texts[idxs[j]]))) > 0:
                    uf.union(idxs[0], idxs[j])
    return [uf.find(i) for i in range(n)]


def group_split(groups: List[int], train=0.90, val=0.05, test=0.05,
                seed: int = 0) -> List[str]:
    """Assign whole families to splits, size-weighted to hit the ratios
    (deterministic largest-first greedy over shuffled tie-order)."""
    import random
    rng = random.Random(seed)
    members: Dict[int, List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    fams = sorted(members.values(), key=lambda m: -len(m))
    rng.shuffle(fams)
    n = len(groups)
    caps = {"train": n * train, "val": n * val, "test": n * test}
    fills = {"train": 0, "val": 0, "test": 0}
    assign: List[str] = [""] * n
    order = ["train", "val", "test"]
    for fam in fams:
        # pick the split with the largest remaining deficit
        deficits = {s: (caps[s] - fills[s]) / max(caps[s], 1e-9) for s in order}
        choice = max(order, key=lambda s: deficits[s])
        if caps[choice] - fills[choice] <= 0:
            choice = max(order, key=lambda s: caps[s] - fills[s])
        for i in fam:
            assign[i] = choice
        fills[choice] += len(fam)
    return assign


def cross_split_leakage(texts: List[str], splits: List[str],
                        near_threshold: float = NEAR_THRESHOLD
                        ) -> Dict[str, int]:
    """Count exact and near-duplicate pairs that ended up in different splits.
    Exact pairs via normalized-text hash; near pairs via LSH candidates."""
    from collections import defaultdict as dd
    exact_by_text: Dict[str, List[int]] = dd(list)
    for i, t in enumerate(texts):
        exact_by_text[normalize(t)].append(i)
    exact_cross = 0
    for ids in exact_by_text.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if splits[ids[a]] != splits[ids[b]]:
                    exact_cross += 1

    toks = [set(tokens(t)) for t in texts]
    sigs = [minhash(t) for t in toks]
    rows = N_PERM // BANDS
    near_cross = 0
    checked = set()
    for band in range(BANDS):
        buckets: Dict[tuple, List[int]] = dd(list)
        lo = band * rows
        for i, sig in enumerate(sigs):
            buckets[sig[lo:lo + rows]].append(i)
        for idxs in buckets.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    key = (min(idxs[a], idxs[b]), max(idxs[a], idxs[b]))
                    if key in checked:
                        continue
                    checked.add(key)
                    if splits[key[0]] != splits[key[1]] \
                            and _jaccard(toks[key[0]], toks[key[1]]) >= near_threshold:
                        near_cross += 1
    return {"cross_split_exact_duplicates": exact_cross,
            "cross_split_near_duplicates": near_cross}
