"""Cohen's kappa for the v2 adjudication (annotator_A vs annotator_B).

Usage: python scripts/compute_kappa.py
Reads adjudication/v2/annotator_A.csv and annotator_B.csv, reports
observed agreement, kappa, disagreement list, and pass/fail vs the
pre-registered threshold (kappa >= 0.8).
"""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
D = REPO / "adjudication" / "v2"


def load(who):
    out = {}
    with open(D / f"annotator_{who}.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            j = (row.get("judgment") or "").strip().upper()
            if j:
                out[row["item_id"]] = j
    return out


def kappa(a, b):
    ids = sorted(set(a) & set(b))
    if not ids:
        return 0.0, 0, []
    cats = sorted(set(a.values()) | set(b.values()))
    po = sum(a[i] == b[i] for i in ids) / len(ids)
    pe = sum((sum(a[i] == c for i in ids) / len(ids)) *
             (sum(b[i] == c for i in ids) / len(ids)) for c in cats)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    dis = [(i, a[i], b[i]) for i in ids if a[i] != b[i]]
    return k, len(ids), dis


def main():
    a, b = load("A"), load("B")
    if not a or not b:
        print("both annotators must have judgments saved first")
        return
    k, n, dis = kappa(a, b)
    print(f"common items judged: {n}")
    print(f"agreement: {n - len(dis)}/{n}")
    print(f"cohen kappa: {k:.4f}  "
          f"({'PASS' if k >= 0.8 else 'FAIL'} vs threshold 0.8)")
    if dis:
        print(f"disagreements ({len(dis)}) -> arbitration.csv:")
        for i, x, y in dis:
            print(f"  {i}: A={x} B={y}")
        with open(D / "arbitration.csv", "a", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            for i, x, y in dis:
                w.writerow([i, "", "", "", "", "", "", x, y, "", ""])
    else:
        print("no disagreements; arbitration not needed")


if __name__ == "__main__":
    main()
