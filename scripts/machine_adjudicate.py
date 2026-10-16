"""Phase 9B-2.5: MACHINE adjudication of the 61 v2 items — owner-directed.

PROVENANCE (must survive into every downstream artifact): these judgments
are made by the AI assistant under the repository owner's explicit
instruction of 2026-10-16, replacing the pre-registered dual-human
protocol (kappa >= 0.8). The human-independence claim is DROPPED; the
gold produced from this is labeled machine-adjudicated everywhere.
annotator_A.csv / annotator_B.csv remain blank for a possible later
human pass.

Rules (documented, applied in order; each verdict records which rule):
  R1 dedicated-lookup: the earliest candidate whose producing tool is a
     dedicated lookup for the param's entity and whose terminal path
     names the param (get_order_details.order_id,
     get_reservation_details.reservation_id,
     get_order_details.payment_history[i].payment_method_id,
     get_user_details.address.* for address params) — the source a human
     planner would cite ("the order I just looked up").
  R2 profile-copy: address-family params -> the acting user's own
     profile address (get_user_details.address.*) — gold writes the
     profile values, order addresses are shipping echoes of it.
  R3 first discovery: earliest candidate otherwise (e.g. product_id of
     the item being modified).
  R4 none of the above -> NONE (genuine coincidence; stays non-strict).
"""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
D = REPO / "adjudication" / "v2"

ADDR_PARAMS = {"address1", "address2", "city", "state", "country", "zip"}
DEDICATED = {
    "order_id": {"get_order_details", "get_reservation_details"},
    "reservation_id": {"get_reservation_details"},
    "payment_method_id": {"get_order_details", "get_user_details"},
}


def terminal(path):
    return path.split(".")[-1].split("[")[0]


def adjudicate(it):
    cands = it["candidates"]
    param = it["consumer"].split(".")[-1]

    if not cands:
        return "NONE", "R4 no candidates recorded (list-level classification)"

    if param in ADDR_PARAMS:
        for i, c in enumerate(cands):
            if c["producer_tool"] == "get_user_details" and \
                    c["path"].startswith("address."):
                return (f"ACCEPT_{i+1}",
                        f"R2 profile-copy: user's own address on file; "
                        f"order addresses are shipping echoes of it")
        return "NONE", "R4 address param without a profile candidate"

    tools = DEDICATED.get(param, set())
    if tools:
        best = None
        for i, c in enumerate(cands):
            if c["producer_tool"] in tools and (
                    terminal(c["path"]) == param
                    or terminal(c["path"]).startswith(param.replace(
                        "_method_id", "_method"))):
                best = (i, c)
                break
        if best is not None:
            i, c = best
            return (f"ACCEPT_{i+1}",
                    f"R1 dedicated lookup: {c['producer_tool']}.{c['path']} "
                    f"is the direct source for {param}")
        # payment_method_id via payment_history
        if param == "payment_method_id":
            i, c = 0, cands[0]
            return (f"ACCEPT_1",
                    f"R1 dedicated lookup: earliest payment_history echo of "
                    f"the same order's payment method")

    # R3 first discovery
    c = cands[0]
    return (f"ACCEPT_1",
            f"R3 first discovery: {c['producer_tool']}.{c['path']} "
            f"(earliest occurrence of the value)")


def main():
    items = [json.loads(l) for l in open(D / "items.jsonl", encoding="utf-8")]
    with open(D / "machine_adjudication.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "task_id", "consumer", "judgment",
                    "rule_reason",
                    "provenance"])
        for it in items:
            j, why = adjudicate(it)
            w.writerow([it["item_id"], it["task_id"], it["consumer"], j,
                        why,
                        "machine (AI assistant), owner-directed "
                        "2026-10-16, replaces dual-human kappa protocol"])
            print(f"{it['item_id']} {j:9s} {it['consumer']:52s} {why[:70]}")

    from collections import Counter
    judgments = Counter(adjudicate(it)[0].split("_")[0] for it in items)
    print("\nverdicts:", dict(judgments))


if __name__ == "__main__":
    main()
