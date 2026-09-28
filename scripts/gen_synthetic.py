"""Generate N synthetic TaskIR programs + (NL task -> TaskIR) training pairs.

Structural archetypes (workload diversity for dataset statistics):
  linear        search -> filter/sort -> aggregate -> generate -> verify
  parallel      two searches -> merge -> filter -> generate -> verify
  branch        search -> draft -> verify -> guarded(t/f) branches -> select
  compute       load -> extract -> two calculates -> compare -> generate
  summary       fetch -> summarize -> generate -> verify
  entities      search(news) -> extract_entities -> filter -> generate

Every program is valid by construction (types tracked while building),
and every file is re-validated after writing.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.taskir import Module, Node, Program, Guard, Retry, save_module, to_text, module_to_dict  # noqa: E402
from src.validator.validator import validate  # noqa: E402

DOMAINS = ["flight", "hotel", "weather", "news", "maps", "product", "paper", "stock"]
PLURAL = {"flight": "flights", "hotel": "hotels", "weather": "forecasts",
          "news": "articles", "maps": "routes", "product": "products",
          "paper": "papers", "stock": "stocks"}
PRICE_KEY = {"flight": "price", "hotel": "price_per_night", "weather": "temp_c",
             "news": "date", "maps": "distance_km", "product": "price",
             "paper": "citations", "stock": "price"}

# NL diversity: randomized qualifiers/intents so dedup-by-task keeps most pairs
QUALIFIERS = ["under $500", "for next month", "with the best reviews",
              "arriving before 8 PM", "with free cancellation",
              "for a party of two", "rated 4 stars or higher",
              "departing early morning", "for this weekend",
              "with flexible dates", "near the city center",
              "within my loyalty program", "for a week-long trip",
              "with the shortest layover", "bookable today"]
SUPERLATIVE = ["best", "cheapest", "top-rated", "fastest", "most reliable",
               "highest-value"]
EXPLAIN = ["explain the choice", "tell me why", "and justify it with the data",
           "and summarize the reasoning in two bullets", "and flag any risks"]


def _q(rng, archetype_nl: str) -> str:
    return f"{archetype_nl} {rng.choice(QUALIFIERS)}."


def gen_linear(rng):
    d1 = rng.choice(DOMAINS)
    q = _q(rng, f"Find the {rng.choice(SUPERLATIVE)} {PLURAL[d1]} matching my "
                f"constraints and {rng.choice(EXPLAIN)}")
    nodes = [Node(id="%1", op="SEARCH", inputs=["@task"],
                  params={"domain": d1}, output_type=f"List[{d1.capitalize()}]")]
    cur = _maybe_list_op(rng, nodes, "%1", d1)
    agg = Node(id=f"%{len(nodes)+1}", op=rng.choice(["ARGMIN", "ARGMAX", "MIN", "MAX"]),
               inputs=[cur], params={"key": PRICE_KEY[d1]})
    nodes.append(agg)
    out = _tail(rng, nodes, agg.id)
    return q, nodes, out


def gen_parallel(rng):
    d1, d2 = rng.sample(DOMAINS, 2)
    q = _q(rng, f"Compare {PLURAL[d1]} and {PLURAL[d2]} options for my request "
                f"and {rng.choice(EXPLAIN)}")
    nodes = [
        Node(id="%1", op="SEARCH", inputs=["@task"], params={"domain": d1},
             output_type=f"List[{d1.capitalize()}]"),
        Node(id="%2", op="SEARCH", inputs=["@task"], params={"domain": d2},
             output_type=f"List[{d2.capitalize()}]"),
        Node(id="%3", op="MERGE", inputs=["%1", "%2"]),
    ]
    cur = _maybe_list_op(rng, nodes, "%3", d1)
    out = _tail(rng, nodes, cur)
    return q, nodes, out


def gen_branch(rng):
    d1 = rng.choice(DOMAINS)
    q = _q(rng, f"Answer my question about {PLURAL[d1]} from retrieved evidence, "
                f"only trust the answer if it checks out, otherwise say so "
                f"plainly")
    nodes = [
        Node(id="%1", op="SEARCH", inputs=["@task"], params={"domain": d1},
             output_type=f"List[{d1.capitalize()}]"),
        Node(id="%2", op="GENERATE", inputs=["%1", "@task"],
             params={"role": "draft"}),
        Node(id="%3", op="VERIFY", inputs=["%2"],
             params={"check": "factual_consistency"}),
        Node(id="%4", op="GENERATE", inputs=["%2"],
             params={"role": "polished_answer"},
             guard=Guard(cond="%3", expect=True)),
        Node(id="%5", op="GENERATE", inputs=["@task"],
             params={"role": "fallback_answer"},
             guard=Guard(cond="%3", expect=False)),
        Node(id="%6", op="SELECT", inputs=["%3", "%4", "%5"]),
    ]
    return q, nodes, "%6"


def gen_compute(rng):
    gap = rng.choice(["with the exact gap", "as a percentage",
                      "with confidence intervals"])
    q = _q(rng, "Load the dataset, compute both metrics and tell me which one "
                f"is larger, {gap}")
    nodes = [
        Node(id="%1", op="LOAD", inputs=["@task"],
             params={"source": "dataset.csv"}),
        Node(id="%2", op="EXTRACT", inputs=["%1"],
             params={"fields": ["value_a", "value_b"]}),
        Node(id="%3", op="CALCULATE", inputs=["%2"],
             params={"expr": "mean(value_a)"}),
        Node(id="%4", op="CALCULATE", inputs=["%2"],
             params={"expr": "mean(value_b)"}),
        Node(id="%5", op="COMPARE", inputs=["%3", "%4"], params={"op": ">"}),
    ]
    out = _tail(rng, nodes, "%5")
    return q, nodes, out


def gen_summary(rng):
    style = rng.choice(["before my 9 AM meeting", "in plain language",
                        "as talking points"])
    q = _q(rng, "Fetch the document at the URL and give me a summary I can "
                f"act on {style}")
    nodes = [
        Node(id="%1", op="FETCH", inputs=["@task"], params={"url": "input_url"}),
        Node(id="%2", op="SUMMARIZE", inputs=["%1"]),
    ]
    out = _tail(rng, nodes, "%2")
    return q, nodes, out


def gen_entities(rng):
    detail = rng.choice(["with their roles", "with dates", "with sources"])
    q = _q(rng, "Search the news for my topic, list the key entities and "
                f"brief me on them {detail}")
    nodes = [
        Node(id="%1", op="SEARCH", inputs=["@task"], params={"domain": "news"},
             output_type="List[News]"),
        Node(id="%2", op="EXTRACT_ENTITIES", inputs=["%1"]),
    ]
    cur = _maybe_list_op(rng, nodes, "%2", "news")
    out = _tail(rng, nodes, cur)
    return q, nodes, out


def _maybe_list_op(rng, nodes, prev_id, domain):
    """Insert an optional list->list transform (type-preserving)."""
    if rng.random() < 0.35:
        op = rng.choice(["FILTER", "SORT", "DEDUP"])
        params = ({"predicate": f"{PRICE_KEY[domain]} in acceptable_range"}
                  if op == "FILTER" else
                  {"key": PRICE_KEY[domain], "order": "asc"} if op == "SORT" else {})
        n = Node(id=f"%{len(nodes)+1}", op=op, inputs=[prev_id], params=params)
        nodes.append(n)
        return n.id
    return prev_id


def _tail(rng, nodes, out_id, nl_role="final_answer"):
    """GENERATE + optional-retry VERIFY tail; returns output id."""
    gen = Node(id=f"%{len(nodes)+1}", op="GENERATE", inputs=[out_id, "@task"],
               params={"role": nl_role})
    nodes.append(gen)
    if rng.random() < 0.5:
        ver = Node(id=f"%{len(nodes)+1}", op="VERIFY", inputs=[gen.id],
                   params={"check": "grounded_in_retrieved_evidence"})
        nodes.append(ver)
        gen.retry = Retry(max_attempts=rng.choice([2, 3]), on=ver.id)
    return gen.id


ARCHETYPES = [gen_linear, gen_parallel, gen_branch,
              gen_compute, gen_summary, gen_entities]


def build_program(rng, idx) -> Module:
    fn = rng.choice(ARCHETYPES)
    q, nodes, out = fn(rng)
    prog = Program(name=f"synthetic_{idx:05d}", description=q,
                   inputs=[{"name": "@task", "type": "Str"}],
                   nodes=nodes, output=out)
    return Module(program=prog, meta={"name": prog.name,
                                      "provenance": {"source": "synthetic",
                                                     "archetype": fn.__name__}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    outdir = ROOT / "data" / "taskir" / "synthetic"
    traindir = ROOT / "data" / "train"
    outdir.mkdir(parents=True, exist_ok=True)
    traindir.mkdir(parents=True, exist_ok=True)

    pairs_path = traindir / "synthetic_pairs.jsonl"
    n_ok, n_bad = 0, 0
    with open(pairs_path, "w", encoding="utf-8") as pf:
        for i in range(1, args.n + 1):
            mod = build_program(rng, i)
            rep = validate(mod)
            if not rep.valid:
                n_bad += 1
                print(f"INVALID synthetic {i}: {rep.summary()}")
                continue
            save_module(mod, outdir / f"syn_{i:05d}.json")
            pf.write(__import__("json").dumps({
                "task": mod.program.description,
                "taskir_text": to_text(mod),
                "taskir_json": module_to_dict(mod),
            }, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"synthetic: {n_ok} written to {outdir} "
          f"({n_bad} invalid rejected), pairs -> {pairs_path}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
