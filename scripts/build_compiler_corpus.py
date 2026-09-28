"""Build the compiler corpus: benchmark samples -> TaskIR -> validate ->
simulate -> unified records (Phase 4 Task 5/6).

    python scripts/build_compiler_corpus.py                  # 10000 samples
    python scripts/build_compiler_corpus.py --n-toolbench 100

Source mix (default): toolbench 4000 / code 3000 / sql 2000 / rtl 1000.

PROVENANCE: samples are SCHEMA-FAITHFUL SYNTHETIC records in each
benchmark's format (ToolBench/API-Bank, HumanEval/MBPP, Spider/BIRD,
RTL/EDA) so the lift pipeline is exercised end-to-end offline — they are
NOT draws from the real datasets. Swap in real data with --raw-dir
(files: toolbench.jsonl / code.jsonl / sql.jsonl / rtl.jsonl with the
same schemas); see docs/dataset-design.md.

Every record follows data/schema/compiler_sample.json. Coverage
accounting (lifted / valid / unsupported-by-reason) is emitted in
stats.json — Phase 4's core question is whether TaskIR can cover the
real task distribution, so unsupported samples are counted, not hidden.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.taskir import module_to_dict, to_text            # noqa: E402
from src.validator.validator import validate                  # noqa: E402
from src.runtime.simulator import Simulator                   # noqa: E402
from src.lifter.benchmark import lift_sample                  # noqa: E402
from src.stats import program_metrics, aggregate              # noqa: E402

# ------------------------------------------------------------- generators

CITIES = ["SFO", "JFK", "LAX", "ORD", "SEA", "BOS", "Austin", "Denver"]
INTL = ["Tokyo", "Paris", "London", "Berlin", "Singapore", "Sydney", "Seoul"]
DATES = ["2026-10-05", "2026-10-12", "2026-11-01", "2026-11-20", "2026-12-03"]
CURR = ["JPY", "EUR", "GBP", "CNY", "AUD", "CHF"]
SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "AMD"]
TOPICS = ["quantum computing", "AI compilers", "electric aviation",
          "fusion energy", "edge AI", "robotics", "photonic chips"]
LANGS = ["Japanese", "French", "German", "Spanish", "Chinese", "Korean"]
EVENTS = ["team offsite", "product review", "investor meeting", "workshop"]
NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley"]

TOOLBENCH_CHAINS = [
    ((("FlightSearch.search", {"origin": "{c1}", "destination": "{c2}", "date": "{d1}"}),
      ("CurrencyConverter.convert", {"amount": "{price}", "from_currency": "USD", "to_currency": "{cur}"})),
     "Find flights from {c1} to {c2} on {d1} and convert the cheapest price to {cur}."),
    ((("WeatherAPI.forecast", {"city": "{c1}", "date": "{d1}"}),
      ("CalendarService.create_event", {"title": "{ev}", "date": "{d1}", "time": "10:00"})),
     "Check the weather in {c1} on {d1} and schedule my {ev} that morning if it looks fine."),
    ((("NewsAPI.search", {"query": "{t}", "category": "technology"}),
      ("TextSummarizer.summarize", {}),
      ("EmailClient.send", {"to": "{name}@example.com", "subject": "Briefing: {t}"})),
     "Get the latest on {t}, summarize it, and email the summary to {name}."),
    ((("StockQuote.get_price", {"symbol": "{s}"}),
      ("Calculator.evaluate", {"expression": "{s} price * 1.05"}),
      ("EmailClient.send", {"to": "{name}@example.com", "subject": "{s} update"})),
     "Look up the current {s} price, project it 5% higher, and send the number to {name}."),
    ((("MapsService.get_directions", {"origin": "{c1}", "destination": "{c2}"}),
      ("Translator.translate", {"text": "directions summary", "target_lang": "{lang}"})),
     "Find directions from {c1} to {c2} and translate the summary into {lang}."),
    ((("HotelDB.find", {"location": "{c2}", "check_in": "{d1}", "check_out": "{d2}"}),
      ("CurrencyConverter.convert", {"amount": "{price}", "from_currency": "USD", "to_currency": "{cur}"})),
     "Find hotels in {c2} from {d1} to {d2} and convert the nightly rate to {cur}."),
    ((("WebSearch.query", {"query": "{t} tutorial"}),
      ("TextSummarizer.summarize", {})),
     "Search for a good {t} tutorial and summarize the key steps."),
    ((("FlightSearch.search", {"origin": "{c1}", "destination": "{c2}", "date": "{d1}"}),
      ("FlightBook.book", {"flight_id": "{fid}", "passenger": "{name}"})),
     "Find a flight from {c1} to {c2} on {d1} and book the best option for {name}."),
    ((("Translator.translate", {"text": "project update", "target_lang": "{lang}"}),
      ("EmailClient.send", {"to": "{name}@example.com", "subject": "Update ({lang})"})),
     "Translate my project update into {lang} and send it to {name}."),
    ((("RestaurantFinder.search", {"location": "{c1}", "cuisine": "{food}"}),
      ("MapsService.get_directions", {"origin": "{c1}", "destination": "{food} place"})),
     "Find a good {food} restaurant near {c1} and get me directions."),
    ((("ShoppingSearch.find", {"product": "{gadget}", "max_price": "{price}"}),
      ("EmailClient.send", {"to": "{name}@example.com", "subject": "{gadget} deal"})),
     "Find the best {gadget} under the budget and email the pick to {name}."),
    ((("MovieDB.search", {"genre": "{genre}"}),
      ("CalendarService.create_event", {"title": "movie night", "date": "{d1}", "time": "20:00"})),
     "Find a highly rated {genre} movie and schedule a movie night on {d1}."),
    ((("NewsAPI.search", {"query": "{t}", "category": "science"}),
      ("TextSummarizer.summarize", {}),
      ("CalendarService.create_event", {"title": "read {t} digest", "date": "{d1}", "time": "18:00"})),
     "Find recent {t} articles, summarize them, and block evening time on {d1} to read."),
    ((("WeatherAPI.forecast", {"city": "{c1}", "date": "{d1}"}),
      ("WebSearch.query", {"query": "indoor activities {c1}"}),
      ("TextSummarizer.summarize", {})),
     "Check the weather in {c1} on {d1}; if it is bad, find indoor activities and summarize."),
    ((("StockQuote.get_price", {"symbol": "{s}"}),
      ("Calculator.evaluate", {"expression": "{s} change percent"}),
      ("Translator.translate", {"text": "market note", "target_lang": "{lang}"})),
     "Get the {s} price change, compute the percent, and translate a short market note into {lang}."),
]
FOODS = ["Italian", "Thai", "Japanese", "Mexican", "Indian"]
GADGETS = ["laptop", "camera", "keyboard", "headphones", "monitor"]
GENRES = ["sci-fi", "documentary", "thriller", "comedy", "drama"]


def gen_toolbench(rng, i):
    chain, qtpl = TOOLBENCH_CHAINS[i % len(TOOLBENCH_CHAINS)]
    env = {"c1": rng.choice(CITIES), "c2": rng.choice(INTL), "d1": rng.choice(DATES),
           "d2": rng.choice(DATES), "cur": rng.choice(CURR), "s": rng.choice(SYMBOLS),
           "t": rng.choice(TOPICS), "lang": rng.choice(LANGS), "ev": rng.choice(EVENTS),
           "name": rng.choice(NAMES), "price": rng.choice([120, 320, 450, 540, 680, 810]),
           "fid": f"FL{rng.randint(100, 999)}", "food": rng.choice(FOODS),
           "gadget": rng.choice(GADGETS), "genre": rng.choice(GENRES)}
    q = qtpl.format(**env)
    traj = [{"tool": name, "args": {k: (v.format(**env) if isinstance(v, str) else v)
                                    for k, v in args.items()}}
            for name, args in chain]
    return {"task_id": f"tb-{i:05d}", "instruction": q, "trajectory": traj}


CODE_FIELDS = ["price", "score", "weight", "age", "rating", "distance", "freq"]
CODE_PATTERNS = [
    ("sort", "Sort the records by {f} in ascending order.",
     'def f(items):\n    return sorted(items, key=lambda x: x["{f}"])'),
    ("sort_desc", "Sort the records by {f}, highest first.",
     'def f(items):\n    return sorted(items, key=lambda x: x["{f}"], reverse=True)'),
    ("argmin", "Return the record with the smallest {f}.",
     'def f(items):\n    return min(items, key=lambda x: x["{f}"])'),
    ("argmax", "Return the record with the largest {f}.",
     'def f(items):\n    return max(items, key=lambda x: x["{f}"])'),
    ("min", "Return the smallest value.",
     'def f(items):\n    return min(items)'),
    ("max", "Return the largest value.",
     'def f(items):\n    return max(items)'),
    ("sum", "Return the total.",
     'def f(items):\n    return sum(items)'),
    ("count", "Return how many there are.",
     'def f(items):\n    return len(items)'),
    ("filter", "Keep only the positive entries.",
     'def f(items):\n    return [x for x in items if x > 0]'),
    ("map_mul", "Double every entry.",
     'def f(items):\n    return [x * 2 for x in items]'),
    ("map_field", "Project the {f} field of every record.",
     'def f(items):\n    return [x["{f}"] for x in items]'),
    ("filter_sort", "Keep the ok records and sort them by {f}.",
     'def f(items):\n    return sorted([x for x in items if x["ok"]], key=lambda x: x["{f}"])'),
    ("filter_argmax", "Among the ok records, return the one with the largest {f}.",
     'def f(items):\n    return max([x for x in items if x["ok"]], key=lambda x: x["{f}"])'),
    ("join", "Concatenate the two lists.",
     'def f(a, b):\n    return a + b'),
    ("dedup_like", "Keep the records with a non-empty {f}, sorted by it.",
     'def f(items):\n    return sorted([x for x in items if x["{f}"] > 0], key=lambda x: x["{f}"])'),
    # intentionally hard (loops / control flow): realistic HumanEval content
    ("loop_sum", "Sum the first n integers with a loop.",
     'def f(n):\n    t = 0\n    for i in range(n):\n        t += i\n    return t'),
    ("loop_str", "Repeat and join the string n times.",
     'def f(s, n):\n    out = ""\n    for _ in range(n):\n        out += s\n    return out'),
    ("if_else", "Clamp: negative to zero, keep the rest.",
     'def f(items):\n    out = []\n    for x in items:\n        if x < 0:\n            out.append(0)\n        else:\n            out.append(x)\n    return out'),
]


def gen_code(rng, i):
    kind, prompt_t, code_t = CODE_PATTERNS[i % len(CODE_PATTERNS)]
    f = rng.choice(CODE_FIELDS)
    return {"task_id": f"code-{i:05d}", "prompt": prompt_t.format(f=f),
            "code": code_t.format(f=f), "pattern": kind}


SQL_TABLES = [
    ("flights", ["origin", "destination", "price", "country"]),
    ("hotels", ["name", "city", "price", "stars"]),
    ("students", ["name", "age", "gpa", "dept"]),
    ("products", ["name", "price", "category", "stock"]),
    ("employees", ["name", "salary", "dept", "years"]),
]


def gen_sql(rng, i):
    table, cols = SQL_TABLES[rng.randrange(len(SQL_TABLES))]
    c1, c2 = rng.sample(cols, 2)
    num = rng.randint(10, 900)
    kind = rng.choice(["simple", "where", "order", "agg", "join", "having"])
    if kind == "simple":
        q = f"SELECT {c1}, {c2} FROM {table}"
        question = f"List the {c1} and {c2} of every row in {table}."
    elif kind == "where":
        q = f"SELECT {c1}, {c2} FROM {table} WHERE {c2} > {num}"
        question = f"List {c1} and {c2} from {table} where {c2} exceeds {num}."
    elif kind == "order":
        q = f"SELECT {c1}, {c2} FROM {table} ORDER BY {c2} DESC LIMIT {rng.randint(3, 10)}"
        question = f"Top rows of {table} by {c2}, highest first."
    elif kind == "agg":
        agg = rng.choice(["AVG", "SUM", "COUNT", "MAX", "MIN"])
        q = f"SELECT {c1}, {agg}({c2}) FROM {table} GROUP BY {c1}"
        question = f"For each {c1} in {table}, give me the {agg} of {c2}."
    elif kind == "having":
        q = f"SELECT {c1}, COUNT(*) FROM {table} GROUP BY {c1} HAVING COUNT(*) > {rng.randint(1, 5)}"
        question = f"Which {c1} groups in {table} have more than a few rows?"
    else:
        t2 = "airports" if table == "flights" else f"{table}_meta"
        q = f"SELECT a.{c1}, b.{c2} FROM {table} a JOIN {t2} b ON a.id = b.id"
        question = f"Join {table} with {t2} and show {c1} with {c2}."
    return {"task_id": f"sql-{i:05d}", "db_id": table, "question": question,
            "query": q, "shape": kind}


RTL_MODULES = [
    ("fifo_ctrl", ["fifo_count", "depth", "wr_en", "rd_en"],
     "FIFO overflow assertion fires in simulation"),
    ("alu_width", ["a", "b", "result", "carry"],
     "result width mismatch truncates the carry chain"),
    ("counter_glitch", ["clk", "rst_n", "count"],
     "glitch on the asynchronous reset path"),
    ("fsm_decode", ["state", "next_state", "valid"],
     "unreachable FSM state flagged by lint"),
    ("mem_arb", ["req", "gnt", "bank_sel"],
     "starvation on bank arbitration under backpressure"),
]


def gen_rtl(rng, i):
    mod, sigs, issue = RTL_MODULES[i % len(RTL_MODULES)]
    kind = rng.choice(["fix", "timing", "lint", "feature"])
    if kind == "timing":
        prompt = (f"In {mod}, {issue}; the setup slack is negative — "
                  f"close timing and produce the fix.")
        sample = {"timing": {"slack_ns": round(rng.uniform(-0.4, 0.1), 2)}}
    elif kind == "lint":
        prompt = f"Clean up {mod}: lint reports that {issue}."
        sample = {}
    elif kind == "feature":
        prompt = f"Extend {mod} with an enable pipeline stage; note that {issue}."
        sample = {}
    else:
        prompt = f"Fix {mod}: {issue}."
        sample = {}
    sample.update({
        "task_id": f"rtl-{i:05d}", "prompt": prompt,
        "rtl": f"module {mod}(input clk, input rst_n);\n  // ...\nendmodule",
        "signals": sigs, "kind": kind})
    return sample

# ------------------------------------------------------------- pipeline


def build_record(sample, dataset, seed):
    mod, reason, lifter = lift_sample(sample)
    if mod is None:
        return None, reason, None
    rep = validate(mod)
    if not rep.valid:
        return None, f"invalid:{rep.errors[0].code}", mod
    res = Simulator(mod, seed=seed, jitter=0).run()
    rec = {
        "source": {"dataset": dataset, "id": sample.get("task_id", ""),
                   "lifter": lifter},
        "input": {"text": (sample.get("instruction") or sample.get("prompt")
                           or sample.get("question") or "")},
        "raw_trace": {k: v for k, v in sample.items() if k != "task_id"},
        "taskir": module_to_dict(mod),
        "taskir_text": to_text(mod),
        "validation": {
            "passed": True,
            "errors": [],
            "warnings": [f"{w.code}@{w.node}" for w in rep.warnings],
        },
        "execution": {
            "trace": {
                "status": res.status,
                "events": [{"node": e.node, "op": e.op, "status": e.status,
                            "attempt": e.attempt, "latency_ms": e.latency_ms}
                           for e in res.events],
            },
            "cost": {
                "latency_sequential_ms": res.seq_latency_ms,
                "critical_path_ms": res.critical_path_ms,
                "tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
                "flops": res.flops, "energy_j": res.energy_j,
                "memory_peak_mb": res.peak_memory_mb,
                "lm_calls": res.lm_calls, "api_calls": res.api_calls,
                "retries": res.retries, "skipped": res.skipped,
            },
        },
    }
    return rec, None, mod


def run_source(name, samples, stats, records, metrics):
    from collections import Counter
    reasons = Counter()
    lifted = valid = 0
    for k, s in enumerate(samples):
        seed = f"corpus:{name}:{s.get('task_id', k)}"
        rec, reason, mod = build_record(s, name, seed)
        if rec is None:
            reasons[reason] += 1
            continue
        lifted += 1
        valid += 1
        records.append(rec)
        metrics.append(program_metrics(mod))
    stats[name] = {
        "generated": len(samples),
        "lifted_and_valid": valid,
        "coverage_pct": round(100.0 * valid / len(samples), 2) if samples else 0,
        "unsupported_reasons": dict(reasons.most_common()),
    }
    print(f"{name:10s} generated={len(samples)} valid={valid} "
          f"coverage={stats[name]['coverage_pct']}% reasons={dict(reasons)}")


def write_split(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-toolbench", type=int, default=4000)
    ap.add_argument("--n-code", type=int, default=3000)
    ap.add_argument("--n-sql", type=int, default=2000)
    ap.add_argument("--n-rtl", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--out", default=str(ROOT / "data" / "compiler_corpus"))
    ap.add_argument("--raw-dir", default=None,
                    help="dir with real toolbench/code/sql/rtl .jsonl files "
                         "(same schemas); skips synthetic generation")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    specs = {"toolbench": (args.n_toolbench, gen_toolbench),
             "code": (args.n_code, gen_code),
             "sql": (args.n_sql, gen_sql),
             "rtl": (args.n_rtl, gen_rtl)}

    records, metrics, stats = [], [], {}
    if args.raw_dir:
        raw = pathlib.Path(args.raw_dir)
        for name in specs:
            f = raw / f"{name}.jsonl"
            if not f.exists():
                continue
            samples = [json.loads(l) for l in
                       open(f, encoding="utf-8").read().splitlines() if l.strip()]
            run_source(name, samples, stats, records, metrics)
    else:
        for name, (n, gen) in specs.items():
            samples = [gen(rng, i) for i in range(1, n + 1)]
            run_source(name, samples, stats, records, metrics)

    # deterministic split
    idx = list(range(len(records)))
    random.Random(args.seed).shuffle(idx)
    n_val = int(len(records) * args.val_ratio)
    val_ids = set(idx[:n_val])
    train = [r for i, r in enumerate(records) if i not in val_ids]
    val = [r for i, r in enumerate(records) if i in val_ids]
    write_split(train, out / "train.jsonl")
    write_split(val, out / "val.jsonl")

    # schema exemplar: first record, verbatim
    (ROOT / "data" / "schema").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "schema" / "compiler_sample.json").write_text(
        json.dumps(records[0], indent=2, ensure_ascii=False), encoding="utf-8")

    total_gen = sum(s["generated"] for s in stats.values())
    total_valid = sum(s["lifted_and_valid"] for s in stats.values())
    stats["ALL"] = {
        "generated": total_gen,
        "lifted_and_valid": total_valid,
        "coverage_pct": round(100.0 * total_valid / total_gen, 2) if total_gen else 0,
        "train": len(train), "val": len(val),
    }
    stats["ir_graph_stats"] = aggregate(metrics, subset="corpus")
    (out / "stats.json").write_text(json.dumps(stats, indent=2,
                                               ensure_ascii=False),
                                    encoding="utf-8")

    print(f"\ncorpus: {len(train)} train / {len(val)} val -> {out}")
    print(f"overall coverage: {stats['ALL']['coverage_pct']}% "
          f"({total_valid}/{total_gen})")
    print(f"schema exemplar -> data/schema/compiler_sample.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
