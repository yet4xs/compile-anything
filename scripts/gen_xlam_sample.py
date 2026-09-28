"""Generate an xLAM-schema sample dataset (data/raw/xlam_sample.jsonl).

Records follow the xLAM function-calling convention:
  {"task_id", "question", "steps": [ "...```json [{name, arguments}]``` ..." ],
   "answer"}

IMPORTANT provenance note: these are SYNTHETIC records written in the xLAM
record/schema style so the lifter pipeline can be exercised end-to-end
offline. They are NOT samples of the Salesforce/xLAM dataset itself.
To use the real corpus, download it from HuggingFace and point
run_xlam_pipeline.py at the file (see data/raw/README.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CITIES = ["SFO", "JFK", "LAX", "ORD", "SEA", "BOS", "Austin", "Denver", "Miami"]
INTL = ["Tokyo", "Paris", "London", "Berlin", "Singapore", "Sydney"]
DATES = ["2026-10-05", "2026-10-12", "2026-11-01", "2026-11-20", "2026-12-03"]
CURR = ["JPY", "EUR", "GBP", "CNY", "AUD"]
SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG"]
TOPICS = ["quantum computing", "AI compilers", "electric aviation",
          "fusion energy", "edge AI", "robotics"]
LANGS = ["Japanese", "French", "German", "Spanish", "Chinese"]
EVENTS = ["team offsite", "product review", "investor meeting", "workshop"]
NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey"]

# (tool chain, question template, arg builders)
CHAINS = [
    ((("FlightSearch.search",
       {"origin": "{c1}", "destination": "{c2}", "date": "{d1}"}),
      ("CurrencyConverter.convert",
       {"amount": "{price}", "from_currency": "USD", "to_currency": "{cur}"})),
     "Find flights from {c1} to {c2} on {d1} and tell me the cheapest price in {cur}.",
     None),
    ((("WeatherAPI.forecast", {"city": "{c1}", "date": "{d1}"}),
      ("CalendarService.create_event",
       {"title": "{ev}", "date": "{d1}", "time": "10:00"})),
     "Check the weather in {c1} on {d1}, then schedule my {ev} that morning.",
     None),
    ((("NewsAPI.search", {"query": "{t}", "category": "technology"}),
      ("TextSummarizer.summarize", {}),
      ("EmailClient.send",
       {"to": "{name}@example.com", "subject": "Briefing: {t}"})),
     "Get the latest news on {t}, summarize it and email the summary to {name}.",
     None),
    ((("StockQuote.get_price", {"symbol": "{s}"}),
      ("Calculator.evaluate", {"expression": "{s} price * 1.05"}),
      ("EmailClient.send", {"to": "{name}@example.com", "subject": "{s} update"})),
     "Look up the current {s} price, project it 5% higher and send the number to {name}.",
     None),
    ((("MapsService.get_directions", {"origin": "{c1}", "destination": "{c2}"}),
      ("Translator.translate", {"text": "directions summary", "target_lang": "{lang}"})),
     "Find directions from {c1} to {c2} and translate the summary into {lang}.",
     None),
    ((("HotelDB.find", {"location": "{c2}", "check_in": "{d1}", "check_out": "{d2}"}),
      ("CurrencyConverter.convert",
       {"amount": "{price}", "from_currency": "USD", "to_currency": "{cur}"})),
     "Find hotels in {c2} from {d1} to {d2} and convert the nightly rate to {cur}.",
     None),
    ((("WebSearch.query", {"query": "{t} tutorial"}),
      ("TextSummarizer.summarize", {})),
     "Search for a {t} tutorial and summarize the key steps for me.",
     None),
    ((("FlightSearch.search",
       {"origin": "{c1}", "destination": "{c2}", "date": "{d1}"}),
      ("FlightBook.book", {"flight_id": "{fid}", "passenger": "{name}"})),
     "Find a flight from {c1} to {c2} on {d1} and book the best option for {name}.",
     None),
    ((("Translator.translate", {"text": "project update", "target_lang": "{lang}"}),
      ("EmailClient.send", {"to": "{name}@example.com", "subject": "Update ({lang})"})),
     "Translate my project update into {lang} and send it to {name}.",
     None),
    ((("NewsAPI.search", {"query": "{t}", "category": "science"}),
      ("TextSummarizer.summarize", {}),
      ("CalendarService.create_event",
       {"title": "read {t} digest", "date": "{d1}", "time": "18:00"})),
     "Find recent {t} articles, summarize them, and block evening time on {d1} to read it.",
     None),
]

LEADINS = ["Let me start.", "First, I'll look that up.", "Working on it.",
           "Sure.", ""]
MIDS = ["Now using that result.", "Next step.", "Based on the above,", "Then,"]
FINALS = ["Done.", "All set.", "Here you go.", "That completes the task."]


def _fmt_args(args, env):
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
            out[k] = env.get(v[1:-1], v)
        else:
            out[k] = v
    return out


def make_record(rng: random.Random, idx: int) -> dict:
    chain, qtpl, _ = CHAINS[idx % len(CHAINS)]
    env = {
        "c1": rng.choice(CITIES), "c2": rng.choice(INTL),
        "d1": rng.choice(DATES), "d2": rng.choice(DATES),
        "cur": rng.choice(CURR), "s": rng.choice(SYMBOLS),
        "t": rng.choice(TOPICS), "lang": rng.choice(LANGS),
        "ev": rng.choice(EVENTS), "name": rng.choice(NAMES),
        "price": rng.choice([320, 450, 540, 680, 810]),
        "fid": f"FL{rng.randint(100, 999)}",
    }
    question = qtpl.format(**{k: str(v) for k, v in env.items()})
    steps, tools = [], []
    for i, (name, args) in enumerate(chain):
        tools.append(name)
        block = json.dumps([{"name": name, "arguments": _fmt_args(args, env)}],
                           ensure_ascii=False)
        lead = rng.choice(LEADINS) if i == 0 else rng.choice(MIDS)
        steps.append(f"{lead} ```json\n{block}\n```")
    answer = (f"{rng.choice(FINALS)} (result synthesized from "
              f"{len(chain)} tool call{'s' if len(chain) > 1 else ''}).")
    return {"task_id": f"xlam-sample-{idx:04d}", "question": question,
            "steps": steps, "answer": answer}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / "xlam_sample.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i in range(1, args.n + 1):
            f.write(json.dumps(make_record(rng, i), ensure_ascii=False) + "\n")
    print(f"xLAM-style sample: {args.n} records -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
