"""Procedurally generated, automatically gradable tasks at three difficulty levels.

Generating tasks (instead of downloading a benchmark) means the suite has no license
or contamination issues, every answer is exactly known, and difficulty is controlled,
which is what a router needs to learn from. You can also load your own JSONL.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import asdict, dataclass

SUFFIX = "\nRespond with only the final answer, no explanation."


@dataclass
class Task:
    id: str
    type: str          # arithmetic | word_problem | extraction | date | counting
    difficulty: str    # easy | medium | hard
    prompt: str
    answer: str        # canonical answer (number, date, word, or JSON)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _arith(r: random.Random, i: int) -> Task:
    level = r.choice(["easy", "medium", "hard"])
    if level == "easy":
        a, b = r.randint(10, 99), r.randint(10, 99)
        return Task(f"ar-{i}", "arithmetic", level, f"What is {a} + {b}?{SUFFIX}", str(a + b))
    if level == "medium":
        a, b = r.randint(12, 99), r.randint(12, 99)
        return Task(f"ar-{i}", "arithmetic", level, f"What is {a} * {b}?{SUFFIX}", str(a * b))
    a, b, c, d = r.randint(100, 999), r.randint(11, 99), r.randint(100, 999), r.randint(2, 9)
    return Task(f"ar-{i}", "arithmetic", level, f"What is ({a} * {b}) - ({c} * {d})?{SUFFIX}", str(a * b - c * d))


NAMES = ["Maya", "Arjun", "Leo", "Sofia", "Wei", "Amara", "Diego", "Hana"]


def _word(r: random.Random, i: int) -> Task:
    n = r.choice(NAMES)
    level = r.choice(["easy", "medium", "hard"])
    p1, q1 = r.randint(2, 9), r.randint(2, 8)
    if level == "easy":
        return Task(f"wp-{i}", "word_problem", level,
                    f"{n} buys {q1} notebooks at ${p1} each. How many dollars does {n} spend?{SUFFIX}", str(p1 * q1))
    p2, q2 = r.randint(3, 12), r.randint(2, 6)
    total = p1 * q1 + p2 * q2
    paid = (total // 10 + r.randint(1, 3)) * 10
    if level == "medium":
        return Task(f"wp-{i}", "word_problem", level,
                    f"{n} buys {q1} pens at ${p1} each and {q2} books at ${p2} each, paying with ${paid}. "
                    f"How many dollars of change does {n} get?{SUFFIX}", str(paid - total))
    disc = r.choice([10, 20, 25])
    weeks = r.randint(3, 6)
    per_week = total * (100 - disc) / 100
    ans = per_week * weeks
    ans_s = str(int(ans)) if ans == int(ans) else f"{ans:.2f}"
    return Task(f"wp-{i}", "word_problem", level,
                f"Every week {n} buys {q1} pens at ${p1} each and {q2} books at ${p2} each. A coupon takes "
                f"{disc}% off each weekly purchase. How many dollars does {n} spend over {weeks} weeks?{SUFFIX}", ans_s)


VENDORS = ["Acme Corp", "Globex LLC", "Initech", "Umbrella Health", "Stark Supplies", "Wayne Logistics"]


def _extract(r: random.Random, i: int) -> Task:
    v = r.choice(VENDORS)
    amt = round(r.uniform(50, 9999), 2)
    d = dt.date(2026, 1, 1) + dt.timedelta(days=r.randint(0, 300))
    level = r.choice(["easy", "medium", "hard"])
    if level == "easy":
        text = f"Invoice from {v} dated {d.isoformat()} for ${amt:,.2f}."
    elif level == "medium":
        text = (f"Hi team, please process the attached bill. The vendor is {v}; total due is ${amt:,.2f}, "
                f"issued on {d.strftime('%B %d, %Y')}. Thanks!")
    else:
        decoy = round(amt * r.uniform(0.1, 0.5), 2)
        text = (f"Re: payment. {v} sent a revised invoice on {d.strftime('%d %b %Y')}. The earlier deposit of "
                f"${decoy:,.2f} was already paid, so ignore it; the amount now due is ${amt:,.2f}.")
    prompt = (f"Extract the vendor, amount due (number), and invoice date (YYYY-MM-DD) as JSON with keys "
              f"vendor, amount, date.\nText: {text}\nRespond with only the JSON.")
    return Task(f"ex-{i}", "extraction", level, prompt,
                json.dumps({"vendor": v, "amount": amt, "date": d.isoformat()}))


def _date(r: random.Random, i: int) -> Task:
    d = dt.date(2026, 1, 1) + dt.timedelta(days=r.randint(0, 364))
    level = r.choice(["easy", "medium", "hard"])
    if level == "easy":
        k = r.randint(1, 9)
        return Task(f"dt-{i}", "date", level, f"What date is {k} days after {d.isoformat()}? Use YYYY-MM-DD.{SUFFIX}",
                    (d + dt.timedelta(days=k)).isoformat())
    if level == "medium":
        k = r.randint(20, 90)
        return Task(f"dt-{i}", "date", level, f"What date is {k} days after {d.isoformat()}? Use YYYY-MM-DD.{SUFFIX}",
                    (d + dt.timedelta(days=k)).isoformat())
    return Task(f"dt-{i}", "date", level, f"What day of the week is {d.isoformat()}?{SUFFIX}", d.strftime("%A"))


WORDS = ["strawberry", "bookkeeper", "mississippi", "committee", "parallel", "assessment", "balloon", "occurrence"]


def _count(r: random.Random, i: int) -> Task:
    level = r.choice(["easy", "medium", "hard"])
    k = {"easy": 1, "medium": 2, "hard": 4}[level]
    words = r.sample(WORDS, k)
    phrase = " ".join(words)
    letter = r.choice(sorted(set("".join(words))))
    return Task(f"ct-{i}", "counting", level,
                f"How many times does the letter '{letter}' appear in \"{phrase}\"?{SUFFIX}", str(phrase.count(letter)))


GENERATORS = [_arith, _word, _extract, _date, _count]


def generate(n: int = 200, seed: int = 42) -> list[Task]:
    r = random.Random(seed)
    return [GENERATORS[i % len(GENERATORS)](r, i) for i in range(n)]


def load_jsonl(path: str) -> list[Task]:
    with open(path) as f:
        return [Task(**json.loads(line)) for line in f if line.strip()]
