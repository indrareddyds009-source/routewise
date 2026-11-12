"""Deterministic graders: no LLM-as-judge, so accuracy numbers are exact and reproducible."""
from __future__ import annotations

import json
import re

from .tasks import Task

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str) -> float | None:
    found = NUM.findall(text.replace("$", ""))
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def _first_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def grade(task: Task, output: str) -> bool:
    out = (output or "").strip()
    if task.type in ("arithmetic", "word_problem", "counting"):
        got = _last_number(out)
        return got is not None and abs(got - float(task.answer)) < 1e-6
    if task.type == "date":
        return task.answer.lower() in out.lower()
    if task.type == "extraction":
        got, exp = _first_json(out), json.loads(task.answer)
        if not got:
            return False
        try:
            amount_ok = abs(float(str(got.get("amount")).replace(",", "").replace("$", "")) - exp["amount"]) < 0.01
        except (TypeError, ValueError):
            amount_ok = False
        return (str(got.get("vendor", "")).strip().lower() == exp["vendor"].lower() and amount_ok
                and str(got.get("date", "")).strip() == exp["date"])
    raise ValueError(task.type)
