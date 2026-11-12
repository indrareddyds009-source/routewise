"""Run every task against every backend at a fixed concurrency and record per-request results.

    python -m routewise.bench --config configs/ollama.yaml --n 200 --concurrency 4 --out results/run.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from .backends import Backend, complete, load_backends
from .grading import grade
from .tasks import Task, generate, load_jsonl


async def bench_backend(b: Backend, tasks: list[Task], concurrency: int, client: httpx.AsyncClient) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(t: Task):
        async with sem:
            r = await complete(client, b, t.prompt)
        return {"backend": b.name, "task_id": t.id, "type": t.type, "difficulty": t.difficulty,
                "correct": (r["error"] is None) and grade(t, r["text"]), **r}

    wall0 = time.perf_counter()
    rows = await asyncio.gather(*(one(t) for t in tasks))
    wall = time.perf_counter() - wall0
    for row in rows:
        row["cost_usd"] = request_cost(b, row, wall, len(rows))
        row["concurrency"] = concurrency
        row["wall_s"] = wall
    return rows


def request_cost(b: Backend, row: dict, wall_s: float, n: int) -> float:
    if b.price_in_per_m is not None:
        return (row["prompt_tokens"] * b.price_in_per_m + row["completion_tokens"] * (b.price_out_per_m or 0)) / 1e6
    if b.gpu_hourly_usd is not None:
        # amortize GPU time over the requests served during the run at this concurrency
        return b.gpu_hourly_usd / 3600 * wall_s / n
    return 0.0


async def run(backends: list[Backend], tasks: list[Task], concurrency: int) -> list[dict]:
    rows: list[dict] = []
    async with httpx.AsyncClient() as client:
        for b in backends:
            print(f"→ {b.name}: {len(tasks)} tasks @ concurrency {concurrency}", flush=True)
            res = await bench_backend(b, tasks, concurrency, client)
            acc = sum(r["correct"] for r in res) / len(res)
            errs = sum(r["error"] is not None for r in res)
            print(f"  accuracy {acc:.1%}, errors {errs}, wall {res[0]['wall_s']:.1f}s", flush=True)
            rows += res
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", help="JSONL of tasks; default = generated suite")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="results/run.jsonl")
    a = ap.parse_args(argv)
    tasks = load_jsonl(a.tasks) if a.tasks else generate(a.n, a.seed)
    rows = asyncio.run(run(load_backends(a.config), tasks, a.concurrency))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    Path(a.out).with_suffix(".tasks.jsonl").write_text("\n".join(t.to_json() for t in tasks) + "\n")
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
