"""Summarize a benchmark run: accuracy by task type and difficulty, latency percentiles,
throughput, cost per 1k requests, and an accuracy-vs-cost Pareto chart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("backend")
    s = pd.DataFrame({
        "accuracy": g["correct"].mean(),
        "p50_latency_s": g["latency_s"].median(),
        "p95_latency_s": g["latency_s"].quantile(0.95),
        "p50_ttft_s": g["ttft_s"].median(),
        "out_tok_per_s": g.apply(lambda x: (x["completion_tokens"] / (x["latency_s"] - x["ttft_s"].fillna(0))
                                            .clip(lower=1e-3)).median(), include_groups=False),
        "req_per_s": g.apply(lambda x: len(x) / x["wall_s"].iloc[0], include_groups=False),
        "cost_per_1k_usd": g["cost_usd"].mean() * 1000,
        "errors": g["error"].apply(lambda e: e.notna().sum()),
    })
    return s.sort_values("cost_per_1k_usd")


def pareto_plot(points: dict[str, tuple[float, float]], path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for name, (cost, acc) in points.items():
        ax.scatter(cost, acc * 100, s=60)
        ax.annotate(name, (cost, acc * 100), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Cost per 1,000 requests (USD)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args(argv)
    df = load(a.results)
    s = summarize(df)
    by = df.pivot_table(index="backend", columns="difficulty", values="correct", aggfunc="mean")
    by_type = df.pivot_table(index="backend", columns="type", values="correct", aggfunc="mean")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pareto_plot({b: (r.cost_per_1k_usd, r.accuracy) for b, r in s.iterrows()}, str(out / "pareto.png"),
                "Accuracy vs cost by backend")
    md = ["# Benchmark report", "", (f"Requests: {len(df)} · Backends: {df.backend.nunique()} · "
          f"Concurrency: {df.concurrency.iloc[0]}"), "", "## Summary", "", s.round(4).to_markdown(), "",
          "## Accuracy by difficulty", "", by.round(3).to_markdown(), "", "## Accuracy by task type", "",
          by_type.round(3).to_markdown(), "", "![pareto](pareto.png)", ""]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    (out / "summary.json").write_text(json.dumps(s.reset_index().to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
