"""Learn a cost-aware router from benchmark results and evaluate it by offline replay.

For every backend except the most expensive, a classifier predicts P(backend answers this
prompt correctly) from the prompt alone. At serving time the router sends the prompt to the
cheapest backend whose predicted success clears a threshold τ, and otherwise escalates to the
most expensive one. Sweeping τ traces the router's accuracy/cost curve, which we compare
against always using one backend and against an oracle that knows which backend will succeed.

    python -m routewise.router train results/run.jsonl --out results/router.joblib
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .report import pareto_plot


def prompt_features(prompts: pd.Series) -> pd.DataFrame:
    nums = prompts.apply(lambda p: [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", p)
                                    if x.replace(",", "").replace(".", "").isdigit()])
    return pd.DataFrame({
        "text": prompts,
        "n_chars": prompts.str.len(),
        "n_numbers": nums.apply(len),
        "log_max_number": nums.apply(lambda v: np.log10(max(v) + 1) if v else 0.0),
        "n_sentences": prompts.str.count(r"[.?!]"),
    })


def make_model():
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000), "text"),
        ("num", StandardScaler(), ["n_chars", "n_numbers", "log_max_number", "n_sentences"]),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"))


@dataclass
class Router:
    order: list[str]                  # backends, cheapest first
    models: dict                      # backend -> fitted pipeline (all but the last)
    threshold: float

    def probabilities(self, prompts: list[str]) -> pd.DataFrame:
        X = prompt_features(pd.Series(prompts))
        return pd.DataFrame({b: m.predict_proba(X)[:, 1] for b, m in self.models.items()})

    def choose(self, probs: pd.DataFrame, threshold: float | None = None) -> list[str]:
        tau = self.threshold if threshold is None else threshold
        out = []
        for _, row in probs.iterrows():
            out.append(next((b for b in self.order[:-1] if row[b] >= tau), self.order[-1]))
        return out

    def route(self, prompt: str) -> str:
        return self.choose(self.probabilities([prompt]))[0]

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> Router:
        return joblib.load(path)


def train_and_evaluate(results_path: str, tasks_path: str | None = None, test_size: float = 0.3,
                       seed: int = 0, target_ratio: float = 0.98) -> tuple[Router, dict]:
    df = pd.read_json(results_path, lines=True)
    tasks_path = tasks_path or str(Path(results_path).with_suffix(".tasks.jsonl"))
    prompts = pd.read_json(tasks_path, lines=True).set_index("id")["prompt"]
    wide_ok = df.pivot(index="task_id", columns="backend", values="correct").astype(int)
    wide_cost = df.pivot(index="task_id", columns="backend", values="cost_usd")
    order = list(wide_cost.mean().sort_values().index)
    wide_ok, wide_cost = wide_ok[order], wide_cost[order]
    X_all = prompt_features(prompts.loc[wide_ok.index].reset_index(drop=True))

    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                  .split(X_all, groups=wide_ok.index))
    models, aucs = {}, {}
    for b in order[:-1]:
        y = wide_ok[b].to_numpy()
        m = make_model()
        if len(set(y[tr])) < 2:            # backend always right/wrong on train: constant model
            m = _Constant(float(y[tr].mean()))
        else:
            m.fit(X_all.iloc[tr], y[tr])
        models[b] = m
        if len(set(y[te])) == 2:
            aucs[b] = round(roc_auc_score(y[te], m.predict_proba(X_all.iloc[te])[:, 1]), 3)

    router = Router(order, models, 0.5)
    ok_te, cost_te = wide_ok.iloc[te], wide_cost.iloc[te]
    probs = router.probabilities(list(X_all.iloc[te]["text"]))

    def replay(choice: list[str]) -> tuple[float, float]:
        acc = np.mean([ok_te.iloc[i][c] for i, c in enumerate(choice)])
        cost = np.mean([cost_te.iloc[i][c] for i, c in enumerate(choice)]) * 1000
        return float(acc), float(cost)

    baselines = {f"always {b}": replay([b] * len(te)) for b in order}
    oracle_choice = [next((b for b in order if ok_te.iloc[i][b]), order[-1]) for i in range(len(te))]
    baselines["oracle"] = replay(oracle_choice)

    curve = []
    for tau in np.round(np.linspace(0.05, 0.95, 19), 2):
        acc, cost = replay(router.choose(probs, tau))
        curve.append({"threshold": float(tau), "accuracy": acc, "cost_per_1k": cost})

    best_single = max((v for k, v in baselines.items() if k.startswith("always")), key=lambda v: v[0])
    eligible = [c for c in curve if c["accuracy"] >= target_ratio * best_single[0]]
    pick = min(eligible, key=lambda c: c["cost_per_1k"]) if eligible else max(curve, key=lambda c: c["accuracy"])
    router.threshold = pick["threshold"]
    summary = {
        "backends_cheapest_first": order, "train_tasks": len(tr), "test_tasks": len(te), "classifier_auc": aucs,
        "baselines": {k: {"accuracy": v[0], "cost_per_1k": v[1]} for k, v in baselines.items()},
        "router": pick, "target": f">= {target_ratio:.0%} of best single-backend accuracy",
        "cost_saving_vs_best_single": (1 - pick["cost_per_1k"] / best_single[1]) if best_single[1] else None,
        "curve": curve,
    }
    return router, summary


class _Constant:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("results")
    t.add_argument("--tasks")
    t.add_argument("--out", default="results/router.joblib")
    t.add_argument("--target", type=float, default=0.98)
    a = ap.parse_args(argv)
    router, s = train_and_evaluate(a.results, a.tasks, target_ratio=a.target)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    router.save(str(out))
    (out.parent / "router_eval.json").write_text(json.dumps(s, indent=2))
    pts = {k: (v["cost_per_1k"], v["accuracy"]) for k, v in s["baselines"].items()}
    pts[f"router (τ={s['router']['threshold']})"] = (s["router"]["cost_per_1k"], s["router"]["accuracy"])
    pareto_plot(pts, str(out.parent / "router_pareto.png"), "Router vs single backends (held-out tasks)")
    lines = ["| Policy | Accuracy | Cost / 1k req |", "|---|---|---|"]
    lines += [f"| {k} | {v[1]:.1%} | ${v[0]:.4f} |" for k, v in pts.items()]
    (out.parent / "router_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nclassifier AUC: {s['classifier_auc']}")
    if s["cost_saving_vs_best_single"] is not None:
        print(f"router saves {s['cost_saving_vs_best_single']:.1%} vs the most accurate single backend "
              f"({s['target']})")


if __name__ == "__main__":
    main()
