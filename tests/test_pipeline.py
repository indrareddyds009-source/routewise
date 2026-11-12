import json

import pandas as pd
from fastapi.testclient import TestClient

from routewise import bench, report, router


def _run(sim_config, tmp_path, n=300):
    out = tmp_path / "run.jsonl"
    bench.main(["--config", sim_config, "--n", str(n), "--concurrency", "32", "--out", str(out)])
    return out


def test_bench_records_metrics(sim_config, tmp_path):
    out = _run(sim_config, tmp_path, n=60)
    df = pd.read_json(out, lines=True)
    assert len(df) == 180 and df["error"].isna().all()
    assert df["usage_reported"].all() and (df["ttft_s"] > 0).all() and (df["latency_s"] >= df["ttft_s"]).all()
    acc = df.groupby("backend")["correct"].mean()
    assert acc["large"] > acc["small"]
    s = report.summarize(df)
    assert list(s.index) == ["small", "medium", "large"]  # sorted by cost


def test_unreachable_backend_is_recorded_not_raised(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text('backends:\n  - {name: dead, base_url: "http://127.0.0.1:9/v1", model: x}\n')
    out = tmp_path / "bad.jsonl"
    bench.main(["--config", str(cfg), "--n", "5", "--out", str(out)])
    df = pd.read_json(out, lines=True)
    assert df["error"].notna().all() and not df["correct"].any()


def test_router_beats_naive_policies(sim_config, tmp_path):
    out = _run(sim_config, tmp_path)
    r, s = router.train_and_evaluate(str(out))
    b = s["baselines"]
    assert s["router"]["cost_per_1k"] < b["always large"]["cost_per_1k"]
    assert s["router"]["accuracy"] > b["always medium"]["accuracy"]
    assert b["oracle"]["accuracy"] >= b["always large"]["accuracy"]
    assert r.route("What is 12 + 30?\nRespond with only the final answer, no explanation.") in r.order
    path = tmp_path / "r.joblib"
    r.save(str(path))
    assert router.Router.load(str(path)).threshold == r.threshold
    json.dumps(s)  # serializable


def test_gateway_routes_and_reports_backend(sim_config, tmp_path, monkeypatch):
    out = _run(sim_config, tmp_path, n=200)
    r, _ = router.train_and_evaluate(str(out))
    r.save(str(tmp_path / "r.joblib"))
    monkeypatch.setenv("ROUTEWISE_CONFIG", sim_config)
    monkeypatch.setenv("ROUTEWISE_ROUTER", str(tmp_path / "r.joblib"))
    from routewise import gateway
    gateway._state.cache_clear()
    c = TestClient(gateway.app)
    resp = c.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200 and resp.headers["x-routewise-backend"] in r.order
    pinned = c.post("/v1/chat/completions", json={"model": "small", "messages": [{"role": "user", "content": "hi"}]})
    assert pinned.headers["x-routewise-backend"] == "small"
