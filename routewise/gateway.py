"""OpenAI-compatible gateway that routes each request with a trained router.

    ROUTEWISE_CONFIG=configs/ollama.yaml ROUTEWISE_ROUTER=results/router.joblib \
        uvicorn routewise.gateway:app --port 9000
Then point any OpenAI client at http://localhost:9000/v1 with model="auto".
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .backends import load_backends
from .router import Router

app = FastAPI(title="RouteWise gateway")


@lru_cache
def _state():
    backends = {b.name: b for b in load_backends(os.environ["ROUTEWISE_CONFIG"])}
    return backends, Router.load(os.environ["ROUTEWISE_ROUTER"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat(body: dict):
    backends, router = _state()
    if body.get("stream"):
        raise HTTPException(400, "streaming is not supported by the gateway yet")
    prompt = "\n".join(m.get("content", "") for m in body.get("messages", []) if m.get("role") == "user")
    name = body["model"] if body.get("model") in backends else router.route(prompt)
    b = backends[name]
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{b.base_url.rstrip('/')}/chat/completions", headers=b.headers, timeout=120,
                         json={**body, "model": b.model, "stream": False})
    resp = JSONResponse(r.json(), status_code=r.status_code)
    resp.headers["x-routewise-backend"] = name
    resp.headers["x-routewise-latency-ms"] = f"{(time.perf_counter() - t0) * 1000:.0f}"
    return resp
