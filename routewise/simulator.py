"""A simulated OpenAI-compatible server with three fake models of different size, speed and price.

Used ONLY for tests, CI and a no-GPU demo of the pipeline. Its numbers are synthetic by
construction; real results come from running `bench` against real backends (see README).

    python -m routewise.simulator --port 8900
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from .tasks import generate

PROFILES = {  # P(correct) by difficulty, seconds to first token, tokens/second
    "sim-small": {"p": {"easy": 0.95, "medium": 0.55, "hard": 0.15}, "ttft": 0.02, "tps": 400},
    "sim-medium": {"p": {"easy": 0.97, "medium": 0.85, "hard": 0.50}, "ttft": 0.04, "tps": 200},
    "sim-large": {"p": {"easy": 0.99, "medium": 0.96, "hard": 0.88}, "ttft": 0.08, "tps": 90},
}
# a few task types are systematically hard for small models (e.g. letter counting)
TYPE_PENALTY = {"sim-small": {"counting": 0.35, "date": 0.2}, "sim-medium": {"counting": 0.15}}


def build_app(n: int = 400, seed: int = 42, speed: float = 1.0) -> FastAPI:
    truth = {t.prompt: t for t in generate(n, seed)}
    app = FastAPI(title="RouteWise simulator")

    def answer(model: str, prompt: str) -> str:
        t = truth.get(prompt)
        prof = PROFILES.get(model)
        if t is None or prof is None:
            return "I'm not sure."
        p = prof["p"][t.difficulty] - TYPE_PENALTY.get(model, {}).get(t.type, 0)
        u = int(hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if u < p:
            return t.answer
        if t.type == "extraction":
            d = json.loads(t.answer)
            d["amount"] = round(d["amount"] * 1.1, 2)
            return json.dumps(d)
        if t.type == "date":
            return "2026-01-01" if t.difficulty != "hard" else "Sunday" if t.answer != "Sunday" else "Monday"
        return str(int(float(t.answer)) + 1 + int(u * 10))

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        model = body.get("model", "")
        prompt = body["messages"][-1]["content"]
        text = answer(model, prompt)
        prof = PROFILES.get(model, PROFILES["sim-medium"])
        pt, ct = max(1, len(prompt) // 4), max(1, len(text) // 4)
        usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        if not body.get("stream"):
            await asyncio.sleep((prof["ttft"] + ct / prof["tps"]) * speed)
            return JSONResponse({"id": "sim", "object": "chat.completion", "model": model, "usage": usage,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                              "finish_reason": "stop"}]})

        async def gen():
            await asyncio.sleep(prof["ttft"] * speed)
            pieces = [text[i:i + 4] for i in range(0, len(text), 4)] or [""]
            for piece in pieces:
                yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": piece}}]}) + "\n\n"
                await asyncio.sleep(speed / prof["tps"])
            yield "data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/health")
    def health():
        return {"status": "ok", "time": time.time()}

    return app


app = build_app()

if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    uvicorn.run(app, host="127.0.0.1", port=ap.parse_args().port, log_level="warning")
