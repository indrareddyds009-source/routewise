"""Backend config + an async streaming client for any OpenAI-compatible endpoint
(vLLM, Ollama, llama.cpp server, TGI, OpenAI, Together, Groq...)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import httpx
import yaml


@dataclass
class Backend:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    # API pricing (USD per 1M tokens) ...
    price_in_per_m: float | None = None
    price_out_per_m: float | None = None
    # ... or self-hosted pricing (USD per GPU-hour); cost is then derived from measured throughput.
    gpu_hourly_usd: float | None = None
    max_tokens: int = 256
    extra: dict = field(default_factory=dict)

    @property
    def headers(self) -> dict:
        key = os.getenv(self.api_key_env, "") if self.api_key_env else ""
        return {"Authorization": f"Bearer {key}"} if key else {}


def load_backends(path: str) -> list[Backend]:
    with open(path) as f:
        return [Backend(**b) for b in yaml.safe_load(f)["backends"]]


async def complete(client: httpx.AsyncClient, b: Backend, prompt: str, timeout: float = 120) -> dict:
    """Stream one chat completion; measure time-to-first-token, total latency and token counts."""
    body = {"model": b.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": b.max_tokens, "stream": True, "stream_options": {"include_usage": True}, **b.extra}
    t0 = time.perf_counter()
    ttft, parts, usage, err = None, [], None, None
    try:
        async with client.stream("POST", f"{b.base_url.rstrip('/')}/chat/completions", json=body,
                                 headers=b.headers, timeout=timeout) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        parts.append(piece)
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        err = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0
    text = "".join(parts)
    # Fall back to a ~4 chars/token estimate if the server does not report usage.
    in_tok = (usage or {}).get("prompt_tokens") or max(1, len(prompt) // 4)
    out_tok = (usage or {}).get("completion_tokens") or max(1, len(text) // 4)
    return {"text": text, "ttft_s": ttft, "latency_s": latency, "prompt_tokens": in_tok,
            "completion_tokens": out_tok, "usage_reported": usage is not None, "error": err}
