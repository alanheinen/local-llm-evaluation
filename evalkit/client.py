"""Talk to any OpenAI-compatible endpoint.

Deliberately minimal and deliberately not tied to a vendor SDK. Anything that
serves /v1/chat/completions works: vLLM, llama.cpp, LM Studio, Ollama's compat
endpoint, a hosted API, or a proxy in front of several of them.

Configuration is two environment variables:

    EVALKIT_BASE_URL   e.g. http://localhost:8000/v1
    EVALKIT_API_KEY    optional; many local servers ignore it

TIMING IS MEASURED HERE AND ONLY HERE, because where you measure it decides
what the number means. We time the request itself and take completion_tokens
from the server's own usage block. Never derive a token count from max_tokens:
a run that stopped early would report tokens it never produced, and the
resulting tok/s is fiction.
"""
from __future__ import annotations

import os
import time

import requests


class EndpointError(RuntimeError):
    pass


class Client:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: int = 900):
        self.base = (base_url or os.environ.get("EVALKIT_BASE_URL") or "").rstrip("/")
        if not self.base:
            raise EndpointError(
                "set EVALKIT_BASE_URL (e.g. http://localhost:8000/v1)")
        self.key = api_key or os.environ.get("EVALKIT_API_KEY") or "not-needed"
        self.timeout = timeout

    def models(self) -> list[str]:
        r = requests.get(f"{self.base}/models",
                         headers={"Authorization": f"Bearer {self.key}"},
                         timeout=60)
        r.raise_for_status()
        return [m["id"] for m in (r.json().get("data") or [])]

    def complete(self, model: str, prompt: str, max_tokens: int = 4096,
                 temperature: float = 0.0) -> dict:
        """One turn. Returns text, timing and the server's own token counts."""
        started = time.monotonic()
        r = requests.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens,
                  "temperature": temperature},
            timeout=self.timeout,
        )
        elapsed = time.monotonic() - started
        r.raise_for_status()
        body = r.json()
        choice = (body.get("choices") or [{}])[0]
        usage = body.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        return {
            "text": (choice.get("message") or {}).get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "seconds": round(elapsed, 2),
            "completion_tokens": completion_tokens,
            # None rather than 0 when the server withheld a count: a missing
            # measurement must not average in as a slow one.
            "tok_s": (round(completion_tokens / elapsed, 2)
                      if completion_tokens and elapsed > 0 else None),
        }
