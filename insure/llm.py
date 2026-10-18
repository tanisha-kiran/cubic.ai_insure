"""Local LLM access. Ollama only (localhost). No hosted APIs are ever called.

`NullLLM` makes every tool fall back to its deterministic rule path, which is what the
unit tests and the offline baseline evaluation use. `OllamaClient` is used on a machine
with Ollama running (default model: the *base* Qwen2.5-VL-7B, chosen over the
fine-tuned adapter because the cubic.ai evaluation found the base model refuses
unanswerable questions far more often: 49 % vs 17 %).
"""
from __future__ import annotations
import json
import re

LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


class NullLLM:
    name = "none"

    def generate(self, prompt: str) -> str | None:
        return None


class OllamaClient:
    def __init__(self, model: str = "qwen2.5vl:7b", host: str = "http://127.0.0.1:11434",
                 timeout: float = 90.0, keep_alive: str = "60m"):
        if not any(h in host for h in LOCAL_HOSTS):
            raise ValueError("OllamaClient only talks to localhost (privacy guarantee)")
        self.model, self.host, self.timeout, self.keep_alive = model, host.rstrip("/"), timeout, keep_alive
        self.name = f"ollama:{model}"

    def generate(self, prompt: str) -> str | None:
        import requests
        try:
            r = requests.post(f"{self.host}/api/generate", timeout=self.timeout, json={
                "model": self.model, "prompt": prompt, "stream": False, "keep_alive": self.keep_alive,
                "options": {"temperature": 0.0, "num_predict": 400}})
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception:
            return None


def extract_json(text: str | None) -> dict | None:
    """Pull the first JSON object out of a model reply; None if absent or invalid."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
