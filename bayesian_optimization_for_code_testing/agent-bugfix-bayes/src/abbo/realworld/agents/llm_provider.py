"""LLM provider abstraction — supports Ollama and OpenAI-compatible APIs.

Providers:
- "ollama": Ollama's /api/generate endpoint (localhost:11434)
- "openai": Any OpenAI-compatible /v1/chat/completions endpoint (vLLM, TGI, etc.)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: str = "ollama"
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 300


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_clock_seconds: float = 0.0


def call_llm(prompt: str, config: LLMConfig | None = None) -> LLMResponse:
    """Call an LLM via the configured provider.

    Args:
        prompt: The prompt text to send.
        config: LLM configuration.

    Returns:
        LLMResponse with the generated text and metadata.
    """
    if config is None:
        config = LLMConfig()

    start = time.perf_counter()

    if config.provider == "ollama":
        return _call_ollama(prompt, config, start)
    elif config.provider == "openai":
        return _call_openai(prompt, config, start)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def _call_ollama(prompt: str, config: LLMConfig, start: float) -> LLMResponse:
    """Call Ollama's /api/generate endpoint."""
    url = f"{config.base_url}/api/generate"
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    }

    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        return LLMResponse(
            text=f"Error calling Ollama: {e}",
            model=config.model,
            wall_clock_seconds=time.perf_counter() - start,
        )

    elapsed = time.perf_counter() - start
    return LLMResponse(
        text=data.get("response", ""),
        model=data.get("model", config.model),
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
        wall_clock_seconds=elapsed,
    )


def _call_openai(prompt: str, config: LLMConfig, start: float) -> LLMResponse:
    """Call an OpenAI-compatible /v1/chat/completions endpoint (vLLM, TGI, etc.)."""
    url = f"{config.base_url}/v1/chat/completions"
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }

    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        return LLMResponse(
            text=f"Error calling OpenAI endpoint: {e}",
            model=config.model,
            wall_clock_seconds=time.perf_counter() - start,
        )

    elapsed = time.perf_counter() - start
    text = ""
    if data.get("choices"):
        text = data["choices"][0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return LLMResponse(
        text=text,
        model=data.get("model", config.model),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        wall_clock_seconds=elapsed,
    )


def is_openai_endpoint_available(base_url: str) -> bool:
    """Check if an OpenAI-compatible endpoint is reachable."""
    try:
        req = Request(f"{base_url}/v1/models")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running."""
    try:
        req = Request(f"{base_url}/api/tags")
        with urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """List available Ollama models."""
    try:
        req = Request(f"{base_url}/api/tags")
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
