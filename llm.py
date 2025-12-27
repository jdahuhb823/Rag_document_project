"""Simple LLM interface: use Ollama (local) if available, otherwise fall back to OpenAI if configured.
Provides `get_llm_predict()` and `predict()` helpers.
"""
from typing import Callable
import os
import subprocess

try:
    import requests
except Exception:
    requests = None

try:
    import openai
except Exception:
    openai = None


def is_ollama_available() -> bool:
    try:
        subprocess.run(["ollama", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def is_openai_configured() -> bool:
    return openai is not None and bool(os.environ.get("OPENAI_API_KEY"))


def get_ollama_predict(model: str = "qwen2.5:3b", timeout: int = 180) -> Callable[[str], str]:
    if not is_ollama_available():
        raise RuntimeError("Ollama not available")
    if requests is None:
        raise RuntimeError("requests package is required for Ollama HTTP mode")

    url = "http://127.0.0.1:11434/api/generate"

    def _predict(prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        j = resp.json()
        for key in ("response", "output", "text"):
            if key in j and isinstance(j[key], str):
                return j[key]
        raise RuntimeError("Unexpected Ollama response: %r" % (j,))

    return _predict


def get_openai_predict(temperature: float = 0.0) -> Callable[[str], str]:
    if not is_openai_configured():
        raise RuntimeError("OpenAI not configured")

    def _predict(prompt: str) -> str:
        # Use ChatCompletion if available
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if getattr(openai, "gpt", None) is not None else "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception:
            resp = openai.Completion.create(model="gpt-4", prompt=prompt, max_tokens=1024, temperature=temperature)
            return resp.choices[0].text

    return _predict


def get_llm_predict(prefer_ollama: bool = True, temperature: float = 0.0) -> Callable[[str], str]:
    if prefer_ollama and is_ollama_available():
        return get_ollama_predict()
    if is_openai_configured():
        return get_openai_predict(temperature=temperature)
    raise RuntimeError("No LLM available: install Ollama or set OPENAI_API_KEY")


def predict(prompt: str, temperature: float = 0.0) -> str:
    fn = get_llm_predict(temperature=temperature)
    return fn(prompt)
