

from typing import Callable, Optional
import os
import subprocess

try:
    import openai
except Exception:
    openai = None

try:
    import ollama
except Exception:
    ollama = None


def is_ollama_available() -> bool:
    if ollama is not None:
        return True
    try:
        subprocess.run(["ollama", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def is_cloud_llm_configured() -> bool:
    if openai is None:
        return False
    return bool(os.environ.get("OPENAI_API_KEY"))


def get_ollama_predict(model: str = "qwen2.5:3b", temperature: float = 0.0, timeout: int = 180) -> Callable[[str], str]:
    if not is_ollama_available():
        raise RuntimeError("Ollama HTTP server not reachable at http://127.0.0.1:11434")

    try:
        import requests
    except Exception as e:
        raise RuntimeError("The 'requests' package is required for Ollama HTTP mode") from e

    url = "http://127.0.0.1:11434/api/generate"

    def _predict(prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Ollama HTTP request failed: {e}") from e

        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"Ollama inference failed: status {resp.status_code}: {resp.text}")

        try:
            j = resp.json()
        except Exception as e:
            raise RuntimeError(f"Ollama returned non-JSON response: {resp.status_code}: {resp.text}") from e

        if "response" in j and isinstance(j["response"], str):
            return j["response"]

        if "output" in j and isinstance(j["output"], str):
            return j["output"]
        if "text" in j and isinstance(j["text"], str):
            return j["text"]

        raise RuntimeError(f"Ollama inference returned unexpected payload: {j}")

    return _predict


def get_default_llm_predict(temperature: float = 0.0) -> Callable[[str], str]:
    if not is_cloud_llm_configured():
        raise RuntimeError("OpenAI is not configured. Set OPENAI_API_KEY and install openai package.")

    def _predict(prompt: str) -> str:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if getattr(openai, "gpt", None) is not None else "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=3000,
            )
            content = resp.choices[0].message.content
            return content
        except Exception:
            resp = openai.Completion.create(model="gpt-4", prompt=prompt, max_tokens=3000, temperature=temperature)
            return resp.choices[0].text

    return _predict


def get_llm_predict(temperature: float = 0.0, prefer_ollama: bool = True) -> Callable[[str], str]:
    if prefer_ollama and is_ollama_available():
        return get_ollama_predict(temperature=temperature)
    if is_cloud_llm_configured():
        return get_default_llm_predict(temperature=temperature)
    raise RuntimeError("No LLM available: install/configure Ollama or set OPENAI_API_KEY")


def predict(prompt: str, temperature: float = 0.0) -> str:
    fn = get_llm_predict(temperature=temperature)
    return fn(prompt)
