"""Simple Python client for the deployed LLM proxy.

Requires:
    pip install requests

Environment variables:
    LLM_PROXY_BASE_URL
    LLM_PROXY_TOKEN
    LLM_PROXY_TIMEOUT       (optional, seconds)
    LLM_PROXY_MAX_TOKENS    (optional, hard client-side ceiling; default 8192)

CHANGES (to match the Gemini-only annotation pipeline):
    * MAX_MAX_TOKENS raised 1000 -> 8192. The old 1000 cap silently clamped every
      request (payload["max_tokens"] = min(max_tokens, MAX_MAX_TOKENS)), which
      truncated long-poem JSON. config.max_tokens_for() now asks for up to 8192,
      so the client must not clamp below that. Now env-overridable.
    * DEFAULT_TIMEOUT raised 60 -> 120s, since longer generations take longer.
    * SUPPORTED_MODELS unchanged (this stays a general client); the pipeline only
      ever calls "gemini" (primary) and "gemini-3-flash" (repair/fallback).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_BASE_URL = "https://llmproxy.magnocode.tech"
DEFAULT_TIMEOUT = 120


def _resolve_max_tokens_ceiling() -> int:
    """Client-side hard ceiling for max_tokens. Aligns with config.max_tokens_for()
    (which tops out at 8192). Override with LLM_PROXY_MAX_TOKENS if your proxy
    allows a different limit."""
    raw = os.getenv("LLM_PROXY_MAX_TOKENS", "8192").strip()
    try:
        value = int(raw)
    except ValueError:
        return 8192
    return value if value >= 1 else 8192


MAX_MAX_TOKENS = _resolve_max_tokens_ceiling()
# General client: the annotation pipeline uses only "gemini" and "gemini-3-flash".
SUPPORTED_MODELS = {"gpt", "claude", "gemini", "gemini-3-flash"}


class LLMProxyError(RuntimeError):
    """Raised when the proxy returns an error response."""

    def __init__(self, status_code: int, code: str, message: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(f"{status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.payload = payload or {}


@dataclass
class LLMProxyClient:
    token: str
    base_url: str = DEFAULT_BASE_URL
    timeout: int = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        if not self.token or not self.token.strip():
            raise ValueError("token is required")

        self.base_url = self.base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token.strip()}",
                "Content-Type": "application/json",
            }
        )

    def list_models(self) -> dict[str, Any]:
        return self._request("GET", "/v1/models")

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
    ) -> dict[str, Any]:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{model}'. Expected one of: {sorted(SUPPORTED_MODELS)}")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        normalized_messages: list[dict[str, str]] = []
        has_user_message = False

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be a dict")

            role = message.get("role")
            content = message.get("content")

            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"messages[{index}].role must be one of system, user, assistant")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"messages[{index}].content must be a non-empty string")

            if role == "user":
                has_user_message = True

            normalized_messages.append({"role": role, "content": content})

        if not has_user_message:
            raise ValueError("messages must include at least one user message")

        payload: dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
        }

        if temperature is not None:
            if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
                raise ValueError("temperature must be between 0 and 2")
            payload["temperature"] = temperature

        if top_p is not None:
            if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
                raise ValueError("top_p must be between 0 and 1")
            payload["top_p"] = top_p

        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1:
                raise ValueError("max_tokens must be a positive integer")
            # Clamp only to the (now 8192) client ceiling instead of the old 1000.
            payload["max_tokens"] = min(max_tokens, MAX_MAX_TOKENS)

        if stop is not None:
            payload["stop"] = stop

        return self._request("POST", "/v1/chat/completions", json=payload)

    def text(
        self,
        prompt: str,
        *,
        model: str = "gemini",
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.chat(
            model,
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        return response["choices"][0]["message"]["content"]

    def gpt(self, prompt: str, **kwargs: Any) -> str:
        return self.text(prompt, model="gpt", **kwargs)

    def claude(self, prompt: str, **kwargs: Any) -> str:
        return self.text(prompt, model="claude", **kwargs)

    def gemini(self, prompt: str, **kwargs: Any) -> str:
        return self.text(prompt, model="gemini", **kwargs)

    def gemini_flash(self, prompt: str, **kwargs: Any) -> str:
        return self.text(prompt, model="gemini-3-flash", **kwargs)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMProxyError(response.status_code, "non_json_response", response.text.strip() or "Non-JSON response") from exc

        if response.ok:
            return payload

        error = payload.get("error") if isinstance(payload, dict) else None
        code = error.get("code", "proxy_error") if isinstance(error, dict) else "proxy_error"
        message = error.get("message", "Proxy request failed") if isinstance(error, dict) else "Proxy request failed"
        raise LLMProxyError(response.status_code, code, message, payload if isinstance(payload, dict) else None)


def build_client_from_env() -> LLMProxyClient:
    token = os.getenv("LLM_PROXY_TOKEN", "").strip()
    if not token:
        raise ValueError("Set LLM_PROXY_TOKEN before using the client")

    base_url = os.getenv("LLM_PROXY_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    timeout_raw = os.getenv("LLM_PROXY_TIMEOUT", str(DEFAULT_TIMEOUT)).strip()

    try:
        timeout = int(timeout_raw)
    except ValueError as exc:
        raise ValueError("LLM_PROXY_TIMEOUT must be an integer") from exc

    return LLMProxyClient(token=token, base_url=base_url, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple client for the LLM proxy")
    parser.add_argument("--model", default="gemini", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--prompt", help="User prompt to send")
    parser.add_argument("--system", help="Optional system instruction")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--list-models", action="store_true", help="List models instead of sending a prompt")
    args = parser.parse_args()

    client = build_client_from_env()

    if args.list_models:
        models = client.list_models()
        for item in models.get("data", []):
            print(f"{item['alias']}\tavailable={item['available']}\tupstream={item['upstream_model']}")
        return

    if not args.prompt:
        parser.error("--prompt is required unless --list-models is used")

    print(
        client.text(
            args.prompt,
            model=args.model,
            system=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )


if __name__ == "__main__":
    main()
