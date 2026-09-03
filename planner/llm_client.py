"""Thin OpenAI-compatible client for llama.cpp's llama-server (stdlib only).

Talks to the local `llama serve` HTTP API at /v1/chat/completions. Model id is
auto-discovered from /v1/models so nothing is hardcoded.
"""

import json
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8080"


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 240.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = self._discover_model()

    def _discover_model(self) -> str:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/v1/models", timeout=min(self.timeout, 10)
            ) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise LLMError(
                f"llama-server not reachable at {self.base_url} ({exc.reason}). "
                f"Start it with: ~/.llama-app/llama serve -m <model.gguf>"
            ) from exc
        models = data.get("data", [])
        if not models:
            raise LLMError(f"llama-server at {self.base_url} reports no models")
        return models[0]["id"]

    def _request(self, messages: list[dict], max_tokens: int = 1024,
                 temperature: float = 0.0, json_mode: bool = True,
                 tools: list[dict] | None = None,
                 tool_choice: str | dict | None = None) -> dict:
        """POST one chat completion and return the assistant message dict."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # URLError covers connection refused / HTTP errors; TimeoutError
            # (socket timeout) is a sibling OSError subclass that the old
            # `except URLError` let escape uncaught — a slow server then
            # crashed the whole loop instead of falling back to the heuristic
            # planner. Convert both to LLMError so callers handle one type.
            if isinstance(exc, TimeoutError):
                detail = f"request timed out after {self.timeout}s"
            else:
                detail = str(getattr(exc, "reason", exc))
            raise LLMError(f"llama-server request failed: {detail}") from exc
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response from llama-server: {data}") from exc

    def chat(self, messages: list[dict], max_tokens: int = 1024,
               temperature: float = 0.0, json_mode: bool = True) -> tuple[str, dict]:
           """Return (content, full_assistant_message).

           `full_assistant_message` may contain additional fields (e.g. reasoning
           content) provided by some servers. Returning it lets callers log or
           inspect reasoning separately from the user-visible `content` string.
           """
           msg = self._request(messages, max_tokens=max_tokens,
                           temperature=temperature, json_mode=json_mode)
           return msg.get("content") or "", msg

    def chat_message(self, messages: list[dict], max_tokens: int = 1024,
                     temperature: float = 0.0, json_mode: bool = False,
                     tools: list[dict] | None = None,
                     tool_choice: str | dict | None = None
                     ) -> tuple[str, list | None]:
        """Full assistant reply: (content, tool_calls). `content` may be empty
        when the model asks to call a tool. `tool_calls` is None when the model
        answered directly."""
        msg = self._request(messages, max_tokens=max_tokens,
                    temperature=temperature, json_mode=json_mode,
                    tools=tools, tool_choice=tool_choice)
        return msg.get("content") or "", msg.get("tool_calls"), msg
