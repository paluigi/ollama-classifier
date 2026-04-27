"""llama.cpp inference backend.

Supports both local and remote llama.cpp servers via the OpenAI-compatible API
provided by ``llama-server``.

Local usage:
    Start a local llama.cpp server:
        ./llama-server -m model.gguf --host 0.0.0.0 --port 8080 -c 4096

    Then connect:
        backend = LlamaCppBackend(model="model", base_url="http://localhost:8080/v1")

Remote usage:
    backend = LlamaCppBackend(model="your-model", base_url="https://your-llamacpp-server.com/v1")
"""

from typing import Any, Dict, List, Optional

import httpx

from .base import ChatMessage, ChatResponse, LLMBackend


class LlamaCppBackend(LLMBackend):
    """Backend for llama.cpp server (``llama-server``).

    llama.cpp provides a lightweight inference server with an OpenAI-compatible
    API. JSON schema constraints and logprobs are supported when compiled with
    the appropriate flags.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        *,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_tokens: int = 256,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the llama.cpp backend.

        Args:
            model: Model identifier (filename or alias used when starting the server).
            base_url: Base URL of the llama.cpp server.
                      Defaults to ``http://localhost:8080/v1``.
            api_key: Optional API key. Defaults to "not-needed".
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens to generate.
            extra_body: Extra parameters merged into every request body.
        """
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            extra_body=extra_body,
        )
        self._max_tokens = max_tokens

    def _build_body(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        guided_json: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> Dict[str, Any]:
        body = super()._build_body(
            messages,
            temperature=temperature,
            guided_json=guided_json,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        body["max_tokens"] = self._max_tokens
        return body

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        guided_json: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Perform a synchronous chat completion via llama.cpp server."""
        url = f"{self._base_url}/chat/completions"
        body = self._build_body(
            messages,
            temperature=temperature,
            guided_json=guided_json,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=self._build_headers(), json=body)
            response.raise_for_status()
            return self._parse_response(response.json())

    async def achat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        guided_json: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Perform an asynchronous chat completion via llama.cpp server."""
        url = f"{self._base_url}/chat/completions"
        body = self._build_body(
            messages,
            temperature=temperature,
            guided_json=guided_json,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, headers=self._build_headers(), json=body)
            response.raise_for_status()
            return self._parse_response(response.json())
