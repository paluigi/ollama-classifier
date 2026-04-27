"""SGLang inference backend.

Supports both local and remote SGLang servers via the OpenAI-compatible API.

Local usage:
    Start a local SGLang server:
        python -m sglang.launch_server --model-path meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0

    Then connect:
        backend = SGLangBackend(model="meta-llama/Llama-3.2-3B-Instruct", base_url="http://localhost:30000")

Remote usage:
    backend = SGLangBackend(model="your-model", base_url="https://your-sglang-server.com")
"""

from typing import Any, Dict, List, Optional

import httpx

from .base import ChatMessage, ChatResponse, LLMBackend


class SGLangBackend(LLMBackend):
    """Backend for SGLang inference server.

    SGLang is a fast serving system for large language models with an
    OpenAI-compatible API. It supports guided decoding and logprobs.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:30000/v1",
        *,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_tokens: int = 256,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the SGLang backend.

        Args:
            model: Model identifier (must match the model loaded on the server).
            base_url: Base URL of the SGLang OpenAI-compatible server.
                      Defaults to ``http://localhost:30000/v1``.
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
        """Perform a synchronous chat completion via SGLang."""
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
        """Perform an asynchronous chat completion via SGLang."""
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
