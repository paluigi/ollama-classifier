"""Ollama inference backend (requires Ollama ≥0.12 for logprobs support).

Wraps the Ollama Python SDK behind the :class:`LLMBackend` interface.

Constraint mechanism: JSON Schema enum via the ``format`` parameter.
The model generates JSON: ``{"label": "<chosen_label>"}``. Structural JSON
tokens (``{``, ``"label"``, ``:``, ``"``, ``}``) are filtered during trie
reconstruction. Context-dependent tokenization is used so the trie matches
the actual response tokens.

Local usage::

    from ollama_classifier.backends import OllamaBackend

    backend = OllamaBackend(model="llama3.2")

Remote usage::

    backend = OllamaBackend(model="llama3.2", host="http://remote-host:11434")
"""

import json
from typing import Any, Dict, List, Optional

from .base import ChatMessage, ChatResponse, LLMBackend, ScoringResponse, Token, TokenLogprob


class OllamaBackend(LLMBackend):
    """Backend for the Ollama runtime (≥v0.12) via the official Python SDK.

    Ollama provides a local LLM runtime with an OpenAI-compatible API and a
    native API. This backend uses the native API via the ``ollama`` Python SDK.
    JSON schema constraints and logprobs are supported as of v0.12.

    Note:
        Ollama's constraint mechanism is JSON Schema enum, which wraps the label
        in JSON structural tokens. The :meth:`tokenize` method supports
        context-dependent tokenization so the label is tokenized within the
        JSON prefix it appears in, ensuring the trie matches the response tokens.
    """

    # JSON prefix that precedes the label text in the response
    _JSON_LABEL_CONTEXT = '{"label": "'

    def __init__(
        self,
        model: str,
        *,
        host: Optional[str] = None,
        sync_client: Any = None,
        async_client: Any = None,
        timeout: float = 120.0,
        max_tokens: int = 256,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Ollama backend.

        Args:
            model: Model name (e.g., ``"llama3.2"``).
            host: Ollama server URL. Defaults to ``http://localhost:11434``.
            sync_client: Pre-initialized ``ollama.Client`` (sync). Created lazily if None.
            async_client: Pre-initialized ``ollama.AsyncClient``. Created lazily if None.
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens to generate.
            extra_body: Extra parameters merged into every request options.
        """
        super().__init__(
            model=model,
            base_url=host or "http://localhost:11434",
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        self._sync_client = sync_client
        self._async_client = async_client

    @property
    def supports_bare_label_constraint(self) -> bool:
        """False — Ollama uses JSON enum wrapper."""
        return False

    # ------------------------------------------------------------------
    # Client management (lazy import of ollama SDK)
    # ------------------------------------------------------------------

    def _get_sync_client(self) -> Any:
        if self._sync_client is None:
            from ollama import Client

            self._sync_client = Client(host=self._base_url, timeout=self._timeout)
        return self._sync_client

    async def _get_async_client(self) -> Any:
        if self._async_client is None:
            from ollama import AsyncClient

            self._async_client = AsyncClient(host=self._base_url, timeout=self._timeout)
        return self._async_client

    # ------------------------------------------------------------------
    # Constraint translation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_json_enum(labels: List[str]) -> Dict[str, Any]:
        """Build JSON schema with enum constraint for Ollama's format parameter."""
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": labels},
            },
            "required": ["label"],
        }

    @staticmethod
    def _get_token_context() -> str:
        """The JSON prefix that precedes the label in the response.

        Used for context-dependent tokenization so the trie matches the
        actual response tokens.
        """
        return OllamaBackend._JSON_LABEL_CONTEXT

    # ------------------------------------------------------------------
    # Logprob parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_logprobs(response: Any) -> Optional[List[TokenLogprob]]:
        """Extract ``TokenLogprob`` list from an Ollama response object."""
        lps = getattr(response, "logprobs", None)
        if not lps:
            return None
        result: list[TokenLogprob] = []
        for lp in lps:
            top: dict[str, float] = {}
            for alt in getattr(lp, "top_logprobs", []) or []:
                top[alt.token] = alt.logprob
            result.append(
                TokenLogprob(
                    token=getattr(lp, "token", ""),
                    logprob=getattr(lp, "logprob", 0.0),
                    top_logprobs=top,
                )
            )
        return result

    @staticmethod
    def _extract_label(content: str) -> str:
        """Extract the label from a JSON response, falling back to raw content."""
        try:
            return json.loads(content).get("label", content)
        except (json.JSONDecodeError, TypeError):
            return content

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        constrain_labels: Optional[List[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Perform a synchronous constrained chat completion via Ollama."""
        client = self._get_sync_client()
        fmt = self._build_json_enum(constrain_labels) if constrain_labels else None

        response = client.chat(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            format=fmt,
            logprobs=logprobs,
            top_logprobs=top_logprobs if logprobs else None,
            options={
                "temperature": temperature,
                "num_predict": self._max_tokens,
                **self._extra_body,
            },
        )

        content = response.message.content
        return ChatResponse(
            content=content,
            label=self._extract_label(content),
            logprobs=self._parse_logprobs(response),
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def score(
        self,
        messages: List[ChatMessage],
        completion: str,
    ) -> ScoringResponse:
        """Score a completion using Ollama's generate endpoint with suffix.

        Uses ``client.generate(prompt=..., suffix=completion)`` to compute
        the per-token logprobs of the completion given the prompt context.
        No generation occurs (``num_predict=0``).
        """
        client = self._get_sync_client()
        prompt = "\n\n".join(
            m.content for m in messages if m.role in ("system", "user")
        )

        response = client.generate(
            model=self._model,
            prompt=prompt,
            suffix=completion,
            logprobs=True,
            options={
                "temperature": 0.0,
                "num_predict": 0,
                **self._extra_body,
            },
        )

        return ScoringResponse(
            completion=completion,
            logprobs=self._parse_logprobs(response) or [],
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def tokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Tokenize text using Ollama's tokenize API.

        If ``context`` is provided, tokenizes ``context + text`` and returns
        only the tokens corresponding to ``text``. This ensures
        context-dependent tokenization matches the actual response tokens
        (critical for JSON-wrapped labels).
        """
        client = self._get_sync_client()
        full_text = (context or "") + text
        response = client.tokenize(model=self._model, text=full_text)
        tokens: list[int] = response.get("tokens", [])

        if context:
            ctx_response = client.tokenize(model=self._model, text=context)
            ctx_len = len(ctx_response.get("tokens", []))
            tokens = tokens[ctx_len:]

        return [Token(text="", id=t) for t in tokens]

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def achat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        constrain_labels: Optional[List[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Async constrained chat completion via Ollama."""
        client = await self._get_async_client()
        fmt = self._build_json_enum(constrain_labels) if constrain_labels else None

        response = await client.chat(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            format=fmt,
            logprobs=logprobs,
            top_logprobs=top_logprobs if logprobs else None,
            options={
                "temperature": temperature,
                "num_predict": self._max_tokens,
                **self._extra_body,
            },
        )

        content = response.message.content
        return ChatResponse(
            content=content,
            label=self._extract_label(content),
            logprobs=self._parse_logprobs(response),
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def ascore(
        self,
        messages: List[ChatMessage],
        completion: str,
    ) -> ScoringResponse:
        """Async completion scoring via Ollama."""
        client = await self._get_async_client()
        prompt = "\n\n".join(
            m.content for m in messages if m.role in ("system", "user")
        )

        response = await client.generate(
            model=self._model,
            prompt=prompt,
            suffix=completion,
            logprobs=True,
            options={
                "temperature": 0.0,
                "num_predict": 0,
                **self._extra_body,
            },
        )

        return ScoringResponse(
            completion=completion,
            logprobs=self._parse_logprobs(response) or [],
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def atokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Async tokenization via Ollama."""
        client = await self._get_async_client()
        full_text = (context or "") + text
        response = await client.tokenize(model=self._model, text=full_text)
        tokens: list[int] = response.get("tokens", [])

        if context:
            ctx_response = await client.tokenize(model=self._model, text=context)
            ctx_len = len(ctx_response.get("tokens", []))
            tokens = tokens[ctx_len:]

        return [Token(text="", id=t) for t in tokens]
