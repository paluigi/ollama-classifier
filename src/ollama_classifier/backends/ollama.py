"""Ollama inference backend (requires Ollama ≥0.12 for logprobs support).

Wraps the Ollama Python SDK behind the :class:`LLMBackend` interface.

Constraint mechanism: JSON Schema enum via the ``format`` parameter.
The model generates JSON: ``{"label": "<chosen_label>"}``. Structural JSON
tokens (``{``, ``"label"``, ``:``, ``"``, ``}``) are filtered during trie
reconstruction and completion scoring.

Note:
    Modern Ollama removed the ``/api/tokenize`` endpoint and does not support
    fill-in-the-middle ("insert") on instruct models. This backend therefore
    obtains exact label tokenization and completion scores through empirical
    *forced constrained generation* (forcing a label as the only valid choice
    and reading back the model's genuine per-token logprobs). No ``/api/tokenize``
    or ``suffix``/insert calls are used.

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
        in JSON structural tokens. Tokenization and completion scoring are both
        obtained through empirical forced constrained generation (modern Ollama
        has no ``/api/tokenize`` and instruct models reject insert/fill-mode),
        so the label tokens used for trie reconstruction match the response
        tokens exactly.
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
        # Empirical tokenization is deterministic per label (the JSON wrapper
        # prefix is constant), so memoize per label to amortize the setup cost.
        self._token_cache: dict[str, list[Token]] = {}

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

    @staticmethod
    def _label_token_logprobs(
        logprobs: List[TokenLogprob], label: str
    ) -> List[TokenLogprob]:
        """Extract the label-value tokens (with their logprobs) from a
        ``{"label": "<label>"}`` constrained response.

        Robust to model-specific whitespace in the emitted JSON. The returned
        tokens keep their *exact* emitted strings so they match the tokens the
        model produces during multi-label constrained generation in
        :meth:`LLMClassifier.generate`.

        Primary strategy: reconstruct the full emitted string, locate the value
        span after the JSON ``:`` separator, and map that character span back to
        token indices. Falls back to JSON-skeleton filtering if the span mapping
        yields nothing.
        """
        full = "".join(lp.token for lp in logprobs)

        # ---- Primary: character-offset span mapping ----
        try:
            colon = full.index(":")
            vstart = full.index(label, colon + 1)
            vend = vstart + len(label)
            out: list[TokenLogprob] = []
            pos = 0
            for lp in logprobs:
                tok_end = pos + len(lp.token)
                if tok_end > vstart and pos < vend:
                    out.append(lp)
                pos = tok_end
            if out:
                return out
        except ValueError:
            pass

        # ---- Fallback: drop pure JSON-structure tokens / the "label" key ----
        out = []
        for lp in logprobs:
            stripped = lp.token.strip()
            cleaned = stripped.strip('"{}: \t\n')
            if cleaned == "" or stripped == "label":
                continue
            out.append(lp)
        return out

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
        """Score a completion by forcing it as the single valid label.

        Modern Ollama (and instruct models in general) do not support the
        fill-in-the-middle ("insert") mode that ``/api/generate`` with
        ``suffix=`` requires. Instead, this forces ``completion`` as the only
        valid label via a JSON-enum constrained :meth:`chat` call and reads back
        the model's genuine per-token logprobs (teacher forcing). No free
        generation occurs beyond the forced label.
        """
        response = self.chat(
            messages=messages,
            temperature=0.0,
            constrain_labels=[completion],
            logprobs=True,
            top_logprobs=1,
        )
        lps = self._label_token_logprobs(response.logprobs or [], completion)
        if not lps:
            raise RuntimeError(
                f"score({completion!r}): forced generation returned no value tokens"
            )
        return ScoringResponse(
            completion=completion,
            logprobs=lps,
            raw=response.raw,
        )

    def tokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Tokenize text via empirical forced generation.

        Modern Ollama removed the ``/api/tokenize`` endpoint (and the SDK no
        longer exposes a ``tokenize`` method). To get the *exact* token strings
        the model emits for ``text`` inside the JSON wrapper, this forces
        ``text`` as the only valid label in a constrained :meth:`chat` call and
        reads back the emitted value tokens. Results are memoized per label.

        The ``context`` argument is accepted for interface compatibility but
        ignored: Ollama always wraps the label in the constant JSON prefix
        ``'{"label": "'`` regardless of surrounding prompt tokens.
        """
        cached = self._token_cache.get(text)
        if cached is not None:
            return cached

        response = self.chat(
            messages=[ChatMessage(role="user", content=text)],
            temperature=0.0,
            constrain_labels=[text],
            logprobs=True,
            top_logprobs=1,
        )
        lps = self._label_token_logprobs(response.logprobs or [], text)
        tokens = [Token(text=lp.token, id=-1) for lp in lps]
        tokens = tokens or [Token(text=text, id=-1)]
        self._token_cache[text] = tokens
        return tokens

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
        """Async completion scoring via forced constrained generation.

        See :meth:`score` for the rationale (no insert/fill-in-the-middle).
        """
        response = await self.achat(
            messages=messages,
            temperature=0.0,
            constrain_labels=[completion],
            logprobs=True,
            top_logprobs=1,
        )
        lps = self._label_token_logprobs(response.logprobs or [], completion)
        if not lps:
            raise RuntimeError(
                f"ascore({completion!r}): forced generation returned no value tokens"
            )
        return ScoringResponse(
            completion=completion,
            logprobs=lps,
            raw=response.raw,
        )

    async def atokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Async empirical tokenization via forced constrained generation.

        See :meth:`tokenize` for the rationale. Reuses the sync memoization
        cache since tokenization is deterministic.
        """
        cached = self._token_cache.get(text)
        if cached is not None:
            return cached

        response = await self.achat(
            messages=[ChatMessage(role="user", content=text)],
            temperature=0.0,
            constrain_labels=[text],
            logprobs=True,
            top_logprobs=1,
        )
        lps = self._label_token_logprobs(response.logprobs or [], text)
        tokens = [Token(text=lp.token, id=-1) for lp in lps]
        tokens = tokens or [Token(text=text, id=-1)]
        self._token_cache[text] = tokens
        return tokens
