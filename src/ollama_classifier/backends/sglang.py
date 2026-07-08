"""SGLang inference backend.

Supports both local and remote SGLang servers via the OpenAI-compatible API.
SGLang supports regex constraints for bare-label generation, producing clean
label text with no JSON wrapper.

**Scoring approach:** ``score()`` uses the ``/v1/completions`` endpoint with
``echo=True`` to recover the model's genuine per-label logprobs via prefill
(no constraint, no generation). This produces differentiated confidence scores
for ``classify()`` because the label is evaluated as an unexpected continuation
of the prompt, not forced by a constraint.

**Tokenization approach:** ``tokenize()`` uses empirical **forced constrained
generation** (forcing the label as the only valid choice). This is necessary
because standalone BPE tokenization produces different token boundaries than
the model emits under regex guidance, which would break trie-based divergence
scoring in ``generate()``.

Local server::

    python -m sglang.launch_server \\
        --model-path meta-llama/Llama-3.2-3B-Instruct \\
        --host 0.0.0.0 --port 30000

Connect::

    backend = SGLangBackend(
        model="meta-llama/Llama-3.2-3B-Instruct",
        base_url="http://localhost:30000/v1",
    )
"""

import re
from typing import Any, Dict, List, Optional

import httpx

from .base import ChatMessage, ChatResponse, LLMBackend, ScoringResponse, Token, TokenLogprob


class SGLangBackend(LLMBackend):
    """Backend for SGLang inference server.

    SGLang is a fast serving system for large language models with an
    OpenAI-compatible API. It supports regex-guided decoding and logprobs.
    Logprobs are pre-mask (raw model logits before regex masking).

    ``score()`` uses echo/prefill (``/v1/completions`` with ``echo=True``) so
    that ``classify()`` produces differentiated confidence. ``tokenize()``
    uses forced constrained generation so the token strings used for trie
    construction match the actual constrained-generation token boundaries.
    """

    # End-of-sequence / special tokens to filter from constrained responses
    _SPECIAL_TOKENS = frozenset(
        {
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
            "<|end_of_turn|>",
            "<|eot_id|>",
            "<|end|>",
            "<|eom_id|>",
        }
    )

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
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        # Empirical tokenization is deterministic per label, so memoize to
        # amortize the forced-generation setup cost.
        self._token_cache: dict[str, list[Token]] = {}
        # Reusable HTTP clients for connection pooling across N classify() calls
        self._sync_http: httpx.Client | None = None
        self._async_http: httpx.AsyncClient | None = None

    @property
    def supports_bare_label_constraint(self) -> bool:
        """True — SGLang regex constraint generates bare label text."""
        return True

    def _get_sync_http(self) -> httpx.Client:
        """Return a pooled sync HTTP client (created lazily)."""
        if self._sync_http is None:
            self._sync_http = httpx.Client(timeout=self._timeout)
        return self._sync_http

    async def _get_async_http(self) -> httpx.AsyncClient:
        """Return a pooled async HTTP client (created lazily)."""
        if self._async_http is None:
            self._async_http = httpx.AsyncClient(timeout=self._timeout)
        return self._async_http

    def _apply_constraint(self, body: Dict[str, Any], labels: List[str]) -> None:
        """Apply regex constraint for bare-label generation."""
        escaped = [re.escape(lbl) for lbl in labels]
        body["regex"] = f"({'|'.join(escaped)})"

    @staticmethod
    def _label_token_logprobs(
        logprobs: List[TokenLogprob],
    ) -> List[TokenLogprob]:
        """Extract label tokens from a bare-label constrained response.

        For SGLang (bare-label backend), this filters out special /
        end-of-sequence tokens. The regex constraint guarantees only the label
        text is generated, so no JSON-structure filtering is needed.
        """
        return [
            lp
            for lp in logprobs
            if lp.token.strip() and lp.token not in SGLangBackend._SPECIAL_TOKENS
        ]

    def _render_prompt(self, messages: List[ChatMessage]) -> str:
        """Render messages to a plain text prompt for the completions endpoint."""
        parts: list[str] = []
        for m in messages:
            if m.role == "system":
                parts.append(f"<|system|>\n{m.content}")
            elif m.role == "user":
                parts.append(f"<|user|>\n{m.content}")
        return "\n\n".join(parts) + "\n\n<|assistant|>\n"

    def _tokenize_count(self, text: str) -> int:
        """Count tokens via the ``/tokenize`` endpoint (correct ``prompt`` field).

        Raises on HTTP errors — no silent masking.
        """
        url = f"{self._base_url}/tokenize"
        body = {"model": self._model, "prompt": text}
        resp = self._get_sync_http().post(url, headers=self._build_headers(), json=body)
        resp.raise_for_status()
        return resp.json().get("count", 0)

    async def _atokenize_count(self, text: str) -> int:
        """Async token count via the ``/tokenize`` endpoint."""
        url = f"{self._base_url}/tokenize"
        body = {"model": self._model, "prompt": text}
        client = await self._get_async_http()
        resp = await client.post(url, headers=self._build_headers(), json=body)
        resp.raise_for_status()
        return resp.json().get("count", 0)

    # ------------------------------------------------------------------
    # Sync
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
        """Synchronous constrained chat completion via SGLang."""
        url = f"{self._base_url}/chat/completions"
        body = self._build_chat_body(
            messages,
            temperature=temperature,
            constrain_labels=constrain_labels,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        resp = self._get_sync_http().post(url, headers=self._build_headers(), json=body)
        resp.raise_for_status()
        result = self._parse_chat_response(resp.json())
        result.label = result.content.strip()
        return result

    def score(
        self,
        messages: List[ChatMessage],
        completion: str,
    ) -> ScoringResponse:
        """Score a completion using echo/prefill logprobs (no constraint).

        Uses ``/v1/completions`` with ``echo=True`` to recover the model's
        genuine per-token logprobs for the label as an unexpected continuation
        of the prompt. This produces differentiated confidence for ``classify()``
        because the label is evaluated without any constraint forcing the model
        to accept it.

        The ``/tokenize`` endpoint (with correct ``"prompt"`` field) pinpoints
        the exact label-token boundary. The spurious ``max_tokens=1`` generated
        token is discarded by slicing to ``total_len``.
        """
        prompt = self._render_prompt(messages)
        prompt_with_completion = prompt + completion

        prompt_len = self._tokenize_count(prompt)
        total_len = self._tokenize_count(prompt_with_completion)

        url = f"{self._base_url}/completions"
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt_with_completion,
            "echo": True,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
            **self._extra_body,
        }
        resp = self._get_sync_http().post(url, headers=self._build_headers(), json=body)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        all_logprobs = choice.get("logprobs", {})
        tokens_list = all_logprobs.get("tokens", [])
        token_lps_list = all_logprobs.get("token_logprobs", [])
        top_lps_list = all_logprobs.get("top_logprobs", [])

        completion_tokens = tokens_list[prompt_len:total_len]
        completion_lps = token_lps_list[prompt_len:total_len]
        completion_top = top_lps_list[prompt_len:total_len]

        token_logprobs: list[TokenLogprob] = []
        for i, tok in enumerate(completion_tokens):
            top: dict[str, float] = {}
            if i < len(completion_top) and completion_top[i]:
                for t, lp in completion_top[i].items():
                    top[t] = lp
            lp = completion_lps[i] if i < len(completion_lps) else 0.0
            token_logprobs.append(TokenLogprob(token=tok, logprob=lp or 0.0, top_logprobs=top))

        if not token_logprobs:
            raise RuntimeError(
                f"score({completion!r}): echo returned no label tokens"
            )
        return ScoringResponse(completion=completion, logprobs=token_logprobs, raw=data)

    def tokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Tokenize text via empirical forced generation.

        Forces ``text`` as the only valid label in a constrained :meth:`chat`
        call and reads back the emitted value tokens. This is necessary because
        standalone BPE tokenization (via the ``/tokenize`` endpoint) produces
        different token boundaries than the model emits under regex guidance,
        which would break trie-based divergence scoring. Results are memoized
        per label.

        The ``context`` argument is accepted for interface compatibility but
        ignored: SGLang generates bare labels with no wrapper.
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
        lps = self._label_token_logprobs(response.logprobs or [])
        tokens = [Token(text=lp.token, id=-1) for lp in lps]
        tokens = tokens or [Token(text=text, id=-1)]
        self._token_cache[text] = tokens
        return tokens

    # ------------------------------------------------------------------
    # Async
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
        """Async constrained chat completion via SGLang."""
        url = f"{self._base_url}/chat/completions"
        body = self._build_chat_body(
            messages,
            temperature=temperature,
            constrain_labels=constrain_labels,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        client = await self._get_async_http()
        resp = await client.post(url, headers=self._build_headers(), json=body)
        resp.raise_for_status()
        result = self._parse_chat_response(resp.json())
        result.label = result.content.strip()
        return result

    async def ascore(
        self,
        messages: List[ChatMessage],
        completion: str,
    ) -> ScoringResponse:
        """Async completion scoring via echo/prefill logprobs.

        See :meth:`score` for the rationale.
        """
        prompt = self._render_prompt(messages)
        prompt_with_completion = prompt + completion

        prompt_len = await self._atokenize_count(prompt)
        total_len = await self._atokenize_count(prompt_with_completion)

        url = f"{self._base_url}/completions"
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt_with_completion,
            "echo": True,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
            **self._extra_body,
        }
        client = await self._get_async_http()
        resp = await client.post(url, headers=self._build_headers(), json=body)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        all_logprobs = choice.get("logprobs", {})
        tokens_list = all_logprobs.get("tokens", [])
        token_lps_list = all_logprobs.get("token_logprobs", [])
        top_lps_list = all_logprobs.get("top_logprobs", [])

        completion_tokens = tokens_list[prompt_len:total_len]
        completion_lps = token_lps_list[prompt_len:total_len]
        completion_top = top_lps_list[prompt_len:total_len]

        token_logprobs: list[TokenLogprob] = []
        for i, tok in enumerate(completion_tokens):
            top: dict[str, float] = {}
            if i < len(completion_top) and completion_top[i]:
                for t, lp in completion_top[i].items():
                    top[t] = lp
            lp = completion_lps[i] if i < len(completion_lps) else 0.0
            token_logprobs.append(TokenLogprob(token=tok, logprob=lp or 0.0, top_logprobs=top))

        if not token_logprobs:
            raise RuntimeError(
                f"ascore({completion!r}): echo returned no label tokens"
            )
        return ScoringResponse(completion=completion, logprobs=token_logprobs, raw=data)

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
        lps = self._label_token_logprobs(response.logprobs or [])
        tokens = [Token(text=lp.token, id=-1) for lp in lps]
        tokens = tokens or [Token(text=text, id=-1)]
        self._token_cache[text] = tokens
        return tokens
