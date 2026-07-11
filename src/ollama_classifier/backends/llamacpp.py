"""llama.cpp inference backend.

Supports both local and remote ``llama-server`` instances via the OpenAI-compatible API.

llama.cpp supports GBNF (GGML BNF) grammar constraints, which can express bare
label alternatives directly::

    root ::= "positive" | "negative" | "neutral"

This generates **bare label text** -- no JSON wrapper -- so logprob
reconstruction is clean. ``llama-server`` accepts a non-standard ``grammar``
field on the ``/v1/chat/completions`` endpoint.

**Scoring approach:** ``score()`` uses forced constrained generation via GBNF
grammar -- forcing each label individually and reading back the model's genuine
per-token logprobs (pre-mask). llama.cpp does NOT support ``echo=True`` on the
completions endpoint (it only returns generated-token logprobs, not prompt
tokens), so the echo/prefill approach used by SGLang/vLLM is not available.

**Tokenization approach:** ``tokenize()`` uses the same forced constrained
generation to obtain empirical token strings that match the actual
constrained-generation token boundaries.

Local server::

    ./llama-server -m model.gguf --host 0.0.0.0 --port 8080 -c 4096

Connect::

    backend = LlamaCppBackend(model="model", base_url="http://localhost:8080/v1")
"""

from typing import Any, Dict, List, Optional

import httpx

from .base import ChatMessage, ChatResponse, LLMBackend, ScoringResponse, Token, TokenLogprob


class LlamaCppBackend(LLMBackend):
    """Backend for llama.cpp server (``llama-server``).

    llama.cpp provides a lightweight inference server with an OpenAI-compatible
    API. GBNF grammar constraints and logprobs are supported. Logprobs are
    pre-mask (model's raw distribution before grammar masking).

    Both ``score()`` and ``tokenize()`` use forced constrained generation via
    GBNF grammar because llama.cpp's ``echo`` only returns generated-token
    logprobs (not prompt tokens), making the echo/prefill approach unusable.
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
        base_url: str = "http://localhost:8080/v1",
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
        self._token_cache: dict[str, list[Token]] = {}
        self._sync_http: httpx.Client | None = None
        self._async_http: httpx.AsyncClient | None = None

    @property
    def supports_bare_label_constraint(self) -> bool:
        """True -- GBNF grammar generates bare label text."""
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
        """Apply GBNF grammar constraint for bare-label generation.

        Builds a grammar rule that allows exactly one of the provided labels::

            root ::= "label1" | "label2" | "label3"
        """
        quoted = [f'"{lbl}"' for lbl in labels]
        body["grammar"] = f"root ::= {' | '.join(quoted)}"

    @staticmethod
    def _label_token_logprobs(
        logprobs: List[TokenLogprob],
    ) -> List[TokenLogprob]:
        """Extract label tokens from a bare-label constrained response.

        For llama.cpp (bare-label backend), this filters out special /
        end-of-sequence tokens and empty strings. The GBNF grammar constraint
        guarantees only the label text is generated, so no JSON-structure
        filtering is needed.
        """
        return [
            lp
            for lp in logprobs
            if lp.token.strip() and lp.token not in LlamaCppBackend._SPECIAL_TOKENS
        ]

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
        """Synchronous constrained chat completion via llama.cpp server."""
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
        """Score a completion by forcing it as the single valid label.

        Forces ``completion`` as the only valid choice via a GBNF grammar
        constraint and reads back the model's genuine per-token logprobs
        (teacher forcing, pre-mask).

        llama.cpp does not support ``echo=True`` on the completions endpoint
        (it only returns generated-token logprobs, not prompt tokens), so the
        echo/prefill approach used by SGLang/vLLM is not available here.
        """
        response = self.chat(
            messages=messages,
            temperature=0.0,
            constrain_labels=[completion],
            logprobs=True,
            top_logprobs=1,
        )
        lps = self._label_token_logprobs(response.logprobs or [])
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

        Forces ``text`` as the only valid label in a GBNF grammar constrained
        :meth:`chat` call and reads back the emitted value tokens. This is
        necessary because standalone BPE tokenization (via ``/tokenize``)
        produces different token boundaries than the model emits under grammar
        guidance, which would break trie-based divergence scoring. Results are
        memoized per label.
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
        """Async constrained chat completion via llama.cpp server."""
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
        """Async completion scoring via forced constrained generation.

        See :meth:`score` for the rationale.
        """
        response = await self.achat(
            messages=messages,
            temperature=0.0,
            constrain_labels=[completion],
            logprobs=True,
            top_logprobs=1,
        )
        lps = self._label_token_logprobs(response.logprobs or [])
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
        lps = self._label_token_logprobs(response.logprobs or [])
        tokens = [Token(text=lp.token, id=-1) for lp in lps]
        tokens = tokens or [Token(text=text, id=-1)]
        self._token_cache[text] = tokens
        return tokens
