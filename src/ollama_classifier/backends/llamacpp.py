"""llama.cpp inference backend.

Supports both local and remote ``llama-server`` instances via the OpenAI-compatible API.

llama.cpp supports GBNF (GGML BNF) grammar constraints, which can express bare
label alternatives directly::

    root ::= "positive" | "negative" | "neutral"

This generates **bare label text** — no JSON wrapper — so logprob
reconstruction is clean. ``llama-server`` accepts a non-standard ``grammar``
field on the ``/v1/chat/completions`` endpoint.

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
    API. GBNF grammar constraints and logprobs are supported when compiled
    with the appropriate flags. Logprobs are pre-mask (model's raw distribution
    before grammar masking).

    Note:
        ``response_format`` with JSON schema on ``/v1/chat/completions`` is buggy
        in llama.cpp (GitHub issues #11988, #11847). This backend uses the
        non-standard ``grammar`` field with GBNF instead, which works reliably
        for bare label generation.
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
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )

    @property
    def supports_bare_label_constraint(self) -> bool:
        """True — GBNF grammar generates bare label text."""
        return True

    def _apply_constraint(self, body: Dict[str, Any], labels: List[str]) -> None:
        """Apply GBNF grammar constraint for bare-label generation.

        Builds a grammar rule that allows exactly one of the provided labels::

            root ::= "label1" | "label2" | "label3"
        """
        # Escape quotes in labels for GBNF
        quoted = [f'"{l}"' for l in labels]
        body["grammar"] = f"root ::= {' | '.join(quoted)}"

    def _render_prompt(self, messages: List[ChatMessage]) -> str:
        """Render messages to a plain text prompt for the completions endpoint."""
        parts: list[str] = []
        for m in messages:
            if m.role == "system":
                parts.append(f"<|system|>\n{m.content}")
            elif m.role == "user":
                parts.append(f"<|user|>\n{m.content}")
        return "\n\n".join(parts) + "\n\n<|assistant|>\n"

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
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            result = self._parse_chat_response(resp.json())
            result.label = result.content.strip()
            return result

    def score(
        self,
        messages: List[ChatMessage],
        completion: str,
    ) -> ScoringResponse:
        """Score a completion using llama.cpp's completions endpoint with suffix.

        Uses ``/v1/completions`` with ``suffix`` to compute the per-token
        logprobs of the completion given the prompt context (fill-in-the-middle
        style scoring).
        """
        prompt = self._render_prompt(messages)
        url = f"{self._base_url}/completions"
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "suffix": completion,
            "max_tokens": 0,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
            **self._extra_body,
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        all_logprobs = choice.get("logprobs", {})
        tokens_list = all_logprobs.get("tokens", [])
        token_lps_list = all_logprobs.get("token_logprobs", [])
        top_lps_list = all_logprobs.get("top_logprobs", [])

        # Find where the completion starts in the token list
        # With suffix + echo, llama.cpp returns prompt tokens followed by completion tokens
        # The completion tokens are at the end
        # We identify them by matching the completion text
        completion_start = self._find_completion_start(tokens_list, completion)

        token_logprobs: list[TokenLogprob] = []
        for i in range(completion_start, len(tokens_list)):
            top: dict[str, float] = {}
            if i < len(top_lps_list) and top_lps_list[i]:
                for t, lp in top_lps_list[i].items():
                    top[t] = lp
            lp = token_lps_list[i] if i < len(token_lps_list) else 0.0
            token_logprobs.append(
                TokenLogprob(token=tokens_list[i], logprob=lp or 0.0, top_logprobs=top)
            )

        return ScoringResponse(completion=completion, logprobs=token_logprobs, raw=data)

    @staticmethod
    def _find_completion_start(tokens: list[str], completion: str) -> int:
        """Heuristically find where the completion tokens begin.

        With suffix + echo, llama.cpp returns [prompt_tokens, completion_tokens].
        We search for the suffix position by accumulating token text until it
        matches the start of the completion string.
        """
        if not tokens:
            return 0
        accumulated = ""
        for i, tok in enumerate(tokens):
            accumulated += tok
            if completion.startswith(accumulated.lstrip()):
                return i
        return len(tokens)

    def tokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Tokenize text using llama.cpp's tokenize endpoint.

        llama-server provides a ``/tokenize`` endpoint.
        """
        full_text = (context or "") + text
        url = f"{self._base_url}/tokenize"
        body = {"content": full_text}
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        tokens_data: list = data.get("tokens", [])
        if context:
            ctx_body = {"content": context}
            with httpx.Client(timeout=self._timeout) as client:
                ctx_resp = client.post(url, headers=self._build_headers(), json=ctx_body)
                ctx_resp.raise_for_status()
                ctx_data = ctx_resp.json()
            tokens_data = tokens_data[len(ctx_data.get("tokens", [])):]

        return [
            Token(
                text=t if isinstance(t, str) else "",
                id=t if isinstance(t, int) else -1,
            )
            for t in tokens_data
        ]

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
        async with httpx.AsyncClient(timeout=self._timeout) as client:
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
        """Async completion scoring via llama.cpp server."""
        prompt = self._render_prompt(messages)
        url = f"{self._base_url}/completions"
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "suffix": completion,
            "max_tokens": 0,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
            **self._extra_body,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        all_logprobs = choice.get("logprobs", {})
        tokens_list = all_logprobs.get("tokens", [])
        token_lps_list = all_logprobs.get("token_logprobs", [])
        top_lps_list = all_logprobs.get("top_logprobs", [])

        completion_start = self._find_completion_start(tokens_list, completion)

        token_logprobs: list[TokenLogprob] = []
        for i in range(completion_start, len(tokens_list)):
            top: dict[str, float] = {}
            if i < len(top_lps_list) and top_lps_list[i]:
                for t, lp in top_lps_list[i].items():
                    top[t] = lp
            lp = token_lps_list[i] if i < len(token_lps_list) else 0.0
            token_logprobs.append(
                TokenLogprob(token=tokens_list[i], logprob=lp or 0.0, top_logprobs=top)
            )

        return ScoringResponse(completion=completion, logprobs=token_logprobs, raw=data)

    async def atokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Async tokenization via llama.cpp server."""
        full_text = (context or "") + text
        url = f"{self._base_url}/tokenize"
        body = {"content": full_text}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        tokens_data: list = data.get("tokens", [])
        if context:
            ctx_body = {"content": context}
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                ctx_resp = await client.post(url, headers=self._build_headers(), json=ctx_body)
                ctx_resp.raise_for_status()
                ctx_data = ctx_resp.json()
            tokens_data = tokens_data[len(ctx_data.get("tokens", [])):]

        return [
            Token(
                text=t if isinstance(t, str) else "",
                id=t if isinstance(t, int) else -1,
            )
            for t in tokens_data
        ]
