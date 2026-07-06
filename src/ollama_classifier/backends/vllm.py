"""vLLM inference backend.

Supports both local and remote vLLM servers via the OpenAI-compatible API.

vLLM supports ``guided_choice`` natively, which constrains the model to
generate exactly one of the provided label strings — **bare label text**
with no JSON wrapper. This makes logprob reconstruction clean and exact.

Local server::

    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --host 0.0.0.0 --port 8000

Connect::

    backend = VLLMBackend(
        model="meta-llama/Llama-3.2-3B-Instruct",
        base_url="http://localhost:8000/v1",
    )
"""

import re
from typing import Any, Dict, List, Optional

import httpx

from .base import ChatMessage, ChatResponse, LLMBackend, ScoringResponse, Token, TokenLogprob


class VLLMBackend(LLMBackend):
    """Backend for vLLM inference server.

    vLLM provides a high-throughput serving engine with an OpenAI-compatible
    API endpoint. It supports guided decoding (``guided_choice``) and logprob
    return out of the box. Logprobs are pre-mask (raw model logits before
    guided decoding masking).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
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
        """True — vLLM ``guided_choice`` generates bare label text."""
        return True

    def _apply_constraint(self, body: Dict[str, Any], labels: List[str]) -> None:
        """Apply ``guided_choice`` constraint (vLLM native)."""
        body["guided_choice"] = labels

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
        """Synchronous constrained chat completion via vLLM."""
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
            # vLLM guided_choice returns bare label text
            result.label = result.content.strip()
            return result

    def score(
        self,
        messages: List[ChatMessage],
        completion: str,
    ) -> ScoringResponse:
        """Score a completion using vLLM's completions endpoint.

        Uses ``/v1/completions`` with ``prompt_logprobs`` to compute the
        per-token logprobs of the completion given the prompt context.
        """
        prompt = self._render_prompt(messages)
        url = f"{self._base_url}/completions"
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt + completion,
            "echo": True,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 0,
            **self._extra_body,
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        # Extract logprobs only for the completion tokens (skip prompt tokens)
        choice = data["choices"][0]
        all_logprobs = choice.get("logprobs", [])
        prompt_token_count = len(self._tokenize_ids(prompt))
        completion_lps = all_logprobs[prompt_token_count:]

        token_logprobs: list[TokenLogprob] = []
        for lp_entry in completion_lps:
            if lp_entry is None:
                continue
            top: dict[str, float] = {}
            for alt in lp_entry.get("top_logprobs", []):
                top[alt["token"]] = alt["logprob"]
            token_logprobs.append(
                TokenLogprob(
                    token=lp_entry.get("tokens", ""),
                    logprob=lp_entry.get("token_logprobs", 0.0),
                    top_logprobs=top,
                )
            )

        return ScoringResponse(completion=completion, logprobs=token_logprobs, raw=data)

    def tokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Tokenize text using vLLM's tokenize endpoint.

        vLLM provides a ``/tokenize`` endpoint (non-standard but supported
        in recent versions).
        """
        full_text = (context or "") + text
        url = f"{self._base_url}/tokenize"
        body = {"model": self._model, "prompt": full_text}
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        token_ids: list[int] = data.get("tokens", [])
        if context:
            ctx_body = {"model": self._model, "prompt": context}
            with httpx.Client(timeout=self._timeout) as client:
                ctx_resp = client.post(url, headers=self._build_headers(), json=ctx_body)
                ctx_resp.raise_for_status()
                ctx_data = ctx_resp.json()
            token_ids = token_ids[len(ctx_data.get("tokens", [])):]

        return [Token(text="", id=t) for t in token_ids]

    def _tokenize_ids(self, text: str) -> list[int]:
        """Return token IDs for text (used for prompt/completion boundary)."""
        try:
            url = f"{self._base_url}/tokenize"
            body = {"model": self._model, "prompt": text}
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, headers=self._build_headers(), json=body)
                resp.raise_for_status()
                return resp.json().get("tokens", [])
        except Exception:
            return []

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
        """Async constrained chat completion via vLLM."""
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
        """Async completion scoring via vLLM."""
        prompt = self._render_prompt(messages)
        url = f"{self._base_url}/completions"
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt + completion,
            "echo": True,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 0,
            **self._extra_body,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        all_logprobs = choice.get("logprobs", [])
        prompt_token_count = len(self._tokenize_ids(prompt))
        completion_lps = all_logprobs[prompt_token_count:]

        token_logprobs: list[TokenLogprob] = []
        for lp_entry in completion_lps:
            if lp_entry is None:
                continue
            top: dict[str, float] = {}
            for alt in lp_entry.get("top_logprobs", []):
                top[alt["token"]] = alt["logprob"]
            token_logprobs.append(
                TokenLogprob(
                    token=lp_entry.get("tokens", ""),
                    logprob=lp_entry.get("token_logprobs", 0.0),
                    top_logprobs=top,
                )
            )

        return ScoringResponse(completion=completion, logprobs=token_logprobs, raw=data)

    async def atokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Async tokenization via vLLM."""
        full_text = (context or "") + text
        url = f"{self._base_url}/tokenize"
        body = {"model": self._model, "prompt": full_text}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        token_ids: list[int] = data.get("tokens", [])
        if context:
            ctx_body = {"model": self._model, "prompt": context}
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                ctx_resp = await client.post(url, headers=self._build_headers(), json=ctx_body)
                ctx_resp.raise_for_status()
                ctx_data = ctx_resp.json()
            token_ids = token_ids[len(ctx_data.get("tokens", [])):]

        return [Token(text="", id=t) for t in token_ids]
