"""SGLang inference backend.

Supports both local and remote SGLang servers via the OpenAI-compatible API.

SGLang supports regex constraints for bare-label generation, producing clean
label text with no JSON wrapper.

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
        """True — SGLang regex constraint generates bare label text."""
        return True

    def _apply_constraint(self, body: Dict[str, Any], labels: List[str]) -> None:
        """Apply regex constraint for bare-label generation."""
        escaped = [re.escape(l) for l in labels]
        body["regex"] = f"({'|'.join(escaped)})"

    def _render_prompt(self, messages: List[ChatMessage]) -> str:
        """Render messages to a plain text prompt."""
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
        """Synchronous constrained chat completion via SGLang."""
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
        """Score a completion using SGLang's completions endpoint."""
        prompt = self._render_prompt(messages)
        url = f"{self._base_url}/completions"
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt + completion,
            "echo": True,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
            **self._extra_body,
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        all_logprobs = choice.get("logprobs", {})
        # SGLang completions format: {tokens: [...], token_logprobs: [...], top_logprobs: [...]}
        tokens_list = all_logprobs.get("tokens", [])
        token_lps_list = all_logprobs.get("token_logprobs", [])
        top_lps_list = all_logprobs.get("top_logprobs", [])

        # Determine prompt token count to skip
        prompt_tokens = self._tokenize_count(prompt)
        completion_tokens = tokens_list[prompt_tokens:]
        completion_lps = token_lps_list[prompt_tokens:]
        completion_top = top_lps_list[prompt_tokens:]

        token_logprobs: list[TokenLogprob] = []
        for i, tok in enumerate(completion_tokens):
            top: dict[str, float] = {}
            if i < len(completion_top) and completion_top[i]:
                for t, lp in completion_top[i].items():
                    top[t] = lp
            lp = completion_lps[i] if i < len(completion_lps) else 0.0
            token_logprobs.append(TokenLogprob(token=tok, logprob=lp or 0.0, top_logprobs=top))

        return ScoringResponse(completion=completion, logprobs=token_logprobs, raw=data)

    def _tokenize_count(self, text: str) -> int:
        """Count tokens using SGLang's tokenize endpoint."""
        try:
            url = f"{self._base_url}/tokenize"
            body = {"model": self._model, "text": text}
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, headers=self._build_headers(), json=body)
                resp.raise_for_status()
                return len(resp.json().get("tokens", []))
        except Exception:
            return 0

    def tokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Tokenize text using SGLang's tokenize endpoint."""
        full_text = (context or "") + text
        url = f"{self._base_url}/tokenize"
        body = {"model": self._model, "text": full_text}
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        token_ids: list[int] = data.get("tokens", [])
        if context:
            ctx_body = {"model": self._model, "text": context}
            with httpx.Client(timeout=self._timeout) as client:
                ctx_resp = client.post(url, headers=self._build_headers(), json=ctx_body)
                ctx_resp.raise_for_status()
                ctx_data = ctx_resp.json()
            token_ids = token_ids[len(ctx_data.get("tokens", [])):]

        return [Token(text="", id=t) for t in token_ids]

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
        """Async completion scoring via SGLang."""
        prompt = self._render_prompt(messages)
        url = f"{self._base_url}/completions"
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt + completion,
            "echo": True,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 1,
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

        prompt_tokens = await self._atokenize_count(prompt)
        completion_tokens = tokens_list[prompt_tokens:]
        completion_lps = token_lps_list[prompt_tokens:]
        completion_top = top_lps_list[prompt_tokens:]

        token_logprobs: list[TokenLogprob] = []
        for i, tok in enumerate(completion_tokens):
            top: dict[str, float] = {}
            if i < len(completion_top) and completion_top[i]:
                for t, lp in completion_top[i].items():
                    top[t] = lp
            lp = completion_lps[i] if i < len(completion_lps) else 0.0
            token_logprobs.append(TokenLogprob(token=tok, logprob=lp or 0.0, top_logprobs=top))

        return ScoringResponse(completion=completion, logprobs=token_logprobs, raw=data)

    async def _atokenize_count(self, text: str) -> int:
        try:
            url = f"{self._base_url}/tokenize"
            body = {"model": self._model, "text": text}
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=self._build_headers(), json=body)
                resp.raise_for_status()
                return len(resp.json().get("tokens", []))
        except Exception:
            return 0

    async def atokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Async tokenization via SGLang."""
        full_text = (context or "") + text
        url = f"{self._base_url}/tokenize"
        body = {"model": self._model, "text": full_text}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=self._build_headers(), json=body)
            resp.raise_for_status()
            data = resp.json()

        token_ids: list[int] = data.get("tokens", [])
        if context:
            ctx_body = {"model": self._model, "text": context}
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                ctx_resp = await client.post(url, headers=self._build_headers(), json=ctx_body)
                ctx_resp.raise_for_status()
                ctx_data = ctx_resp.json()
            token_ids = token_ids[len(ctx_data.get("tokens", [])):]

        return [Token(text="", id=t) for t in token_ids]
