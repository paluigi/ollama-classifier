"""Base backend protocol for inference engines.

All backends communicate via HTTP using the OpenAI-compatible chat completions
API (which vLLM, SGLang, and llama.cpp server all support). Each backend
translates the high-level ``constrain_labels`` parameter to its native
constraint mechanism (``guided_choice``, ``regex``, JSON enum, or GBNF grammar).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str
    content: str


@dataclass
class TokenLogprob:
    """A single token with its log probability and alternatives.

    ``top_logprobs`` follows the OpenAI format: a list of dicts, each with
    ``"token"``, ``"logprob"``, and optionally ``"bytes"`` keys.
    """

    token: str
    token_id: int = -1
    logprob: float = 0.0
    top_logprobs: dict[str, float] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from a constrained generation call."""

    content: str
    label: str = ""
    logprobs: Optional[List[TokenLogprob]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringResponse:
    """Response from a completion-scoring call (for ``classify()``).

    Contains per-token logprobs for the COMPLETION text only, not the prompt.
    """

    completion: str
    logprobs: List[TokenLogprob] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Token:
    """A tokenized unit."""

    text: str
    id: int


class LLMBackend(ABC):
    """Abstract base class for LLM inference backends.

    All backends communicate with their respective inference engine via HTTP
    (OpenAI-compatible API). Each backend translates ``constrain_labels`` to
    its native constraint mechanism and implements ``score()`` for completion
    scoring and ``tokenize()`` for context-dependent tokenization.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "",
        *,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_tokens: int = 256,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the backend.

        Args:
            model: The model identifier to use.
            base_url: Base URL of the inference server.
            api_key: Optional API key. Defaults to ``"not-needed"``.
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens to generate.
            extra_body: Extra parameters merged into every request body.
        """
        self._model = model
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._api_key = api_key or "not-needed"
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._extra_body = extra_body or {}

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    @abstractmethod
    def supports_bare_label_constraint(self) -> bool:
        """True if ``chat()`` generates bare label text (no JSON wrapper).

        ``vLLM``/``SGLang``/``llama.cpp``: ``True``.
        ``Ollama``: ``False`` (uses JSON enum wrapper).
        """

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        constrain_labels: Optional[List[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Perform a synchronous constrained chat completion.

        Each backend translates ``constrain_labels`` to its native constraint
        mechanism (``guided_choice``, ``regex``, JSON enum, or GBNF grammar).

        Args:
            messages: List of chat messages.
            temperature: Sampling temperature.
            constrain_labels: Optional list of valid label strings.
                If provided, output is constrained to exactly one of these.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of top log probabilities per token.

        Returns:
            ``ChatResponse`` with content and optional logprobs.
        """

    @abstractmethod
    def score(
        self,
        messages: List[ChatMessage],
        completion: str,
    ) -> ScoringResponse:
        """Score a completion by computing per-token logprobs of the
        completion text given the message context.

        No generation occurs — the completion is provided and the backend
        returns the model's logprob for each token of the completion.

        Args:
            messages: The chat messages forming the context.
            completion: The completion text to score.

        Returns:
            ``ScoringResponse`` with per-token logprobs.
        """

    @abstractmethod
    def tokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Tokenize text.

        If ``context`` is provided, tokenizes ``context + text`` and returns
        only the tokens corresponding to ``text`` (context-dependent
        tokenization). This is critical for backends that wrap labels in JSON
        (Ollama), where the tokenization of a label inside JSON differs from
        standalone tokenization.

        Args:
            text: The text to tokenize.
            context: Optional prefix to include for context-dependent
                tokenization.

        Returns:
            List of ``Token`` objects.
        """

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def achat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        constrain_labels: Optional[List[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Async constrained chat completion."""

    @abstractmethod
    async def ascore(
        self,
        messages: List[ChatMessage],
        completion: str,
    ) -> ScoringResponse:
        """Async completion scoring."""

    @abstractmethod
    async def atokenize(
        self,
        text: str,
        *,
        context: Optional[str] = None,
    ) -> List[Token]:
        """Async tokenization."""

    # ------------------------------------------------------------------
    # Shared HTTP helpers (for OpenAI-compatible backends)
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _build_chat_body(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        constrain_labels: Optional[List[str]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> Dict[str, Any]:
        """Build request body for OpenAI-compatible ``/v1/chat/completions``.

        Subclasses override ``_apply_constraint()`` to add backend-specific
        constraint fields.
        """
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if constrain_labels is not None:
            self._apply_constraint(body, constrain_labels)
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        body.update(self._extra_body)
        return body

    def _apply_constraint(self, body: Dict[str, Any], labels: List[str]) -> None:
        """Apply backend-specific output constraint.

        Override in subclasses to add ``guided_choice``, ``regex``,
        ``format`` (JSON enum), or ``grammar`` (GBNF) to the request body.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support label constraints"
        )

    @staticmethod
    def _parse_chat_response(data: Dict[str, Any]) -> ChatResponse:
        """Parse OpenAI-compatible chat completion response."""
        choice = data["choices"][0]
        content = choice["message"].get("content", "")

        logprobs_list: Optional[List[TokenLogprob]] = None
        if choice.get("logprobs") and choice["logprobs"].get("content"):
            logprobs_list = []
            for ti in choice["logprobs"]["content"]:
                # Flatten top_logprobs from OpenAI format [{token, logprob}, ...]
                # to {token: logprob} dict for easy lookup during trie reconstruction
                top_lps: dict[str, float] = {}
                for alt in ti.get("top_logprobs", []):
                    top_lps[alt["token"]] = alt["logprob"]
                token_id = -1
                if isinstance(ti.get("bytes"), list) and ti["bytes"]:
                    token_id = ti["bytes"][0]
                logprobs_list.append(
                    TokenLogprob(
                        token=ti["token"],
                        token_id=token_id,
                        logprob=ti["logprob"],
                        top_logprobs=top_lps,
                    )
                )

        return ChatResponse(content=content, logprobs=logprobs_list, raw=data)
