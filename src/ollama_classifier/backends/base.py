"""Base backend protocol for inference engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str
    content: str


@dataclass
class ChatResponse:
    """Response from a chat completion call."""

    content: str
    logprobs: Optional[List[Dict[str, Any]]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    """Abstract base class for LLM inference backends.

    All backends communicate via HTTP using the OpenAI-compatible chat
    completions API (which vLLM, SGLang, and llama.cpp server all support).
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the backend.

        Args:
            model: The model identifier to use.
            base_url: Base URL of the inference server.
            api_key: Optional API key. Defaults to "not-needed".
            timeout: Request timeout in seconds.
            extra_body: Extra parameters merged into every request body.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or "not-needed"
        self._timeout = timeout
        self._extra_body = extra_body or {}

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        guided_json: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Perform a synchronous chat completion.

        Args:
            messages: List of chat messages.
            temperature: Sampling temperature.
            guided_json: Optional JSON schema for structured output.
            logprobs: Whether to return log probabilities.
            top_logprobs: Number of top log probabilities per token.

        Returns:
            ChatResponse with content and optional logprobs.
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
        guided_json: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> ChatResponse:
        """Perform an asynchronous chat completion."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _build_body(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.0,
        guided_json: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> Dict[str, Any]:
        """Build the request body for the OpenAI-compatible API."""
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }

        if guided_json is not None:
            body["guided_json"] = guided_json
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "classification",
                    "schema": guided_json,
                    "strict": True,
                },
            }

        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs

        body.update(self._extra_body)
        return body

    def _parse_response(self, data: Dict[str, Any]) -> ChatResponse:
        """Parse the JSON response from the OpenAI-compatible API."""
        choice = data["choices"][0]
        content = choice["message"].get("content", "")

        logprobs_list: Optional[List[Dict[str, Any]]] = None
        if choice.get("logprobs") and choice["logprobs"].get("content"):
            logprobs_list = []
            for token_info in choice["logprobs"]["content"]:
                logprobs_list.append(
                    {
                        "token": token_info["token"],
                        "logprob": token_info["logprob"],
                        "top_logprobs": token_info.get("top_logprobs", []),
                    }
                )

        return ChatResponse(content=content, logprobs=logprobs_list, raw=data)
