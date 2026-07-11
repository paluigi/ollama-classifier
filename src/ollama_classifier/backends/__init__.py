"""Inference engine backends for ollama-classifier.

Each backend implements the :class:`LLMBackend` protocol and communicates with
its respective inference engine. ``OllamaBackend`` uses the native Ollama SDK;
``VLLMBackend``, ``SGLangBackend``, and ``LlamaCppBackend`` communicate via the
OpenAI-compatible API.

Each backend translates the high-level ``constrain_labels`` parameter to its
native constraint mechanism:

- **Ollama**: JSON Schema enum via ``format``
- **vLLM**: ``structured_outputs.choice``
- **SGLang**: ``regex``
- **llama.cpp**: GBNF ``grammar``
"""

from .base import (
    ChatMessage,
    ChatResponse,
    LLMBackend,
    ScoringResponse,
    Token,
    TokenLogprob,
)
from .llamacpp import LlamaCppBackend
from .ollama import OllamaBackend
from .sglang import SGLangBackend
from .vllm import VLLMBackend

__all__ = [
    "LLMBackend",
    "ChatMessage",
    "ChatResponse",
    "ScoringResponse",
    "TokenLogprob",
    "Token",
    "OllamaBackend",
    "VLLMBackend",
    "SGLangBackend",
    "LlamaCppBackend",
]
