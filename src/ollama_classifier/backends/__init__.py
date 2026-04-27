"""Inference engine backends for ollama-classifier.

Each backend implements the LLMBackend protocol and communicates with
its respective inference engine via HTTP (OpenAI-compatible API).
"""

from .base import LLMBackend
from .vllm import VLLMBackend
from .sglang import SGLangBackend
from .llamacpp import LlamaCppBackend

__all__ = [
    "LLMBackend",
    "VLLMBackend",
    "SGLangBackend",
    "LlamaCppBackend",
]
