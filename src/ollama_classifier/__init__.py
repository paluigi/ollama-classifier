"""Ollama Classifier - Text classification with constrained output and confidence scoring.

Supports multiple inference backends: Ollama (≥0.12), vLLM, SGLang, and llama.cpp.

Two scoring methods:

- ``generate()``: Adaptive constrained generation with divergence-aware
  confidence. Budget-controlled via ``max_calls``.
- ``classify()``: Multi-call completion scoring with geometric-mean
  normalization. Gold-standard accuracy.

Example::

    from ollama_classifier import LLMClassifier, ClassificationResult
    from ollama_classifier.backends import OllamaBackend

    backend = OllamaBackend(model="llama3.2")
    classifier = LLMClassifier(backend)

    result = classifier.classify(
        text="I love this product!",
        choices=["positive", "negative", "neutral"],
    )
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Method: {result.method}")
"""

from .classifier import LLMClassifier
from .types import ClassificationResult, ChoicesType

__all__ = [
    "LLMClassifier",
    "ClassificationResult",
    "ChoicesType",
]

__version__ = "0.4.1"
