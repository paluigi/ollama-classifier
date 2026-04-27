"""Ollama Classifier - A wrapper around Ollama Python SDK for text classification.

This package provides classifiers for text classification with constrained
output and confidence scoring using multiple inference backends.

Ollama backend (original)::

    from ollama import Client
    from ollama_classifier import OllamaClassifier, ClassificationResult

    client = Client()
    classifier = OllamaClassifier(client, model="llama3.2")

    result = classifier.classify(
        text="I love this product!",
        choices=["positive", "negative", "neutral"]
    )
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")

Generic backend (vLLM, SGLang, llama.cpp)::

    from ollama_classifier.backends import VLLMBackend
    from ollama_classifier import LLMClassifier

    backend = VLLMBackend(model="meta-llama/Llama-3.2-3B-Instruct")
    classifier = LLMClassifier(backend)

    result = classifier.classify(
        text="I love this product!",
        choices=["positive", "negative", "neutral"]
    )
"""

from .types import ClassificationResult, ChoicesType
from .classifier import OllamaClassifier
from .llm_classifier import LLMClassifier

__all__ = [
    "OllamaClassifier",
    "LLMClassifier",
    "ClassificationResult",
    "ChoicesType",
]

__version__ = "0.3.0"
