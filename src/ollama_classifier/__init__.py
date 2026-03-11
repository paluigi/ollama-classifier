"""Ollama Classifier - A wrapper around Ollama Python SDK for text classification.

This package provides a classifier for text classification with constrained output
and confidence scoring using the Ollama Python SDK.

Example usage:
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
"""

from .types import ClassificationResult, ChoicesType
from .classifier import OllamaClassifier

__all__ = [
    "OllamaClassifier",
    "ClassificationResult",
    "ChoicesType",
]

__version__ = "0.1.0"