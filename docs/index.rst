.. ollama-classifier documentation master file

Welcome to ollama-classifier's documentation!
=============================================

A Python wrapper around the Ollama Python SDK for text classification with constrained output and confidence scoring. **Supports multiple inference backends**: Ollama, vLLM, SGLang, and llama.cpp.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   backends
   api
   changelog

Features
--------

- **Constrained Output**: Uses JSON schema with enum constraints to ensure only valid choices are generated
- **Confidence Scoring**: Multi-call evaluation with softmax for calibrated probabilities
- **Sync & Async**: Full support for both synchronous and asynchronous operations
- **Batch Processing**: Classify multiple texts efficiently
- **Flexible Choices**: Support for simple labels or labels with descriptions
- **Custom Prompts**: Override the default system prompt for specialized tasks
- **Multiple Backends**: Use Ollama, vLLM, SGLang, or llama.cpp as your inference engine

Quick Start
-----------

**Ollama** (default backend):

.. code-block:: python

   from ollama import Client
   from ollama_classifier import OllamaClassifier

   client = Client()
   classifier = OllamaClassifier(client, model="llama3.2")

   result = classifier.classify(
       text="I love this product!",
       choices=["positive", "negative", "neutral"]
   )

   print(f"Prediction: {result.prediction}")
   print(f"Confidence: {result.confidence:.2%}")

**vLLM** backend:

.. code-block:: python

   from ollama_classifier.backends import VLLMBackend
   from ollama_classifier import LLMClassifier

   backend = VLLMBackend(model="meta-llama/Llama-3.2-3B-Instruct")
   classifier = LLMClassifier(backend)

   result = classifier.classify(
       text="I love this product!",
       choices=["positive", "negative", "neutral"]
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
