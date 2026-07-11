.. ollama-classifier documentation master file

Welcome to ollama-classifier's documentation!
=============================================

A Python library for LLM-based text classification with constrained output
and confidence scoring. **Supports multiple inference backends**: Ollama
(≥0.12), vLLM, SGLang, and llama.cpp — all behind a single unified
:class:`~ollama_classifier.classifier.LLMClassifier`.

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

- **Two Scoring Methods**: ``generate()`` for adaptive budget-controlled scoring, ``classify()`` for exact gold-standard confidence
- **Constrained Output**: Output is guaranteed to be one of your labels (JSON enum, ``structured_outputs.choice``, regex, or GBNF — depending on backend)
- **Calibrated Confidence**: Probability distribution over all choices with geometric-mean normalization (no token-count bias)
- **Sync & Async**: Full support for both synchronous and asynchronous operations
- **Batch Processing**: Classify multiple texts efficiently with parallel execution
- **Flexible Choices**: Support for simple labels or labels with descriptions
- **Custom Prompts**: Override the default system prompt for specialized tasks
- **Multiple Backends**: Use Ollama, vLLM, SGLang, or llama.cpp as your inference engine

Quick Start
-----------

All backends follow the same pattern: create a backend, wrap it in an
``LLMClassifier``, and call ``generate()`` or ``classify()``.

**Ollama** backend:

.. code-block:: python

   from ollama_classifier import LLMClassifier
   from ollama_classifier.backends import OllamaBackend

   backend = OllamaBackend(model="llama3.2")
   classifier = LLMClassifier(backend)

   result = classifier.classify(
       text="I love this product!",
       choices=["positive", "negative", "neutral"]
   )

   print(f"Prediction: {result.prediction}")
   print(f"Confidence: {result.confidence:.2%}")

**vLLM** backend:

.. code-block:: python

   from ollama_classifier import LLMClassifier
   from ollama_classifier.backends import VLLMBackend

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
