.. ollama-classifier documentation master file

Welcome to ollama-classifier's documentation!
=============================================

A Python wrapper around the Ollama Python SDK for text classification with constrained output and confidence scoring.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   changelog

Features
--------

- **Constrained Output**: Uses JSON schema with enum constraints to ensure only valid choices are generated
- **Confidence Scoring**: Two methods available:
  - **Fast**: Single API call with logprob extraction
  - **Complete**: Multi-call evaluation with softmax for calibrated probabilities
- **Sync & Async**: Full support for both synchronous and asynchronous operations
- **Batch Processing**: Classify multiple texts efficiently
- **Flexible Choices**: Support for simple labels or labels with descriptions
- **Custom Prompts**: Override the default system prompt for specialized tasks

Quick Start
-----------

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
   print(f"Probabilities: {result.probabilities}")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`