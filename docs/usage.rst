Usage
=====

This section covers the common usage patterns for ollama-classifier.

Every backend is used the same way: instantiate a backend, wrap it in an
:class:`~ollama_classifier.classifier.LLMClassifier`, and call
``generate()`` or ``classify()``.

.. code-block:: python

   from ollama_classifier import LLMClassifier
   from ollama_classifier.backends import OllamaBackend

   backend = OllamaBackend(model="llama3.2")
   classifier = LLMClassifier(backend)

Basic Classification
--------------------

The simplest way to classify text. ``classify()`` makes one completion-scoring
call per label and returns calibrated confidence via geometric-mean
normalization.

.. code-block:: python

   result = classifier.classify(
       text="The goalkeeper made an incredible save!",
       choices=["sports", "politics", "technology", "entertainment"],
   )

   print(f"Prediction: {result.prediction}")
   print(f"Confidence: {result.confidence:.2%}")
   print(f"Method: {result.method}")  # "multi_call"

Classification with Label Descriptions
--------------------------------------

Providing descriptions helps the model understand each category better:

.. code-block:: python

   choices = {
       "positive": "Text expresses happiness, satisfaction, or approval",
       "negative": "Text expresses anger, disappointment, or disapproval",
       "mixed": "Text contains both positive and negative sentiments",
       "neutral": "Text is factual without strong emotional content",
   }

   result = classifier.classify(
       text="The food was amazing but the service was terrible.",
       choices=choices,
   )

Custom System Prompt
--------------------

Override the default system prompt for specialized tasks:

.. code-block:: python

   result = classifier.classify(
       text="The quarterly earnings exceeded analyst expectations.",
       choices=["bullish", "bearish", "neutral"],
       system_prompt="You are a financial sentiment analyzer. "
                     "Classify financial news based on market sentiment."
   )

Choosing a Scoring Method
-------------------------

``LLMClassifier`` provides two scoring methods. Each returns a
:class:`~ollama_classifier.types.ClassificationResult` with a full
probability distribution.

+-------------------+-----------------------------------------------------+-------------------------------------------------------------+
|                   | ``generate()``                                      | ``classify()``                                              |
+===================+=====================================================+=============================================================+
| How it works      | Adaptive constrained generation over a prefix trie  | Multi-call completion scoring: each label scored as a       |
|                   | of label tokens; per-label logprobs reconstructed   | completion of the prompt without generation.                |
|                   | from the winning path and unresolved clusters.      |                                                             |
+-------------------+-----------------------------------------------------+-------------------------------------------------------------+
| API calls         | 1 to ``max_calls`` (adaptive)                       | N calls for N labels                                        |
+-------------------+-----------------------------------------------------+-------------------------------------------------------------+
| Confidence        | Divergence-aware (may be partial for labels that    | Exact (geometric-mean normalization)                        |
|                   | diverge early from the winning path)                |                                                             |
+-------------------+-----------------------------------------------------+-------------------------------------------------------------+
| Approximate?      | ``result.approximate`` is ``True`` when any label   | Always ``False`` — fully resolved                           |
|                   | has partial coverage                                |                                                             |
+-------------------+-----------------------------------------------------+-------------------------------------------------------------+
| ``result.method`` | ``"adaptive_generate"``                             | ``"multi_call"``                                            |
+-------------------+-----------------------------------------------------+-------------------------------------------------------------+
| Best for          | Speed, large label sets, when a single call         | Gold-standard accuracy, small-to-medium label sets          |
|                   | suffices                                            |                                                             |
+-------------------+-----------------------------------------------------+-------------------------------------------------------------+

Adaptive Generation (``generate``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``generate()`` makes 1 to ``max_calls`` constrained API calls. Each call walks
a prefix trie of label tokens. After each call, labels are scored up to their
divergence point from the winning path. Unresolved clusters trigger
supplementary calls (recursive cluster resolution) until the budget is
exhausted.

The ``max_calls`` parameter controls the accuracy/cost tradeoff:

+---------------------+-----------------------------------------------------------+---------+
| ``max_calls``       | Behavior                                                  | Calls   |
+=====================+===========================================================+=========+
| ``1`` *(default)*   | Single constrained call. Labels are scored up to their    | 1       |
|                     | divergence point. Fast but approximate.                   |         |
+---------------------+-----------------------------------------------------------+---------+
| ``K``               | Adaptive resolution. Unresolved label clusters trigger    | ≤ K     |
|                     | supplementary calls until the budget is exhausted.        |         |
+---------------------+-----------------------------------------------------------+---------+
| ``None``            | Fully recursive resolution. Every cluster is resolved to  | ≤ N     |
|                     | completion. Equivalent to exact scoring.                  |         |
+---------------------+-----------------------------------------------------------+---------+

.. code-block:: python

   # Fast: single call, approximate confidence
   result = classifier.generate(
       text="The team won the championship!",
       choices=["sports", "finance", "politics"],
       max_calls=1,
   )
   print(result.prediction)   # "sports"
   print(result.approximate)  # True (if any label had partial coverage)
   print(result.coverage)     # {"sports": 1.0, "finance": 1.0, "politics": 1.0}

   # Adaptive: allow up to 3 calls for better resolution
   result = classifier.generate(text="...", choices=[...], max_calls=3)

   # Exact: resolve everything (unlimited calls)
   result = classifier.generate(text="...", choices=[...], max_calls=None)

Multi-Call Scoring (``classify``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Makes one completion-scoring call per label. Each label's per-token logprobs
are extracted *without generation*, then normalized via geometric mean to
eliminate token-count bias. This is the gold-standard confidence method.

.. code-block:: python

   result = classifier.classify(
       text="The movie was fantastic!",
       choices=["positive", "negative", "neutral"],
   )
   print(result.method)        # "multi_call"
   print(result.approximate)   # False
   print(result.n_calls)       # 3

Batch Processing
----------------

Classify multiple texts efficiently. Batch methods are parallelized via a
thread pool (sync) or ``asyncio.gather`` (async):

.. code-block:: python

   texts = [
       "The goalkeeper made an incredible save!",
       "The central bank raised interest rates.",
       "The new smartphone features a revolutionary camera.",
   ]

   results = classifier.batch_classify(
       texts=texts,
       choices=["sports", "finance", "technology"],
   )

   for text, result in zip(texts, results):
       print(f"{text} -> {result.prediction} ({result.confidence:.2%})")

Async Usage
-----------

For concurrent processing, use the async methods:

.. code-block:: python

   import asyncio
   from ollama_classifier import LLMClassifier
   from ollama_classifier.backends import OllamaBackend

   async def main():
       backend = OllamaBackend(model="llama3.2")
       classifier = LLMClassifier(backend)

       # Single classification
       result = await classifier.aclassify(
           text="The concert was amazing!",
           choices=["positive", "negative", "neutral"],
       )

       # Batch classification (concurrent)
       results = await classifier.abatch_classify(
           texts=["Text 1", "Text 2", "Text 3"],
           choices=["positive", "negative", "neutral"],
       )

   asyncio.run(main())

Working with Results
--------------------

The :class:`~ollama_classifier.types.ClassificationResult` object contains:

.. code-block:: python

   result = classifier.classify(
       text="I love this product!",
       choices=["positive", "negative", "neutral"],
   )

   # The predicted label
   print(result.prediction)  # "positive"

   # Confidence score (0.0 to 1.0)
   print(result.confidence)  # 0.92

   # Full probability distribution
   print(result.probabilities)
   # {"positive": 0.92, "negative": 0.05, "neutral": 0.03}

   # Scoring method used
   print(result.method)  # "multi_call" or "adaptive_generate"

   # Whether the result is approximate (only for generate())
   print(result.approximate)  # False for classify()

   # Per-label fraction of tokens scored (only for generate())
   print(result.coverage)  # {"positive": 1.0, ...}

   # Number of API calls made
   print(result.n_calls)  # 3

   # Raw data for debugging
   print(result.raw_response)

Using Sample Data
-----------------

The package ships with ready-to-use sample datasets in
``examples/sample_data.py`` so you can test the classifier immediately.
Two datasets are provided, both containing 20 short customer-support
ticket texts across four labels (``billing``, ``technical_support``,
``account``, ``general``):

- ``DATASET_WITHOUT_DESCRIPTIONS`` — labels as a plain ``list``.
- ``DATASET_WITH_DESCRIPTIONS`` — labels as a ``dict`` mapping each
  label to a human-readable description, which typically improves
  classification accuracy.

Each dataset is a :class:`~dataclasses.dataclass` with four fields:
``texts``, ``choices``, ``expected_labels``, and ``description``.

Import the datasets:

.. code-block:: python

   from examples.sample_data import (
       DATASET_WITHOUT_DESCRIPTIONS,
       DATASET_WITH_DESCRIPTIONS,
   )

Quick test with the basic dataset:

.. code-block:: python

   from ollama_classifier import LLMClassifier
   from ollama_classifier.backends import OllamaBackend
   from examples.sample_data import DATASET_WITHOUT_DESCRIPTIONS

   backend = OllamaBackend(model="llama3.2")
   classifier = LLMClassifier(backend)

   results = classifier.batch_generate(
       texts=DATASET_WITHOUT_DESCRIPTIONS.texts,
       choices=DATASET_WITHOUT_DESCRIPTIONS.choices,
   )

   correct = sum(r.prediction == e for r, e in
                   zip(results, DATASET_WITHOUT_DESCRIPTIONS.expected_labels))
   print(f"Accuracy: {correct}/{len(results)}")

Run the bundled example script directly:

.. code-block:: bash

   python -m examples.run_sample_data
