API Reference
=============

This section provides the complete API reference for ollama-classifier.

LLMClassifier
-------------

The single, unified, backend-agnostic classifier. Accepts any
:class:`~ollama_classifier.backends.base.LLMBackend` instance and exposes
two scoring methods: ``generate()`` (hierarchical constrained generation with
reproportion-based cluster resolution) and ``classify()`` (exact multi-call
completion scoring).

.. autoclass:: ollama_classifier.classifier.LLMClassifier
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

ClassificationResult
--------------------

A Pydantic model returned by both ``generate()`` and ``classify()``.

.. autoclass:: ollama_classifier.types.ClassificationResult
   :members:
   :undoc-members:
   :show-inheritance:

ChoicesType
-----------

.. autodata:: ollama_classifier.types.ChoicesType

Backends
--------

.. automodule:: ollama_classifier.backends
   :members:
   :undoc-members:

.. autoclass:: ollama_classifier.backends.base.LLMBackend
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: ollama_classifier.backends.ollama.OllamaBackend
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: ollama_classifier.backends.vllm.VLLMBackend
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: ollama_classifier.backends.sglang.SGLangBackend
   :members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: ollama_classifier.backends.llamacpp.LlamaCppBackend
   :members:
   :show-inheritance:
   :special-members: __init__

Method Summary
--------------

:class:`LLMClassifier` exposes the following methods. All return a
:class:`~ollama_classifier.types.ClassificationResult`.

+----------------------------------------------------+------------------------+----------------------------------------------------------+
| Method                                             | Async                  | Description                                              |
+====================================================+========================+==========================================================+
| ``generate(text, choices, system_prompt, *,        | ``agenerate``          | Hierarchical constrained generation (1 to ``max_calls``  |
| max_calls)``                                       |                        | calls). ``method="adaptive_generate"``.                 |
+----------------------------------------------------+------------------------+----------------------------------------------------------+
| ``classify(text, choices, system_prompt)``         | ``aclassify``          | Exact multi-call completion scoring (N calls for N       |
|                                                    |                        | labels). ``method="multi_call"``.                        |
+----------------------------------------------------+------------------------+----------------------------------------------------------+
| ``batch_generate(texts, choices, system_prompt,    | ``abatch_generate``    | Batch hierarchical generation (parallelized).            |
| *, max_calls)``                                    |                        |                                                          |
+----------------------------------------------------+------------------------+----------------------------------------------------------+
| ``batch_classify(texts, choices, system_prompt)``  | ``abatch_classify``    | Batch multi-call classification (parallelized).          |
+----------------------------------------------------+------------------------+----------------------------------------------------------+

Choosing a Scoring Method
-------------------------

+------------------------------------------+--------------------------------------------------+
| Use Case                                 | Recommended Method                               |
+==========================================+==================================================+
| Speed is critical, approximate OK        | ``generate(max_calls=1)``                        |
+------------------------------------------+--------------------------------------------------+
| Adaptive accuracy within a budget        | ``generate(max_calls=K)``                        |
+------------------------------------------+--------------------------------------------------+
| Fully resolved, unlimited calls          | ``generate(max_calls=None)``                     |
+------------------------------------------+--------------------------------------------------+
| Gold-standard confidence (exact)         | ``classify``                                     |
+------------------------------------------+--------------------------------------------------+
| Batch processing                         | ``batch_classify`` / ``batch_generate``          |
+------------------------------------------+--------------------------------------------------+
| Concurrent processing                    | Async variants (``aclassify``, ``agenerate``)    |
+------------------------------------------+--------------------------------------------------+
