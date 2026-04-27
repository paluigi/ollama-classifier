API Reference
=============

This section provides the complete API reference for ollama-classifier.

ClassificationResult
--------------------

.. autoclass:: ollama_classifier.types.ClassificationResult
   :members:
   :undoc-members:
   :show-inheritance:

ChoicesType
-----------

.. autodata:: ollama_classifier.types.ChoicesType

OllamaClassifier
-----------------

.. autoclass:: ollama_classifier.classifier.OllamaClassifier
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

LLMClassifier
-------------

The generic, backend-agnostic classifier. Accepts any
:class:`~ollama_classifier.backends.base.LLMBackend` instance and
exposes the same API as :class:`OllamaClassifier`.

.. autoclass:: ollama_classifier.llm_classifier.LLMClassifier
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Backends
--------

.. automodule:: ollama_classifier.backends
   :members:
   :undoc-members:

.. autoclass:: ollama_classifier.backends.base.LLMBackend
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

Both :class:`OllamaClassifier` and :class:`LLMClassifier` expose the
same method set:

+-----------------------------------------------+------------------------+---------------------------------------------+
| Method                                        | Async                  | Description                                 |
+===============================================+========================+=============================================+
| ``generate(text, choices, system_prompt)``    | ``agenerate``          | Constrained output only (fastest)           |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``score(text, choices, system_prompt)``       | ``ascore``             | Multi-call evaluation with softmax          |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``classify(text, choices, system_prompt)``    | ``aclassify``          | Full classification with confidence scores  |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``batch_generate(texts, choices, ...)``       | ``abatch_generate``    | Batch constrained output                    |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``batch_score(texts, choices, ...)``          | ``abatch_score``       | Batch scoring                               |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``batch_classify(texts, choices, ...)``       | ``abatch_classify``    | Batch classification                        |
+-----------------------------------------------+------------------------+---------------------------------------------+

Choosing a Method
-----------------

+------------------------------------------+--------------------------------------------------+
| Use Case                                 | Recommended Method                               |
+==========================================+==================================================+
| Speed is critical, no confidence needed  | ``generate``                                     |
+------------------------------------------+--------------------------------------------------+
| Accurate confidence scores               | ``classify`` / ``score``                         |
+------------------------------------------+--------------------------------------------------+
| Batch processing                         | ``batch_classify`` or ``batch_score``            |
+------------------------------------------+--------------------------------------------------+
| Concurrent processing                    | Async variants (``aclassify``, etc.)             |
+------------------------------------------+--------------------------------------------------+
