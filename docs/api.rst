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
----------------

.. autoclass:: ollama_classifier.classifier.OllamaClassifier
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Method Summary
--------------

+-----------------------------------------------+------------------------+---------------------------------------------+
| Method                                        | Async                  | Description                                 |
+===============================================+========================+=============================================+
| ``generate(text, choices, system_prompt)``    | ``agenerate``          | Constrained output only (fastest)           |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``score_fast(text, choices, system_prompt)``  | ``ascore_fast``        | Single-call logprob extraction              |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``score_complete(...)``                       | ``ascore_complete``    | Multi-call evaluation with softmax          |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``classify(text, choices, system_prompt)``    | ``aclassify``          | Generate + score_fast                       |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``classify_complete(...)``                    | ``aclassify_complete`` | Generate + score_complete                   |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``batch_generate(texts, choices, ...)``       | ``abatch_generate``    | Batch constrained output                    |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``batch_score_fast(texts, choices, ...)``     | ``abatch_score_fast``  | Batch fast scoring                          |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``batch_score_complete(texts, ...)``          | ``abatch_score_complete`` | Batch complete scoring                   |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``batch_classify(texts, choices, ...)``       | ``abatch_classify``    | Batch classify (fast)                       |
+-----------------------------------------------+------------------------+---------------------------------------------+
| ``batch_classify_complete(texts, ...)``       | ``abatch_classify_complete`` | Batch classify (complete)              |
+-----------------------------------------------+------------------------+---------------------------------------------+

Choosing a Method
-----------------

+------------------------------------------+--------------------------------------------------+
| Use Case                                 | Recommended Method                               |
+==========================================+==================================================+
| Speed is critical, no confidence needed  | ``generate``                                     |
+------------------------------------------+--------------------------------------------------+
| Speed with confidence scores             | ``classify`` / ``score_fast``                    |
+------------------------------------------+--------------------------------------------------+
| Accurate confidence scores               | ``classify_complete`` / ``score_complete``       |
+------------------------------------------+--------------------------------------------------+
| Batch processing                         | ``batch_classify`` or ``batch_classify_complete``|
+------------------------------------------+--------------------------------------------------+
| Concurrent processing                    | Async variants (``aclassify``, etc.)             |
+------------------------------------------+--------------------------------------------------+