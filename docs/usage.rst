Usage
=====

This section covers the common usage patterns for ollama-classifier.

Basic Classification
--------------------

The simplest way to classify text:

.. code-block:: python

   from ollama import Client
   from ollama_classifier import OllamaClassifier

   client = Client()
   classifier = OllamaClassifier(client, model="llama3.2")

   result = classifier.classify(
       text="The goalkeeper made an incredible save!",
       choices=["sports", "politics", "technology", "entertainment"]
   )

   print(f"Prediction: {result.prediction}")
   print(f"Confidence: {result.confidence:.2%}")

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
       choices=choices
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

Scoring Methods
---------------

Fast Scoring (Single API Call)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use when you need confidence scores but speed is important:

.. code-block:: python

   result = classifier.score_fast(
       text="The movie was fantastic!",
       choices=["positive", "negative", "neutral"]
   )

Complete Scoring (Multi-Call with Softmax)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

More accurate but makes N API calls for N choices:

.. code-block:: python

   result = classifier.score_complete(
       text="The movie was fantastic!",
       choices=["positive", "negative", "neutral"]
   )

Generate Only (Fastest)
-----------------------

When you only need the prediction without confidence scores:

.. code-block:: python

   prediction = classifier.generate(
       text="The team won the championship!",
       choices=["sports", "finance", "politics"]
   )

Batch Classification
--------------------

Classify multiple texts efficiently:

.. code-block:: python

   texts = [
       "The goalkeeper made an incredible save!",
       "The central bank raised interest rates.",
       "The new smartphone features a revolutionary camera.",
   ]

   # Fast batch classification
   results = classifier.batch_classify(
       texts=texts,
       choices=["sports", "finance", "technology"]
   )

   # Complete batch classification (more accurate)
   results = classifier.batch_classify_complete(
       texts=texts,
       choices=["sports", "finance", "technology"]
   )

   for text, result in zip(texts, results):
       print(f"{text} -> {result.prediction} ({result.confidence:.2%})")

Async Usage
-----------

For concurrent processing, use the async methods:

.. code-block:: python

   import asyncio
   from ollama import AsyncClient
   from ollama_classifier import OllamaClassifier

   async def main():
       client = AsyncClient()
       classifier = OllamaClassifier(client, model="llama3.2")
       
       # Single classification
       result = await classifier.aclassify(
           text="The concert was amazing!",
           choices=["positive", "negative", "neutral"]
       )
       
       # Batch classification (concurrent)
       results = await classifier.abatch_classify(
           texts=["Text 1", "Text 2", "Text 3"],
           choices=["positive", "negative", "neutral"]
       )

   asyncio.run(main())

Working with Results
--------------------

The :class:`~ollama_classifier.types.ClassificationResult` object contains:

.. code-block:: python

   result = classifier.classify(
       text="I love this product!",
       choices=["positive", "negative", "neutral"]
   )

   # The predicted label
   print(result.prediction)  # "positive"

   # Confidence score (0.0 to 1.0)
   print(result.confidence)  # 0.92

   # Full probability distribution
   print(result.probabilities)  
   # {"positive": 0.92, "negative": 0.05, "neutral": 0.03}

   # Raw response for debugging
   print(result.raw_response)