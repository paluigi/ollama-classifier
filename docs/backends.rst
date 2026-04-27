Inference Backends
==================

ollama-classifier supports four inference backends. The original
``OllamaClassifier`` uses the Ollama Python SDK directly.  For all other
backends, use ``LLMClassifier`` with the appropriate backend class.

All non-Ollama backends communicate with their inference servers via the
`OpenAI-compatible chat completions API
<https://platform.openai.com/docs/api-reference/chat>`__.

.. contents::
   :local:
   :depth: 1

Ollama
------

The default backend. Requires the Ollama runtime installed locally.

.. code-block:: python

   from ollama import Client
   from ollama_classifier import OllamaClassifier

   classifier = OllamaClassifier(Client(), model="llama3.2")

vLLM
----

High-throughput serving engine for LLMs. Supports guided decoding and
logprobs out of the box.

**Local server:**

.. code-block:: bash

   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3.2-3B-Instruct \
       --host 0.0.0.0 --port 8000

**Connect:**

.. code-block:: python

   from ollama_classifier.backends import VLLMBackend
   from ollama_classifier import LLMClassifier

   backend = VLLMBackend(
       model="meta-llama/Llama-3.2-3B-Instruct",
       base_url="http://localhost:8000/v1",
   )
   classifier = LLMClassifier(backend)

**Remote server:**

.. code-block:: python

   backend = VLLMBackend(
       model="your-model",
       base_url="https://your-vllm-server.com/v1",
       api_key="your-api-key",  # if auth is required
   )
   classifier = LLMClassifier(backend)

SGLang
------

Fast serving system for large language models with efficient radix attention.

**Local server:**

.. code-block:: bash

   python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.2-3B-Instruct \
       --host 0.0.0.0 --port 30000

**Connect:**

.. code-block:: python

   from ollama_classifier.backends import SGLangBackend
   from ollama_classifier import LLMClassifier

   backend = SGLangBackend(
       model="meta-llama/Llama-3.2-3B-Instruct",
       base_url="http://localhost:30000/v1",
   )
   classifier = LLMClassifier(backend)

llama.cpp
---------

Lightweight inference via ``llama-server``. Ideal for CPU or mixed
CPU/GPU environments.

**Local server:**

.. code-block:: bash

   ./llama-server -m model.gguf --host 0.0.0.0 --port 8080 -c 4096

**Connect:**

.. code-block:: python

   from ollama_classifier.backends import LlamaCppBackend
   from ollama_classifier import LLMClassifier

   backend = LlamaCppBackend(
       model="model",
       base_url="http://localhost:8080/v1",
   )
   classifier = LLMClassifier(backend)

.. note::

   JSON schema constraints and logprobs require llama.cpp to be compiled
   with the appropriate flags (``LLAMA_JSON_SCHEMA`` and
   ``LLAMA_SUPPORT_LOGPROBS``).

Backend Configuration
---------------------

All backends share common configuration options:

+--------------+------------------+---------------------------------------------------+
| Parameter    | Default          | Description                                       |
+==============+==================+===================================================+
| ``model``    | *(required)*     | Model identifier                                  |
+--------------+------------------+---------------------------------------------------+
| ``base_url`` | Engine-specific  | Base URL of the inference server                  |
+--------------+------------------+---------------------------------------------------+
| ``api_key``  | ``"not-needed"`` | API key for authentication                        |
+--------------+------------------+---------------------------------------------------+
| ``timeout``  | ``120.0``        | Request timeout in seconds                        |
+--------------+------------------+---------------------------------------------------+
| ``max_tokens`` | ``256``        | Maximum tokens to generate                        |
+--------------+------------------+---------------------------------------------------+
| ``extra_body`` | ``{}``         | Extra parameters merged into every request body   |
+--------------+------------------+---------------------------------------------------+

Switching Backends
------------------

The ``LLMClassifier`` exposes the **same API** regardless of which backend
you use, making it trivial to switch between engines:

.. code-block:: python

   from ollama_classifier.backends import VLLMBackend, SGLangBackend, LlamaCppBackend
   from ollama_classifier import LLMClassifier

   # Switch just by changing the backend
   backends = [
       VLLMBackend(model="my-model", base_url="http://localhost:8000/v1"),
       SGLangBackend(model="my-model", base_url="http://localhost:30000/v1"),
       LlamaCppBackend(model="my-model", base_url="http://localhost:8080/v1"),
   ]

   for backend in backends:
       classifier = LLMClassifier(backend)
       result = classifier.classify(
           text="Hello world!",
           choices=["a", "b", "c"],
       )
       print(f"{backend.__class__.__name__}: {result.prediction}")
