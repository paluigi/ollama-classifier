Inference Backends
==================

ollama-classifier supports four inference backends, all behind a single
unified :class:`~ollama_classifier.classifier.LLMClassifier`. Create a
backend and wrap it in a classifier:

.. code-block:: python

   from ollama_classifier import LLMClassifier
   from ollama_classifier.backends import OllamaBackend  # or VLLMBackend, etc.

   backend = OllamaBackend(model="llama3.2")
   classifier = LLMClassifier(backend)

All backends communicate with their inference engine via HTTP. ``OllamaBackend``
uses the native Ollama Python SDK; ``VLLMBackend``, ``SGLangBackend``, and
``LlamaCppBackend`` communicate via the `OpenAI-compatible chat completions API
<https://platform.openai.com/docs/api-reference/chat>`__.

Each backend translates the high-level ``constrain_labels`` parameter to its
native constraint mechanism so the classifier code stays identical regardless
of engine.

.. contents::
   :local:
   :depth: 1

Ollama
------

Backend wrapping the Ollama runtime (≥v0.12) via the official Python SDK.

.. important::

   ``OllamaBackend`` requires **Ollama runtime v0.12 or later** for logprobs
   support. The ``ollama`` Python SDK is an optional dependency — install it
   with ``pip install "ollama-classifier[ollama]"``.

Constraint mechanism: **JSON Schema enum** via the ``format`` parameter. The
model generates ``{"label": "<chosen_label>"}``; structural JSON tokens
(``{``, ``"label"``, ``:``, ``"``, ``}``) are filtered during trie
reconstruction and completion scoring.

Modern Ollama runtimes removed the ``/api/tokenize`` endpoint and do not
support fill-in-the-middle ("insert") on instruct models. ``OllamaBackend``
therefore obtains both label tokenization and completion scores through
**empirical forced constrained generation**: it forces a label as the only
valid choice in a ``chat()`` call and reads back the model's genuine
per-token logprobs. Tokenization results are memoized per label to amortize
the cost.

Because Ollama wraps labels in JSON, its
:attr:`~ollama_classifier.backends.base.LLMBackend.supports_bare_label_constraint`
property is ``False``.

.. code-block:: python

   from ollama_classifier import LLMClassifier
   from ollama_classifier.backends import OllamaBackend

   # Local (defaults to http://localhost:11434)
   backend = OllamaBackend(model="llama3.2")

   # Remote
   backend = OllamaBackend(model="llama3.2", host="http://remote-host:11434")

   classifier = LLMClassifier(backend)

vLLM
----

High-throughput serving engine for LLMs. Supports guided decoding and
logprobs out of the box.

Constraint mechanism: **``guided_choice``** — constrains the model to generate
exactly one of the provided label strings, as **bare label text** (no JSON
wrapper). Logprobs are pre-mask (raw model logits before guided-decoding
masking). ``supports_bare_label_constraint`` is ``True``.

**Local server:**

.. code-block:: bash

   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3.2-3B-Instruct \
       --host 0.0.0.0 --port 8000

**Connect:**

.. code-block:: python

   from ollama_classifier.backends import VLLMBackend

   backend = VLLMBackend(
       model="meta-llama/Llama-3.2-3B-Instruct",
       base_url="http://localhost:8000/v1",
   )

**Remote server:**

.. code-block:: python

   backend = VLLMBackend(
       model="your-model",
       base_url="https://your-vllm-server.com/v1",
       api_key="your-api-key",  # if auth is required
   )

SGLang
------

Fast serving system for large language models with efficient radix attention.

Constraint mechanism: **regex** — builds a regex from the escaped labels
(``("label1|label2|...")``) so the model generates **bare label text** with no
JSON wrapper. Logprobs are pre-mask (raw model logits before regex masking).
``supports_bare_label_constraint`` is ``True``.

**Local server:**

.. code-block:: bash

   python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.2-3B-Instruct \
       --host 0.0.0.0 --port 30000

**Connect:**

.. code-block:: python

   from ollama_classifier.backends import SGLangBackend

   backend = SGLangBackend(
       model="meta-llama/Llama-3.2-3B-Instruct",
       base_url="http://localhost:30000/v1",
   )

llama.cpp
---------

Lightweight inference via ``llama-server``. Ideal for CPU or mixed
CPU/GPU environments.

Constraint mechanism: **GBNF grammar** — builds a grammar rule that allows
exactly one of the provided labels::

   root ::= "label1" | "label2" | "label3"

This generates **bare label text** with no JSON wrapper, so logprob
reconstruction is clean. ``llama-server`` accepts a non-standard ``grammar``
field on the ``/v1/chat/completions`` endpoint (``response_format`` with JSON
schema is buggy in llama.cpp). ``supports_bare_label_constraint`` is ``True``.

**Local server:**

.. code-block:: bash

   ./llama-server -m model.gguf --host 0.0.0.0 --port 8080 -c 4096

**Connect:**

.. code-block:: python

   from ollama_classifier.backends import LlamaCppBackend

   backend = LlamaCppBackend(
       model="model",
       base_url="http://localhost:8080/v1",
   )

.. note::

   Logprobs and grammar constraints require llama.cpp to be compiled
   with the appropriate flags (e.g. ``LLAMA_SUPPORT_LOGPROBS``).

Constraint Mechanisms Summary
-----------------------------

Each backend translates the high-level ``constrain_labels`` parameter to its
native constraint mechanism:

+----------------------+------------------------+--------------------------+-----------------------------+
| Backend              | Constraint mechanism   | Output format            | ``supports_bare_label_``    |
|                     |                        |                          | ``constraint``              |
+======================+========================+==========================+=============================+
| ``OllamaBackend``    | JSON Schema enum       | ``{"label": "..."}``     | ``False``                   |
|                      | (``format``)           |                          |                             |
+----------------------+------------------------+--------------------------+-----------------------------+
| ``VLLMBackend``      | ``guided_choice``      | bare label text          | ``True``                    |
+----------------------+------------------------+--------------------------+-----------------------------+
| ``SGLangBackend``    | ``regex``              | bare label text          | ``True``                    |
+----------------------+------------------------+--------------------------+-----------------------------+
| ``LlamaCppBackend``  | GBNF ``grammar``       | bare label text          | ``True``                    |
+----------------------+------------------------+--------------------------+-----------------------------+

``supports_bare_label_constraint``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each backend exposes a
:attr:`~ollama_classifier.backends.base.LLMBackend.supports_bare_label_constraint`
property that tells the classifier how to tokenize labels for trie
construction:

- **``True``** (vLLM, SGLang, llama.cpp): ``chat()`` generates bare label
  text with no wrapper. Labels are tokenized standalone via the engine's
  ``/tokenize`` endpoint.
- **``False``** (Ollama): ``chat()`` wraps labels in JSON. Labels are
  tokenized through forced constrained generation (forcing the label as the
  only valid choice) so the resulting tokens match the actual response
  tokens exactly.

This is handled automatically by ``LLMClassifier`` — you do not need to
inspect it directly.

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
| ``api_key``  | ``"not-needed"`` | API key for authentication (not used by Ollama)   |
+--------------+------------------+---------------------------------------------------+
| ``timeout``  | ``120.0``        | Request timeout in seconds                        |
+--------------+------------------+---------------------------------------------------+
| ``max_tokens`` | ``256``        | Maximum tokens to generate                        |
+--------------+------------------+---------------------------------------------------+
| ``extra_body`` | ``{}``         | Extra parameters merged into every request body   |
+--------------+------------------+---------------------------------------------------+

``OllamaBackend`` additionally accepts a ``host`` parameter (defaults to
``http://localhost:11434``) and accepts pre-initialized ``sync_client`` /
``async_client`` instances for testing or connection reuse.

Switching Backends
------------------

The ``LLMClassifier`` exposes the **same API** regardless of which backend
you use, making it trivial to switch between engines:

.. code-block:: python

   from ollama_classifier.backends import (
       OllamaBackend, VLLMBackend, SGLangBackend, LlamaCppBackend,
   )
   from ollama_classifier import LLMClassifier

   # Switch just by changing the backend
   backends = [
       OllamaBackend(model="llama3.2"),
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
