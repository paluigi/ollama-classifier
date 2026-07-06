Installation
============

Package Installation
--------------------

Install the package:

.. code-block:: bash

   pip install ollama-classifier

Or with uv:

.. code-block:: bash

   uv add ollama-classifier

``httpx`` and ``pydantic`` are required dependencies and are installed
automatically. The ``ollama`` Python SDK is **optional** — install it only
if you use the Ollama backend:

.. code-block:: bash

   pip install "ollama-classifier[ollama]"

The vLLM, SGLang, and llama.cpp backends communicate over HTTP using
``httpx`` (already a core dependency), so no extra install is needed for
them.

Prerequisites
-------------

Before using ollama-classifier, you need at least one inference backend:

**Ollama backend**

1. **Ollama ≥0.12 installed and running**

   Download and install Ollama from: https://ollama.com/download

   .. note::

      Ollama runtime **v0.12 or later** is required for logprobs support.

2. **The Ollama Python SDK** (optional dependency):

   .. code-block:: bash

      pip install "ollama-classifier[ollama]"

3. **A model pulled**

   Pull a model to use for classification:

   .. code-block:: bash

      ollama pull llama3.2

   You can use any model that supports JSON structured output. Recommended models:

   - ``llama3.2`` - Fast and capable
   - ``llama3.1`` - Larger and more capable
   - ``mistral`` - Good balance of speed and quality
   - ``qwen2.5`` - Excellent for classification tasks

**vLLM backend**

Install vLLM: https://docs.vllm.ai/en/latest/getting_started/installation.html

Start a local server:

.. code-block:: bash

   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3.2-3B-Instruct \
       --host 0.0.0.0 --port 8000

**SGLang backend**

Install SGLang: https://docs.sglang.ai/start/install.html

Start a local server:

.. code-block:: bash

   python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.2-3B-Instruct \
       --host 0.0.0.0 --port 30000

**llama.cpp backend**

Download or build ``llama-server``: https://github.com/ggerganov/llama.cpp

Start a local server:

.. code-block:: bash

   ./llama-server -m model.gguf --host 0.0.0.0 --port 8080 -c 4096

Development Installation
------------------------

To contribute to the project or run the documentation locally:

.. code-block:: bash

   git clone https://github.com/paluigi/ollama-classifier.git
   cd ollama-classifier
   uv sync --extra docs --extra ollama

Building Documentation Locally
------------------------------

After installing with the docs extra:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

Then open ``_build/html/index.html`` in your browser.
