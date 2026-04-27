Installation
============

Package Installation
--------------------

Install the core package (Ollama backend only):

.. code-block:: bash

   pip install ollama-classifier

Or with uv:

.. code-block:: bash

   uv add ollama-classifier

Install with additional backends (vLLM, SGLang, llama.cpp):

.. code-block:: bash

   pip install "ollama-classifier[backends]"

Or with uv:

.. code-block:: bash

   uv add "ollama-classifier[backends]"

Prerequisites
-------------

Before using ollama-classifier, you need at least one inference backend:

**Ollama backend**

1. **Ollama installed and running**

   Download and install Ollama from: https://ollama.com/download

2. **A model pulled**

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
   uv sync --extra docs --extra backends

Building Documentation Locally
------------------------------

After installing with the docs extra:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

Then open ``_build/html/index.html`` in your browser.
