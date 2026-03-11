Installation
============

Package Installation
--------------------

Install the package using pip:

.. code-block:: bash

   pip install ollama-classifier

Or with uv:

.. code-block:: bash

   uv add ollama-classifier

Prerequisites
-------------

Before using ollama-classifier, you need to have:

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

Development Installation
------------------------

To contribute to the project or run the documentation locally:

.. code-block:: bash

   git clone https://github.com/paluigi/ollama-classifier.git
   cd ollama-classifier
   uv sync --extra docs

Building Documentation Locally
------------------------------

After installing with the docs extra:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

Then open ``_build/html/index.html`` in your browser.