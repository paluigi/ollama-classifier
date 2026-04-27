Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[0.3.0] - 2025-04-27
--------------------

Added
~~~~~

- ``LLMClassifier`` — a generic, backend-agnostic classifier that works with any inference engine
- ``VLLMBackend`` — inference backend for vLLM (local and remote)
- ``SGLangBackend`` — inference backend for SGLang (local and remote)
- ``LlamaCppBackend`` — inference backend for llama.cpp server (local and remote)
- ``ollama_classifier.backends`` package with ``LLMBackend`` abstract base class
- ``[backends]`` optional dependency group (``httpx``) for non-Ollama engines
- ``docs/backends.rst`` — dedicated documentation page for inference backends

Changed
~~~~~~~

- Bumped package version to 0.3.0

[0.1.0] - 2024-01-01
--------------------

Added
~~~~~

- Initial release
- ``OllamaClassifier`` class with sync and async methods
- Constrained output generation using JSON schema
- Two scoring methods: fast (single-call) and complete (multi-call with softmax)
- Batch processing support
- Support for simple labels and labels with descriptions
- Custom system prompt support
- ``ClassificationResult`` dataclass for structured results
