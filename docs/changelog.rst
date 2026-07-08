Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.5.0] - 2026-07-07
--------------------

**Behavior change:** ``score()`` now performs constrained generation (teacher
forcing) instead of a no-generation forward pass. This is a minor version bump
because the ``score()`` / ``classify()`` contract and per-call cost change.

Both ``OllamaBackend`` and ``SGLangBackend`` were rewritten to use empirical
**forced constrained generation** for tokenization and completion scoring. This
fixes compatibility with modern Ollama (removed ``/api/tokenize``, no insert
support) and SGLang (tokenization mismatch between standalone BPE and
constrained generation).

Fixed
~~~~~

- ``OllamaBackend.tokenize()`` no longer calls the removed ``client.tokenize``
  (``AttributeError``). ``SGLangBackend.tokenize()`` no longer sends the wrong
  field name (``"text"`` vs the API's ``"prompt"``). Both now use forced
  constrained generation. Results are memoized per label.
- ``OllamaBackend.score()`` no longer uses ``client.generate(suffix=...)``
  (``does not support insert`` HTTP 400 on instruct models).
  ``SGLangBackend.score()`` no longer relies on the broken ``/tokenize``
  endpoint and spurious-token ``echo`` approach. Both now force the candidate
  label as the single valid choice via ``chat()`` and extract the model's
  genuine per-token logprobs.
- Async variants (``atokenize()``, ``ascore()``) updated to match for both
  backends.
- ``score()`` / ``ascore()`` now raise ``RuntimeError`` when forced generation
  yields no value tokens (previously returned empty logprobs silently, which
  ``classify()`` treated as ``-inf``).

Changed
~~~~~~~

- **Behavior change:** ``LLMBackend.score()`` base contract updated —
  ``score()`` now performs teacher-forced constrained generation, not a
  no-generation forward pass. Per-call cost increases (one full generation per
  label instead of a prefill).
- ``Token.id`` from ``tokenize()`` is now always ``-1`` (empirical tokens have
  no stable server-side ID). Downstream consumers should not rely on
  ``Token.id`` for Ollama/SGLang backends.
- ``OllamaBackend`` and ``SGLangBackend`` module/class docstrings updated to
  document the forced-generation mechanism.
- ``SGLangBackend._render_prompt()``, ``_tokenize_count()``, and
  ``_atokenize_count()`` removed (only used by the old ``score()``).

Added
~~~~~

- ``OllamaBackend._label_token_logprobs()`` — extracts label-value tokens from
  a ``{"label": "..."}`` response via char-offset span mapping.
- ``SGLangBackend._label_token_logprobs()`` — filters special/EOS tokens from
  bare-label responses. Expanded ``_SPECIAL_TOKENS`` set covers Llama-3, Phi,
  and Qwen EOS markers.
- ``tests/test_ollama_backend.py`` — unit tests for the Ollama helper (no
  server required).

[0.4.1] - 2026-07-06
--------------------

Changed
~~~~~~~

- Updated project metadata

[0.4.0] - 2025-07-06
--------------------

A major redesign that unifies the two previous classifier classes
(``OllamaClassifier`` and ``LLMClassifier``) into a single backend-agnostic
``LLMClassifier`` backed by a unified ``LLMBackend`` ABC, and introduces two
distinct confidence scoring methods. **This release contains breaking
changes.**

Added
~~~~~

- ``OllamaBackend`` — inference backend for the Ollama runtime (≥v0.12),
  using the native Ollama SDK behind the unified ``LLMBackend`` interface
- Adaptive ``generate()`` with budget-controlled ``max_calls``: ``1`` for
  fast/approximate, ``K`` for adaptive resolution, ``None`` for exact. Makes
  1 to ``max_calls`` constrained API calls and reconstructs per-label
  logprobs from a prefix trie over label tokens
- ``coverage`` (per-label fraction of tokens scored) and ``n_calls`` (API
  calls made) fields on ``ClassificationResult``
- ``method`` field on ``ClassificationResult`` (``"adaptive_generate"`` or
  ``"multi_call"``) and ``approximate`` flag
- ``ollama_classifier.scoring`` module with geometric-mean normalization,
  prefix-trie, divergence-aware scoring, and cluster resolution
- ``supports_bare_label_constraint`` property on ``LLMBackend`` to drive
  context-dependent tokenization (bare labels vs. JSON-wrapped labels)
- Comprehensive test suite (52 tests, including integration tests with
  mocked backends)
- Parallelized batch methods (sync via thread pool, async via
  ``asyncio.gather``)

Changed
~~~~~~~

- **BREAKING:** ``generate()`` now returns a ``ClassificationResult`` (was
  a ``str`` in 0.3.0)
- **BREAKING:** Unified into a single ``LLMClassifier`` class. The old
  ``OllamaClassifier`` (from ``classifier.py``) and the old
  ``LLMClassifier`` (from ``llm_classifier.py``) are replaced by one
  ``LLMClassifier`` that accepts any ``LLMBackend``
- **BREAKING:** ``classify()`` rewritten: now uses multi-call completion
  scoring with geometric-mean normalization, eliminating the concentration
  (token-count) bias present in earlier versions. Makes N calls for N labels
- ``ClassificationResult`` is now a Pydantic ``BaseModel`` (was a dataclass)
- ``httpx`` and ``pydantic`` are now **required** dependencies (were optional
  under the ``[backends]`` extra)
- ``ollama`` Python SDK is now an **optional** dependency, installed via the
  ``[ollama]`` extra (only required when using ``OllamaBackend``)
- All backends now communicate via HTTP using the OpenAI-compatible API and
  implement ``chat()``, ``score()``, ``tokenize()``, and their async variants

Fixed
~~~~~

- Concentration bias (token-count bias) in confidence scoring — fixed by
  applying geometric-mean normalization across both ``generate()`` and
  ``classify()``
- Silent ``0.0`` logprob fallback for unscored tokens — unscored tokens now
  use ``-inf`` so they are correctly handled by softmax
- Code duplication between ``OllamaClassifier`` and the old
  ``LLMClassifier`` — eliminated by the unified architecture

Removed
~~~~~~~

- ``OllamaClassifier`` class (replaced by ``LLMClassifier`` + ``OllamaBackend``)
- Old ``LLMClassifier`` from ``llm_classifier.py`` (replaced by the unified
  ``LLMClassifier`` in ``classifier.py``)
- ``[backends]`` optional dependency group (``httpx`` is now a core dependency)

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
