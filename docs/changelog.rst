Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.6.0] - 2026-07-13
--------------------

**Behavior change:** ``generate()`` cluster resolution was rewritten to use
**hierarchical reproportioning** instead of mixing logprobs from different
constraint contexts. This fixes a critical bug where increasing
``max_calls`` could *decrease* classification accuracy.

Fixed
~~~~~

- **Critical:** ``generate()`` with ``max_calls > 1`` could produce *worse*
  predictions than ``max_calls=1``. Supplementary constrained calls (to
  resolve label clusters) changed the constraint set, placing their logprobs
  in a different probability space. Mixing these into the geometric mean
  corrupted the score ranking — post-mask logprobs (≈0.0) inflated the
  scores of labels with many unscored tokens. In the paper benchmark,
  accuracy dropped monotonically from 73.8% (``mc=1``) to 50.9% (``mc=8``)
  for the "names only" configuration.

  **Fix:** Supplementary calls now only **reproportion** probability mass
  *within* a cluster of labels, never changing between-group totals. The
  cluster's total probability (from the initial call) is redistributed
  among its members using softmax of geometric-mean scores from the subset
  call. This guarantees accuracy never degrades with increasing
  ``max_calls``.

Changed
~~~~~~~

- ``generate()`` and ``agenerate()`` rewritten with the reproportion
  approach. The BFS cluster-resolution loop is retained, but supplementary
  calls only resolve multi-label clusters (≥2 labels). Single-label
  clusters with partial coverage are skipped — their probability is already
  fixed by the between-group distribution, and no reproportioning call
  would change it.
- ``max_calls=1`` now means "no cluster resolution" (single call, purely
  divergence-based scoring from the initial constrained call).
- The ``raw_response`` dict now always includes ``step_logprobs`` and
  ``scored_lengths`` for both sync and async paths.
- Module and method docstrings updated to reflect the hierarchical
  reproportion algorithm.

Added
~~~~~

- ``tests/test_classifier.py::TestMaxCallsMonotonicity`` — three regression
  tests verifying that: (1) increasing ``max_calls`` never flips a correct
  prediction, (2) between-group probability mass is preserved during
  reproportioning, and (3) single-token labels require no resolution calls.

[0.5.0] - 2026-07-11
--------------------

**Behavior change:** All four backends were rewritten with a unified
architecture: ``tokenize()`` uses empirical forced constrained generation
across all backends, and ``score()`` uses echo/prefill (vLLM, SGLang) or
forced constrained generation (Ollama, llama.cpp) depending on server
capabilities. This is a minor version bump because the ``score()`` /
``classify()`` contract and per-call cost change.

Fixed
~~~~~

- ``OllamaBackend.tokenize()`` no longer calls the removed ``client.tokenize``
  (``AttributeError``). All backends' ``tokenize()`` now use forced constrained
  generation so token boundaries match actual constrained-generation output.
  Results are memoized per label.
- ``OllamaBackend.score()`` no longer uses ``client.generate(suffix=...)``
  (``does not support insert`` HTTP 400 on instruct models). Now uses forced
  constrained generation via JSON Schema enum.
- ``SGLangBackend.score()`` rewritten to use echo/prefill (``/v1/completions``
  with ``echo=True``) with the correct ``"prompt"`` field in ``/tokenize`` for
  boundary detection. Produces differentiated confidence for ``classify()``
  (was near-uniform with forced generation due to prompt-priming).
- ``SGLangBackend.tokenize()`` no longer sends the wrong field name (``"text"``
  vs the API's ``"prompt"``). Now uses forced constrained generation via regex.
- ``VLLMBackend`` constraint updated from deprecated ``guided_choice`` (removed
  in vLLM v0.12.0) to ``structured_outputs.choice``. ``score()`` rewritten to
  use echo/prefill; ``tokenize()`` uses forced constrained generation.
- ``LlamaCppBackend.score()`` rewritten from broken ``suffix``-based completions
  to forced GBNF grammar generation (llama.cpp does not support ``echo=True``
  on the completions endpoint).
- All backends: ``score()`` / ``ascore()`` now raise ``RuntimeError`` when no
  value tokens are returned (previously returned empty logprobs silently).
- All backends: HTTP client pooling added (reusable ``httpx.Client`` /
  ``AsyncClient``) to avoid per-call connection setup.
- ``VLLMBackend.supports_bare_label_constraint`` changed from ``False`` to
  ``True`` (``structured_outputs.choice`` generates bare label text).

Changed
~~~~~~~

- **Behavior change:** ``LLMBackend.score()`` base contract updated —
  ``score()`` now uses echo/prefill (vLLM, SGLang) or forced constrained
  generation (Ollama, llama.cpp), not a no-generation forward pass.
- ``Token.id`` from ``tokenize()`` is now always ``-1`` (empirical tokens have
  no stable server-side ID). Downstream consumers should not rely on
  ``Token.id``.
- All backend module/class docstrings updated to document the scoring and
  tokenization mechanisms.
- ``LlamaCppBackend._render_prompt()``, ``_find_completion_start()``, and
  ``_tokenize_ids()`` removed (only used by the old ``score()``).

Added
~~~~~

- ``OllamaBackend._label_token_logprobs()`` — extracts label-value tokens from
  a ``{"label": "..."}`` response via char-offset span mapping.
- ``SGLangBackend._label_token_logprobs()`` and ``VLLMBackend._label_token_logprobs()``
  — filter special/EOS tokens from bare-label responses.
  ``_SPECIAL_TOKENS`` frozenset covers Llama-3, Phi, and Qwen EOS markers.
- ``LlamaCppBackend._label_token_logprobs()`` — same special-token filter.
- ``SGLangBackend._render_prompt()``, ``_tokenize_count()``, ``_atokenize_count()``
  — reintroduced for the echo/prefill ``score()``.
- ``VLLMBackend._render_prompt()``, ``_tokenize_count()``, ``_atokenize_count()``
  — same echo/prefill helpers.
- ``tests/test_ollama_backend.py`` — unit tests for the Ollama helper (no
  server required).
- ``local_tests/`` — integration test infrastructure with dataset evaluation
  and CSV output for all four backends (Ollama, SGLang, vLLM, llama.cpp).

[0.4.1] - 2026-07-06
--------------------
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
