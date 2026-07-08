# Plan: Switch SGLang score() to echo/prefill (restore differentiated classify)

## Context (verified via diagnostic + live server)

The diagnostic (`diagnose_sglang_logprobs.py`) proved that:
- **Echo/prefill** (`/v1/completions` with `echo=True`) recovers the model's genuine per-label logprobs: entropy 0.625, sports 68% / politics 32% / others ~0%.
- **Forced generation** (current `score()`) compresses all labels to near-zero logprobs: entropy 1.385 ≈ uniform.
- The root cause is **prompt-priming** (the regex constraint tells the model "you must output this label"), NOT post-mask logprobs (logprobs are confirmed pre-mask).

The echo approach was the original SGLang implementation before the forced-generation rewrite. It was abandoned due to three bugs (wrong tokenize field name `"text"` vs `"prompt"`, spurious generated token, silent exception masking). All three are now understood and fixable.

## Scope

- `src/ollama_classifier/backends/sglang.py` — rewrite `score()`/`ascore()` to echo/prefill; reintroduce `_render_prompt()` and `_tokenize_count()`; keep `tokenize()`/`atokenize()` as forced generation.
- `tests/test_local_sglang.py` — restore prediction-specific assertions (matching Ollama tests); remove flat-logprob notes.
- No changes to `ollama.py` (Ollama has no echo support; its forced-generation approach stays).

---

## Tasks (ordered)

### 1. Reintroduce `_render_prompt()` (sglang.py)

Same implementation as vLLM (generic chat markers, model-agnostic):
```python
def _render_prompt(self, messages: List[ChatMessage]) -> str:
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"<|system|>\n{m.content}")
        elif m.role == "user":
            parts.append(f"<|user|>\n{m.content}")
    return "\n\n".join(parts) + "\n\n<|assistant|>\n"
```

### 2. Reintroduce `_tokenize_count()` (sync + async)

Uses `/tokenize` with the correct **`"prompt"`** field (NOT `"text"`). Returns `data["count"]` or `len(data["tokens"])`. **No try/except** — let HTTP errors propagate.

```python
def _tokenize_count(self, text: str) -> int:
    url = f"{self._base_url}/tokenize"
    body = {"model": self._model, "prompt": text}
    resp = self._get_sync_http().post(url, headers=self._build_headers(), json=body)
    resp.raise_for_status()
    return resp.json().get("count", 0)
```

Async `_atokenize_count()` mirrors with `await self._get_async_http()`.

### 3. Rewrite `score()` to echo/prefill (sglang.py)

```python
def score(self, messages, completion):
    prompt = self._render_prompt(messages)
    prompt_with_completion = prompt + completion

    prompt_len = self._tokenize_count(prompt)
    total_len = self._tokenize_count(prompt_with_completion)

    body = {
        "model": self._model,
        "prompt": prompt_with_completion,
        "echo": True,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 1,  # NOT 0 — SGLang 500s on logprobs=0
        **self._extra_body,
    }
    resp = self._get_sync_http().post(url, headers=..., json=body)
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    all_logprobs = choice.get("logprobs", {})
    tokens_list = all_logprobs.get("tokens", [])
    token_lps_list = all_logprobs.get("token_logprobs", [])
    top_lps_list = all_logprobs.get("top_logprobs", [])

    # Extract ONLY label tokens [prompt_len:total_len] — discard spurious gen token
    completion_tokens = tokens_list[prompt_len:total_len]
    completion_lps = token_lps_list[prompt_len:total_len]
    completion_top = top_lps_list[prompt_len:total_len]

    token_logprobs = []
    for i, tok in enumerate(completion_tokens):
        top = {}
        if i < len(completion_top) and completion_top[i]:
            for t, lp in completion_top[i].items():
                top[t] = lp
        lp = completion_lps[i] if i < len(completion_lps) else 0.0
        token_logprobs.append(TokenLogprob(token=tok, logprob=lp or 0.0, top_logprobs=top))

    if not token_logprobs:
        raise RuntimeError(f"score({completion!r}): echo returned no label tokens")
    return ScoringResponse(completion=completion, logprobs=token_logprobs, raw=data)
```

### 4. Rewrite `ascore()` to async echo/prefill

Same logic as `score()` but using `await self._get_async_http()` and `await client.post(...)`.

### 5. Keep `tokenize()`/`atokenize()` as forced generation

No change. The trie still needs empirical tokens that match constrained-generation paths. Forced generation is the only way to get those.

`_label_token_logprobs()` and `_SPECIAL_TOKENS` stay — they're used by `tokenize()` only now.

### 6. Update module/class docstrings (sglang.py)

- Module docstring: clarify that `score()` uses echo/prefill while `tokenize()` uses forced generation.
- `score()`/`ascore()` docstrings: document echo/prefill rationale.
- Remove stale references to "both use forced constrained generation."

### 7. Restore test assertions in `test_local_sglang.py`

With echo/prefill, `classify()` should now produce differentiated confidence. Restore the assertions to match `test_local_ollama.py`:

- `test_classify_basic`: restore `assert result.prediction == "technology"`
- `test_classify_with_descriptions`: restore `assert result.prediction in ("positive", "mixed")`
- `test_classify_custom_prompt`: restore `assert result.prediction == "bullish"`
- `test_generate_exact_unlimited`: restore `assert result.prediction == "science"`
- `test_aclassify`/`test_batch_classify`: add prediction-specific assertions where the text is unambiguous
- Remove all "near-uniform confidence" / "AWQ model too compliant" comments

**Caution:** These assertions assume the AWQ 3B model produces the same predictions with echo as the Ollama model. The implementer must run the tests and adjust if the model disagrees on a specific example. If any prediction differs, loosen to a reasonable alternative set rather than asserting `prediction in labels`.

---

## Validation

1. `uv run python -m pytest tests/test_classifier.py tests/test_scoring.py tests/test_ollama_backend.py -q` — existing unit tests must stay green.
2. `uv run python -m pytest tests/test_local_sglang.py -s` — all 10 tests should pass with differentiated confidence (not flat). Verify `classify()` confidence is > 50% for clear cases.
3. `uv run ruff check src/ollama_classifier/backends/sglang.py` — lint clean.
4. Cross-check: run the diagnostic script to verify echo entropy is still low (~0.6) after the code changes.

## Risks

- **`_render_prompt` uses generic markers** (`<|system|>`) not Qwen's native `<|im_start|>`. The diagnostic used `<|im_start|>` and got entropy 0.625. The generic markers may produce slightly different (but still differentiated) logprobs. If entropy is still flat with generic markers, switch to Qwen-style markers. Test empirically.
- **`/tokenize` field name**: confirmed `"prompt"` works, `"text"` returns 400. Must not revert to `"text"`.
- **`logprobs: 0`** causes SGLang HTTP 500. Must use `logprobs: 1`.
- **AWQ model accuracy**: the 3B model may predict differently than Ollama's model on some examples. Test assertions must be verified empirically.

## Out of scope
- Ollama backend (no echo support; forced generation stays).
- vLLM backend (already has echo approach; has the same `_tokenize_ids` except-swallowing bug but no running server to test).
- Refactoring `_render_prompt`/`_tokenize_count` into the base class (duplication flagged in code review but deferred).
